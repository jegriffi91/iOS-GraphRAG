# iOS-GraphRAG Benchmark Harness

Phase 4a measurement scaffold. Runs against any repo path and produces a
single JSON file in `benchmarks/results/<git-sha>-<YYYYMMDD-HHMMSS>.json`
that captures full + incremental indexing, server cold start, and
in-process latency distributions for the two hot tools.

The numbers from this harness are the precondition for Phase 4b-f
architecture decisions (HNSW vs `np.argpartition` for vector search,
daemon vs per-call indexer, mmap-backed embedding matrix). Per the
production roadmap (`docs/PRODUCTION_ROADMAP.md`, sections 1.3 and 3),
the projected scale numbers (~300-500k symbols, 2.5-3 GB DB,
~1.5 GB embedding matrix) are speculative until measured against the
real 1M-LOC repo.

## Usage

```bash
# Default: temp DB, results in benchmarks/results/
uv run python benchmarks/run.py --repo /path/to/repo

# Custom DB path (kept after the run for inspection)
uv run python benchmarks/run.py --repo /path/to/repo --db /tmp/bench.db --output benchmarks/results/

# Smaller query corpus for a quick sanity check
uv run python benchmarks/run.py --repo /path/to/repo --n-queries 50
```

The harness prints the path to the produced JSON on stdout (everything
else goes to stderr). The exit code is 0 on success, non-zero only when
something catastrophic happens; individual sections that can't run record
`{"skipped": true, "reason": "..."}` rather than crashing.

## What it measures

The harness runs six measurement sections sequentially. Each maps to one
key in the output JSON:

| Section            | Output key                        | What it measures                                                                                  |
|--------------------|------------------------------------|---------------------------------------------------------------------------------------------------|
| 1. Full index      | `full_index`                       | Wall time, peak RSS, per-phase breakdown, symbol/edge/DB-size counts of a `--full` indexer run.   |
| 2. Incremental     | `incremental.{10,100,1000}_files`  | Same metrics for incremental reindexes that simulate 10/100/1000-file deltas via `git checkout`.  |
| 3. Server cold start | `server_cold_start`              | Time from `subprocess.Popen` of `ios-graphrag-server` until it emits the `READY` token on stderr. |
| 4. Semantic search latency | `semantic_search_latency_ms` | p50/p95/p99/min/max/mean over 1000 random `_semantic_search(query, top_k=5)` calls.            |
| 5. Trace dependencies latency | `trace_dependencies_latency_ms` | Same shape, over 1000 random `_trace_dependencies(file_path)` calls.                       |
| 6. Memory snapshot | `memory`                           | Embedding matrix size in bytes; node + resolved-edge counts.                                      |

The per-phase breakdown in section 1 (`full_index.phases`) is populated by
parsing `PHASE_START`/`PHASE_END` markers that the indexer emits at DEBUG
level around each pipeline stage: hash, parse, sql_write, weave, embed.
The harness sets `GRAPHRAG_LOG_LEVEL=DEBUG` so those markers show up in
the indexer's stderr; INFO-level usage is unaffected.

## Output JSON shape

```json
{
  "metadata": {
    "git_sha": "<orchestrator HEAD short sha>",
    "repo_git_sha": "<the indexed repo's HEAD short sha>",
    "timestamp": "2026-05-06T20:00:00Z",
    "repo_path": "/abs/path/to/repo",
    "loc": 367,
    "platform": {"system": "Darwin", "machine": "arm64"},
    "python_version": "3.13.0",
    "torch_version": "2.5.1"
  },
  "full_index": {
    "wall_seconds": 17.7,
    "peak_rss_bytes": 1629913088,
    "phases": {
      "hash_seconds": 3.6,
      "parse_seconds": 3.6,
      "sql_write_seconds": 0.004,
      "weave_seconds": 0.004,
      "embed_seconds": 7.9
    },
    "symbol_count": 56,
    "edge_count": 45,
    "db_size_bytes": 319488
  },
  "incremental": {
    "10_files":   {"wall_seconds": 0.5, "peak_rss_bytes": ..., "phases": {...}, "files_changed": 10},
    "100_files":  {"skipped": true, "reason": "insufficient git history for 100-file delta"},
    "1000_files": {"skipped": true, "reason": "..."}
  },
  "server_cold_start": {
    "spawn_to_ready_seconds": 2.45,
    "spawn_to_first_result_seconds": 2.45
  },
  "semantic_search_latency_ms": {
    "p50": 6.85, "p95": 18.2, "p99": 20.7,
    "min": 6.5, "max": 22.5, "mean": 8.2, "n": 1000
  },
  "trace_dependencies_latency_ms": {
    "p50": 0.12, "p95": 0.35, "p99": 0.41,
    "min": 0.11, "max": 0.43, "mean": 0.17, "n": 1000
  },
  "memory": {
    "embedding_matrix_bytes": 1500000000,
    "graph_nodes": 350000,
    "graph_edges": 1200000
  }
}
```

### Notes on individual fields

- `peak_rss_bytes` is normalized to bytes regardless of OS. macOS reports
  `getrusage().ru_maxrss` in bytes; Linux reports kilobytes. The harness
  branches on `sys.platform` to handle both.
- `server_cold_start.spawn_to_first_result_seconds` is currently equal to
  `spawn_to_ready_seconds` because the server is ready to handle a tool
  call the moment it's ready to read stdin. A true end-to-end handshake
  (initialize + tools/list over JSON-RPC) would add <10ms — out of scope
  for this harness; ask if you want it.
- `semantic_search_latency_ms.n` may be smaller than the requested query
  count if the index has fewer unique symbols than requested. On
  `test_fixtures/CalculatorApp` you'll see ~56 (the symbol count). On the
  real repo, expect 1000.
- `trace_dependencies_latency_ms.n` tracks distinct file paths in the
  index, capped at the requested query count. CalculatorApp has 7 files,
  so n=7 there.
- `incremental.<size>.skipped` will be `true` on repos with insufficient
  history (the harness needs `git diff HEAD~k..HEAD` to produce at least
  N changed files for some k) or on non-git repos. This is normal and
  not a failure.

## Defensive behaviour around the user's working tree

The incremental section mutates the repo via `git checkout HEAD~1 -- <files>`
to stage realistic deltas. To avoid corrupting the user's in-flight work:

1. Before any incremental run, the harness inspects `git status --porcelain`.
   If anything is dirty, it `git stash push -u`-es (untracked files
   included) so the working tree is clean for the duration.
2. After each delta measurement, it `git checkout HEAD -- <files>`-es to
   restore those files.
3. After the entire incremental section, it `git stash pop`s to restore
   the user's in-flight work.

If `git stash pop` fails (e.g. because of a conflict the harness can't
auto-resolve), the work is still safe in the stash list — `git stash list`
shows it and `git stash apply` recovers it. This is the only place the
harness ever touches the working tree, and it never runs against a repo
without first verifying it IS a git repo.

If you want to run the harness against a repo that has an in-flight
half-staged commit, pass it through anyway — the stash will save you.

## Sample run on the test fixture

```bash
uv run python benchmarks/run.py --repo test_fixtures/CalculatorApp
```

Takes ~30-60 seconds depending on whether the embedding model is cached.
Produces a JSON like the one above with `symbol_count=56`, `edge_count=45`,
and incremental sections all marked skipped (CalculatorApp has only one
commit's worth of history).
