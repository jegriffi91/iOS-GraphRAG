# `tools/` — operator utilities

Utilities the project owner runs ad-hoc against real repos. Distinct from
`benchmarks/run.py` (which measures throughput/latency) and from the
indexer/server (which are part of the shipped product).

---

## `parse_audit.py` — Phase 4.5a parse-error audit

`parse_audit.py` walks a repository, parses every `.swift` / `.h` / `.m`
file with the **same tree-sitter grammars the indexer uses**
(`tree-sitter-swift`, `tree-sitter-objc`), and reports per-file `ERROR`
and `MISSING` node counts plus aggregate stats. Its output drives the
Phase 4.5b/c/d strategy decision:

| % files with any parse error | Recommended strategy |
| --- | --- |
| `< 5%`   | Strategy A — tree-sitter primary, hardened fallback. |
| `5–20%`  | Strategy A + B — tree-sitter primary, SourceKitten validator in CI. |
| `>= 20%` | Strategy C — consider SourceKit-LSP as primary. |

The boundary handling is conservative: `5.0%` is A+B and `20.0%` triggers
Strategy C.

### Usage

```bash
# Audit a directory; JSON lands in benchmarks/results/.
uv run python tools/parse_audit.py --repo /path/to/repo --output benchmarks/results/

# Limit to specific extensions.
uv run python tools/parse_audit.py --repo /path/to/repo --extensions .swift,.h

# Tune top-N most error-prone files in the report (default 20).
uv run python tools/parse_audit.py --repo /path/to/repo --top-n 50
```

The tool exits `0` regardless of the error rate — the audit reporting
"20% files have errors" is the audit working correctly, not a failure.

### JSON output

Filename: `parse-audit-<orchestrator-sha>-<YYYYMMDD-HHMMSS>.json`.

```json
{
  "metadata": {
    "git_sha": "4e1fc5a",
    "timestamp": "2026-05-07T13:07:08Z",
    "repo_path": "/abs/path/to/repo",
    "tree_sitter_version": "0.25.2",
    "tree_sitter_swift_version": "0.7.2",
    "tree_sitter_objc_version": "3.0.2",
    "extensions_audited": [".swift", ".h", ".m"],
    "platform": {"system": "Darwin", "machine": "arm64", "release": "..."},
    "python_version": "3.14.3",
    "skip_dirs": [".build", ".git", "Carthage", "DerivedData", "Pods", "node_modules", "vendor"]
  },
  "summary": {
    "total_files": 1234,
    "files_with_has_error": 45,
    "files_with_has_error_percent": 3.65,
    "files_with_any_error_node": 50,
    "files_with_any_error_node_percent": 4.05,
    "histogram_error_node_counts": {"0": 1184, "1-5": 30, "6-20": 15, "21-100": 4, "100+": 1},
    "by_extension": {
      ".swift": {"total": 1000, "files_with_error": 35, "percent": 3.5},
      ".h":     {"total": 100,  "files_with_error": 5,  "percent": 5.0},
      ".m":     {"total": 134,  "files_with_error": 10, "percent": 7.46}
    },
    "timeouts": 0,
    "skipped": 0,
    "decision_recommendation": "Strategy A (tree-sitter primary, hardened fallback)"
  },
  "top_files": [
    {"path": "...", "error_node_count": 234, "missing_node_count": 12,
     "total_nodes": 5678, "has_error": true, "size_bytes": 45000}
  ],
  "files": [
    {"path": "...", "extension": ".swift", "size_bytes": 289,
     "has_error": false, "error_node_count": 0, "missing_node_count": 0,
     "total_nodes": 107, "timeout": false, "skipped_reason": null}
  ]
}
```

The `summary` object is the small slice consumers usually want. `top_files`
holds the N worst offenders sorted by `error_node_count` then `size_bytes`
descending. `files` is the full per-file table — keep it for downstream
slicing without re-running the audit.

### Console summary

```
=== Parse-error audit ===
  Repo: /path/to/repo
  Files audited: 1234
  Files with any parse error: 50 (4.1%)
  Decision: Strategy A (tree-sitter primary, hardened fallback)

  By extension:
    .swift  3.5% error rate (35/1000)
    .h      5.0% error rate (5/100)
    .m      7.5% error rate (10/134)

  Top 5 error-prone files:
    Some/File.swift             234 ERROR nodes
    Other/File.swift             89 ERROR nodes
    ...

  Full JSON: benchmarks/results/parse-audit-...-...json
```

Console output goes to stderr so the JSON path on stdout-redirect pipelines
stays usable.

### Recommended workflow

Per the roadmap (Phase 4.5a), run progressively wider:

1. **10k-LOC slice** of the real repo. Confirms the tool runs cleanly,
   gives a fast first read on whether SwiftUI/macro patterns are a real
   issue. Wall-clock: seconds.
2. **100k-LOC slice.** Detects local hot-spots (e.g. one feature module
   that uses heavy macros).
3. **Full ~1M-LOC repo.** The number that drives the Phase 4.5b/c/d
   decision. Wall-clock: a minute or two on a multi-core dev machine.

Compare the percentage between slices. If the 10k slice is 0% but the
full repo is 25%, the macros/SwiftUI patterns are concentrated in
specific modules — the per-extension and `top_files` sections of the
JSON tell you which ones.

### Reproducibility

The `metadata` block records the orchestrator git SHA and the exact
versions of `tree-sitter`, `tree-sitter-swift`, and `tree-sitter-objc`
used. Two runs against the same repo with the same pinned grammars
will produce identical error counts (the only variable would be a file
ordering difference, which we sort out in `discover_files`).

If you need to compare results across grammar versions, pin the grammar,
run, then bump and re-run — the metadata difference will be visible in
the JSON. Don't compare runs that used different grammar versions
without acknowledging that.

### Known limitations

- Per-file parse timeout is 30 s. A pathological file that exceeds this
  is recorded with `timeout: true` and counted toward `total_files` but
  not toward error rates. If timeouts are common, treat that as its own
  parser-strategy signal — it suggests grammar pathology, not just
  recoverable error rate.
- The directory walk excludes `.build`, `Pods`, `Carthage`,
  `DerivedData`, `vendor`, `.git`, and `node_modules`. It does **not**
  parse `.gitignore`. If your repo has unusual ignore patterns that
  matter, audit a pre-filtered subtree.
- The audit does NOT apply the indexer's `should_index` name filters
  (`*Tests*`, `Mock*`, `*.generated.swift`). The audit is measuring
  parser robustness against the actual file population, not the
  indexer's filtered subset — measuring only the indexer's filtered
  subset would understate exposure to SwiftUI/macro patterns that
  often live in tests/generated code.
