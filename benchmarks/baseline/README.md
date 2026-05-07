# `benchmarks/baseline/` — committed reference run

This directory holds **fixture-bound** baseline JSONs that the nightly CI
workflow (`.github/workflows/nightly-benchmark.yml`) compares against to
catch pipeline regressions. They are NOT real-world performance numbers
and they will not match what you see on a developer laptop, a `c5.xlarge`
runner, or a production indexing host.

## What's in here

| File | Producer | What it captures |
| --- | --- | --- |
| `full_index.json` | `benchmarks/run.py` | Full + (skipped) incremental indexing wall time, peak RSS, server cold start, in-process p50/p95/p99 latencies for `semantic_search` + `trace_dependencies`, embedding-matrix bytes, node/edge counts. |
| `parse_audit.json` | `tools/parse_audit.py` | Per-file `has_error` / ERROR-node / MISSING-node counts plus aggregate stats and the Phase 4.5a Strategy A/B/C recommendation. |

Both were produced against `test_fixtures/CalculatorApp` (14 source files,
~165 indexed symbols) and pinned to the orchestrator git SHA recorded in
their `metadata.git_sha` field.

## What these baselines represent

The CalculatorApp fixture is a synthetic, hand-curated test scaffold —
not a real iOS app. Its absolute numbers are noisy because:

- Wall times are dominated by CPython import + tree-sitter grammar load
  rather than by actual parsing/indexing work (the fixture is too small
  for the work-heavy phases to swamp the constants).
- Peak RSS is dominated by the embedding model's footprint, which is
  fixture-independent.
- Latency p99s see a long tail from a handful of cold calls; with only
  14 files / 165 symbols the percentile sample is small.

**What these baselines DO catch:**

- A pipeline-level regression — e.g. the indexer's parse phase suddenly
  takes 3x longer because someone reintroduced a per-file model load,
  or a new resolver pass blows up edge construction time.
- A correctness regression in the parser — e.g. a tree-sitter grammar
  bump that newly trips ERROR nodes on patterns the fixture exercises
  (Swift 5+ macros, SwiftUI patterns, ObjC bridging).
- A latency regression in `semantic_search` — e.g. an HNSW migration
  that's slower than the existing `np.argpartition` for top-k.

**What they DON'T catch:**

- Real-world indexing scale issues (1M-LOC repos behave differently).
- Memory pressure on a real codebase — the fixture's embedding matrix
  is ~500 KB; the real one is ~1.5 GB.
- Cold-start latency under realistic model-cache contention.

The CI ratios (regression % vs baseline) are the signal; absolute numbers
are not directly comparable to production targets in
`docs/PRODUCTION_ROADMAP.md`.

## When to regenerate

Regenerate after any **intentional** performance change:

- Phase 4d.2 HNSW landing (semantic_search latency baseline shifts).
- Phase 4c watcher daemon (incremental sections start populating once
  the fixture grows enough git history).
- Tree-sitter grammar bumps (parse errors may shift, mostly harmlessly).
- Major embedding-pipeline changes (e.g. switching the underlying model
  or batching strategy).

Regeneration is a deliberate act, not an automatic one. Treat the
baseline files as policy artifacts: a PR that bumps them needs reviewer
sign-off because it relaxes the regression check's floor.

## Why we commit these (despite hardware-dependence)

CI needs *something* to compare against. The alternatives we rejected:

- **No regression check at all** — the nightly workflow becomes a
  monitoring artifact only, with no signal that prevents a regression
  from landing on `main`.
- **Compare to last night's run** — drift accumulates without bound;
  a 1% slowdown per night is invisible but compounds to ~30% in a month.
- **Compare to a dynamically-fetched baseline (e.g. last successful
  build)** — fragile, depends on artifact retention, and means a single
  flaky run can poison subsequent comparisons.

A committed baseline gives the regression check a fixed reference point.
The check tolerances (in `benchmarks/check_regression.py`) are loosened
to absorb fixture-noise, so the gate triggers only on egregious
regressions. Micro-optimization wins that show up here are nice-to-have
but not the primary signal.

## How to regenerate

Two commands. Both must run from the repo root in a uv-managed venv:

```bash
# Phase 4a benchmark (full indexer + latency distributions).
uv run python benchmarks/run.py \
    --repo test_fixtures/CalculatorApp \
    --output /tmp/bl-out/

# Phase 4.5a parse audit.
uv run python tools/parse_audit.py \
    --repo test_fixtures/CalculatorApp \
    --output /tmp/bl-out/

# Promote the freshly-generated outputs to the committed baseline.
cp /tmp/bl-out/*-*.json benchmarks/baseline/full_index.json
cp /tmp/bl-out/parse-audit-*.json benchmarks/baseline/parse_audit.json
```

The two harnesses produce filenames containing the orchestrator SHA and a
timestamp; we rename to stable filenames here so the regression check can
reference them by a fixed path.

After regeneration, commit the JSONs with a message that says *why* the
baseline shifted (e.g. `bench: regenerate baseline after HNSW landing`)
so future readers can correlate baseline movement with intentional
performance work.

## Reading the JSONs by hand

Both files are pretty-printed (`json.dump(..., indent=2, sort_keys=True)`).
The fields the regression check reads are:

- `full_index.json`:
  - `full_index.wall_seconds` — full indexer pass wall time.
  - `semantic_search_latency_ms.p99` — 99th-percentile per-call latency.
- `parse_audit.json`:
  - `summary.files_with_any_error_node_percent` — fraction of audited
    files whose tree-sitter parse produced any ERROR / MISSING node or
    set `root.has_error`.

Other fields are recorded for diagnostic value. They are not gated, so a
shift in `peak_rss_bytes` or `phases.embed_seconds` won't fail CI on its
own; you'll see it in the artifact upload, not in the regression-check
status.
