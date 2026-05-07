# Changelog

All notable changes to iOS-GraphRAG are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Planned
- Phase 4b: in-process embedding model load (eliminate per-incremental subprocess reload).
- Phase 4c: indexer daemon + file watcher (`ios-graphrag-watch`).
- Phase 4.5b/c/d: parser strategy + SwiftUI symbol enrichment.
- Phase 5: coverage gaps (properties, SwiftUI views, property wrappers, init/deinit, trailing closures, enum cases).
- Phase 6g: optional telemetry endpoint (owner decision pending).

## [0.2.0] — 2026-05-07

The hardening release. Lays the foundation for SLA-grade production
deployment to the internal iOS team. Highlights:

### Added
- **`ios-graphrag-preflight`** — 7-check installer sanity script (`--smoke` mode for CI).
- **`ios-graphrag-doctor`** — 8-diagnostic health check; `--bug-report` / `--verify` / `--tail-errors` flags.
- **TLS controls** — `--cert-bundle PATH` for corporate CA bundles; `GRAPHRAG_INSECURE_TLS=1` opt-in bypass (default off).
- **Offline model staging** — `IOS_GRAPHRAG_MODEL_DIR` and `~/.cache/ios-graphrag/models/` overrides; `huggingface-cli download` workflow documented.
- **Schema versioning** (Phase 6a) — `engine/database/migrations/`, runner, server-side version mismatch refusal.
- **Edge dedup + integrity checks** (Phase 6c) — `idx_edges_unique`, end-of-run integrity logging, `doctor --verify`.
- **Structured logging** (Phase 6b) — JSON file output at `~/Library/Logs/ios-graphrag/`, rotation 50 MB x 5, UUID4 trace IDs on tool calls.
- **Crash diagnostics** (Phase 6d) — automatic crashdump on unhandled server exception, recent-calls ring buffer, `docs/REPORTING.md`.
- **Benchmark harness** (Phase 4a) — `benchmarks/run.py` for per-phase timing, latency percentiles, memory snapshots.
- **Parse-error audit** (Phase 4.5a) — `tools/parse_audit.py` produces Strategy A/A+B/C recommendation.
- **CI** (Phase 6e) — GitHub Actions on every PR (ruff + pytest + preflight --smoke), nightly benchmark with artifact upload.
- **Hot-path improvements** (Phase 4d.1, 4f) — `np.argpartition` for top-k semantic search; SQL-pushed bridging filter; mtime-based GRAPH auto-reload to eliminate cache drift.

### Changed
- Repo restructured to src-layout (`src/ios_graphrag/`).
- All deps pinned to minor-version ranges; `uv.lock` committed.
- Default behavior: TLS verification ON (was unconditional bypass at module import).
- Tool descriptions toned down (no more all-caps directives).
- `upstream` / `downstream` semantics in `swift_dependency_tracer` aligned with LSP convention (`upstream` = dependencies, `downstream` = dependents).

### Fixed
- Per-file crash isolation in indexer; one bad file no longer aborts the run.
- Resolution determinism: `name_map` and `selector_map` now sort tied candidates deterministically; collisions are logged.
- Removed `print()` from indexer hot path (was risk of corrupting MCP stdio).
- Server schema mismatch now exits cleanly with remediation pointer.

### Security
- Module-level SSL bypass code removed; consolidated into `_tls.py` and gated behind explicit env var.

### Removed
- Stale prototype files (`engine/indexer.py`, `engine/setup_lab.py`, `engine/lab/verify.py`, `engine/test_ts.py`, `engine/PRODUCTION_HARDENING_PROMPT.md`).
- 1311 build artifacts (`test_fixtures/CalculatorApp/.build/`) untracked from the repo.

## [0.1.0] — pre-Phase-0

Initial prototype. See `git log --before=2026-05-06 --oneline` for details.
