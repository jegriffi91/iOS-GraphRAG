# iOS-GraphRAG — Known Limitations

This document is intentionally narrow and honest. It captures the resolution
limits introduced (or made deterministic) in Phase 1, plus a couple of
crash-isolation cases that the indexer cannot fully recover from. It is not
aspirational — every entry below is observable on the current codebase.

## L1 — Cross-module same-named symbols may resolve to the wrong target

**Symptom.** When two or more declarations across different files share a
symbol name (or a Swift selector), the indexer's name and selector maps
keep exactly one node id per key. Edge resolution (`INHERITS`, `CONFORMS`,
`EXTENDS`, `CALLS`, plus the late `update_unresolved_edges` pass) always
picks the first candidate in `(file_path, start_byte, id)` order. That
choice has no relationship to the import graph or Swift module boundaries
of the calling site. If `Foo.calculate(...)` exists in both `Calculator.swift`
and `ScientificCalculator.swift`, every cross-file `CALLS calculate(_:...)`
edge resolves to `Calculator.swift` (the alphabetically earlier file),
even when the source code's import context would have made the
`ScientificCalculator` overload the only correct target.

**Where this happens.**

- `src/ios_graphrag/indexer.py` `resolve_all_symbol_ids` — the SELECT is
  sorted by `(symbol_name, file_path, start_byte, id)` and the first
  candidate per name wins. Edge consumers in `build_edges_for_file`
  (`INHERITS`, `CONFORMS`, `EXTENDS` paths) pick `target_ids[0]`.
- `src/ios_graphrag/indexer.py` `resolve_selector_ids` — the SELECT is
  sorted by `(selector, file_path, start_byte, id)` and the first
  candidate per selector wins; later candidates are recorded as a
  collision warning and dropped from the map.
- `src/ios_graphrag/indexer.py` `update_unresolved_edges` — uses the same
  `name_map`, so this pass inherits the same first-wins behavior.
- `src/ios_graphrag/indexer.py` `build_edges_for_file` (`CALLS` branch)
  — falls back from `selector_map` to `name_map` when the selector is
  not present, again picking `target_ids[0]`.

**What you'll see in the logs.** Every collision is recorded as a
`WARNING` line in `indexing_errors.log` with each candidate's
`(file_path, line_number)`. Running the indexer against
`test_fixtures/CalculatorApp` produces 4 name collisions and 2 selector
collisions on a clean build. Expect this number to grow with codebase
size; an audit of the warnings is the recommended way to discover where
disambiguation matters.

**Mitigation roadmap.**

- Phase 1 (here): make the picked candidate deterministic so two indexer
  runs over the same tree produce byte-identical output, and surface
  every collision via the existing logger.
- Phase 4 P4.5d (planned): the SwiftUI enrichment work introduces
  per-symbol module attribution. That same plumbing makes it feasible to
  disambiguate by Swift module name at edge-build time. Out of scope
  for Phase 1.

## L2 — Per-file crash isolation is best-effort, not absolute

`index_repository` wraps each file's parse → write → edge-build lifecycle
in a `try/except` (P1.4). A malformed file logs an `ERROR` line, the
file is skipped, and the rest of the index continues. The end-of-run
summary reports `N indexed, M failed`.

Cases where isolation does NOT recover cleanly:

- **Hash workers and parse workers run in `ProcessPoolExecutor`.** A
  crash in a worker (e.g., a Tree-sitter segfault on hostile input)
  kills that worker, and the executor logs a per-future error. The main
  process keeps running; the file is treated as failed and counted
  toward `M`.
- **The embedding step is batched across all files.** If the embedding
  worker crashes mid-batch, the symptoms are not isolated to one file.
  Phase 4 will revisit this with per-file embedding error handling.

In `tests/test_crash_isolation.py` we explicitly verify the
parse-and-store path against four malformed inputs (binary garbage,
empty, UTF-8 BOM, very long single line). Any new mode of failure that
turns up in production should be added there as a regression case.
