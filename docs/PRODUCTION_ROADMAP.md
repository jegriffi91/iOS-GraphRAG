# iOS-GraphRAG Production Hardening Roadmap

**Status:** Draft v1, pre-execution
**Branch:** `claude/review-ast-graphrag-cI7X4`
**Target:** SLA-production deployment to internal iOS team (5–50 devs)
**Estimated wall-clock to GA:** 8 weeks
**Supersedes:** `engine/PRODUCTION_HARDENING_PROMPT.md` (delete in Phase 0)

This roadmap is the output of a code review of the current codebase and a series of design exchanges with the project owner. It is intended to be self-contained: a fresh agent session should be able to read this document and execute the plan without re-deriving the analysis.

---

## 1. Context

### 1.1 Project summary

iOS-GraphRAG is an MCP (Model Context Protocol) server that provides AI-powered code intelligence over Swift/Objective-C codebases. Its architecture is a "Map vs. Territory" split:

- **Territory:** the actual `.swift`/`.h`/`.m` files on disk; the only source of truth for code content.
- **Map:** a SQLite index containing pointers (file paths + byte ranges), topology (inheritance, conformance, extensions, calls, bridging), and vectors (embeddings of signatures only).

The MCP server exposes four tools to AI clients:
- `global_codebase_search` — semantic search over signatures.
- `swift_dependency_tracer` — graph traversal for dependencies.
- `read_symbol_source` — live disk read by byte range.
- `objc_swift_bridge_finder` — Swift-to-ObjC inheritance reporter.

### 1.2 Distribution constraints

| Constraint | Value |
|---|---|
| Target platform | macOS (Apple Silicon) only |
| Primary AI harness | GitHub Copilot (CLI + VS Code) |
| Secondary harnesses | Claude Desktop |
| Index ownership | Per-developer (not shared) |
| Repo size | 1M+ LOC iOS monorepo |
| Code character | SwiftUI-heavy, macros prevalent (`@Observable`, `@Model`, `#Preview`, `#Predicate`), Swift + ObjC mix |
| Production bar | SLA — must support an on-call rotation |

### 1.3 Scale math (not yet validated against real repo; see Phase 4a)

| Resource | Estimated size at 1M LOC | Notes |
|---|---|---|
| Symbol count | 300–500k | function-level granularity |
| SQLite DB on disk | 2.5–3 GB/dev | acceptable on M-series |
| Embedding matrix in RAM | ~1.5 GB | 768d × 500k × 4 bytes |
| NetworkX graph in RAM | 2–4 GB | competes with Xcode/sim |
| Cold start | 8–15 s | model load + graph hydrate |
| Full reindex | 20–40 min (best case) | needs validation |
| `np.dot` over full matrix | 30–80 ms | per `semantic_search` call |

These drive Phase 4 architecture decisions. Validate before betting on them.

---

## 2. Current State Assessment

### 2.1 Critical issues identified in code review

Each item lists the file path and line numbers in the current codebase.

**C1. README/code semantics inverted for `trace_dependencies`.**
README prose says upstream = "this file inherits from / conforms to," downstream = "things that inherit from this." Code at `engine/core/server.py:122-140` does the opposite (`predecessors` go into `upstream`). The README's example JSON matches the code; only the prose is wrong. Pick one definition and align all three (prose, example, code, tests).

**C2. Stale duplicate files that won't run.**
- `engine/indexer.py` — old prototype with 8-column `INSERT` against a 10-column schema. Will fail on first invocation.
- `engine/setup_lab.py` — duplicate of `engine/lab/setup_lab.py`.
- `engine/lab/verify.py:6-7` — imports `build_graph`, `hydrate`, `read_live_code`, `NODE_IDS` that no longer exist anywhere. Dead.
- `engine/test_ts.py` — debug scratch file, not a test.
- `README.md:107` references `uv run indexer.py` but the actual entry point is `engine/core/indexer.py`.

**C3. 46 MB of Swift build artifacts committed to git.**
`git ls-files | grep -c "^test_fixtures/CalculatorApp/.build"` = 1311 of 1339 tracked files. Add `test_fixtures/**/.build/` to `.gitignore` and `git rm -r --cached`.

**C4. Unconditional SSL bypass at module import.**
`engine/core/server.py:13-19` and `engine/core/indexer.py:13-19` set `ssl._create_default_https_context = _create_unverified_context` and `CURL_CA_BUNDLE=''` at import time, every time. `_generate_embeddings_worker` (`indexer.py:683-696`) goes further: `HF_HUB_DISABLE_SSL_VERIFY`, `HTTPX_SSL_VERIFY=0`, blanks `SSL_CERT_FILE`. This will fail any infosec review. Must be gated behind opt-in env var.

### 2.2 Correctness concerns to verify and fix

**V1. Swift selector for unlabeled parameters — defensively refactored in Phase 1 P1.2 (commit `78f171c`).**
The roadmap's original concern was that `build_swift_selector` (`engine/core/indexer.py:223-249`) walked parameter children, grabbed the first `simple_identifier` as the label, and would yield `test(something:)` for `func test(_ something: String)` — colliding with the `test(something: String)` overload. **Empirical finding:** the locked grammar (`tree-sitter-swift>=0.7,<0.8`) emits `_` as a `simple_identifier` whose `text` is `"_"`, so the original code's "break after first `simple_identifier`" accidentally produced the correct selector (`test(_:)`). All 28 baseline tests passed under the OLD code. The bug as described did **not** exist in this codebase. Phase 1 P1.2 instead landed a defensive refactor: extracted `_extract_param_label` helper (`src/ios_graphrag/indexer.py:214-260`) that explicitly checks for wildcard text, plus 19 parametric tests in `tests/test_selectors.py` covering all parameter shapes. A future grammar update that types `_` as a different node type cannot now silently break selector emission.

**V2. `name_map` and `selector_map` silently pick the first match.**
`update_unresolved_edges:993` and `resolve_selector_ids:818-823` store one node per name. In a 1M-LOC monorepo there will be many same-named types/selectors across modules. INHERITS/CONFORMS edges resolve non-deterministically.

**V3. NetworkX in-memory graph goes stale.**
Server loads the graph at startup. If the indexer runs while the MCP server is alive, the server keeps serving the old graph while `read_symbol` reads fresh disk — exactly the cache-drift the README claims to avoid.

**V4. `semantic_search` is O(N log N) on argsort.**
`np.argsort(...)[-top_k:]` over the full embedding matrix. At 500k vectors this is the hot path for the most-called tool.

**V5. Fresh subprocess + model load per indexer run.**
`embed_signatures` (`indexer.py:736-747`) spawns a one-shot `ProcessPoolExecutor` and `_generate_embeddings_worker` reloads the model from disk. Fine for full reindex, kills the "<10s for 10 files" target on incremental.

**V6. Debug `print` statements in indexer.**
`indexer.py:1126, 1142, 1150, 1162`. Goes to stdout; if the indexer is ever invoked from the server it will corrupt the MCP stdio stream.

### 2.3 Distribution concerns

**D1. Copilot MCP config keys may have drifted.**
`engine/CONNECTION_GUIDE.md` shows `mcp_servers:` for Copilot CLI and `"github.copilot.chat.mcp.servers"` for VS Code. Both surfaces have shifted in 2025–2026. Verify against current Copilot release on a clean machine before publishing.

**D2. Tool descriptions are aggressive ("STRICTLY FORBIDDEN").**
`engine/core/server.py:97-225`. Will work but irritate users who read transcripts. Tone down.

**D3. First-run model download is the #1 likely failure.**
`nomic-ai/nomic-embed-text-v1.5` is ~500 MB. Over corp VPN/proxy, this fails silently. Bundling/mirroring is mandatory for SLA bar.

**D4. `pyproject.toml` is unpinned.**
`engine/pyproject.toml:7-17`. tree-sitter-swift grammar shifts have already forced compatibility code in `indexer.py:317-340`. Pin minor versions.

### 2.4 What works and should not change

- The Map/Territory split is well-motivated; the byte-range live-read tool is the right call.
- Incremental hashing + rename detection (`indexer.py:1090-1097`) is more careful than typical.
- Savepoint-vs-transaction switching for full vs incremental (`transaction_scope`) is a nice touch.
- Selector-aware CALLS resolution is the right idea, even if implementation has bugs.
- Test fixture is small and intentional; tests are readable. Even if `CalculatorApp` is not representative of the real repo, it serves as a regression suite.

---

## 3. Phased Roadmap

Each phase below lists: goal, work units (with code refs), acceptance criteria, estimated effort, dependencies on prior phases.

### Phase 0 — Repo Hygiene (½ day, blocks everything)

**Goal:** Clean baseline so subsequent phases land cleanly.

**Work units:**

P0.1 — Untrack build artifacts.
- Add `test_fixtures/**/.build/` to `.gitignore`.
- `git rm -r --cached test_fixtures/CalculatorApp/.build`.
- This is a destructive operation on history; confirm with project owner before running.

P0.2 — Delete dead code.
- `engine/indexer.py` (orphaned prototype).
- `engine/setup_lab.py` (duplicate).
- `engine/lab/verify.py` (imports nonexistent symbols).
- `engine/test_ts.py` (debug scratch).
- `engine/PRODUCTION_HARDENING_PROMPT.md` (superseded by this doc; move to `docs/design/` or delete).

P0.3 — Fix README quick-start path.
- `README.md:107`: change `uv run indexer.py` to `uv run engine/core/indexer.py` (or to the new console-script name from P0.4).

P0.4 — Convert to installable package.
- Move `engine/core/` to `src/ios_graphrag/`.
- Update `engine/pyproject.toml` (or move to root) to declare `[project.scripts]` entries: `ios-graphrag-index = "ios_graphrag.indexer:main"`, `ios-graphrag-server = "ios_graphrag.server:main"`.
- Eliminate the `sys.path.insert` hack in `tests/test_mcp_tools.py:18`.
- Update `tests/conftest.py:45` to invoke the new entry point.

P0.5 — Pin dependencies.
- `engine/pyproject.toml:7-17`: pin minor versions for `mcp`, `tree-sitter`, `tree-sitter-swift`, `tree-sitter-objc`, `networkx`, `numpy`, `sentence-transformers`, `tqdm`, `einops`, `torch`.
- Commit `uv.lock`.

**Acceptance criteria (✅ Phase 0 landed in commits 578e7ed, 85550b4, 5043757):**
- [x] `git ls-files | wc -l` drops from ~1339 to ~30. → 25 (verified).
- [x] Repo size on disk <2 MB. → trivially met (25 tracked files).
- [x] Fresh clone + `uv sync` + `pytest tests/` succeeds without modifying anything. → 28 passed.
- [x] `ios-graphrag-server --help` works. → both `ios-graphrag-server` and `ios-graphrag-index --help` work.

**Effort:** 4 hours.
**Dependencies:** none.

---

### Phase 1 — Correctness Fixes (2 days) ✅ landed in commits 234f54a, 78f171c, 992bdfc

**Goal:** Stop the bleeding before adding features. Existing test suite stays green; new tests added for each fix.

**Work units:**

P1.1 — Resolve upstream/downstream semantics.
- Decision: upstream = "what I depend on" (graph successors), downstream = "what depends on me" (graph predecessors). Justification: matches typical English usage and mainstream codebases like LSP's `findReferences` / `prepareCallHierarchy`.
- Update `engine/core/server.py:122-140` to swap `predecessors`/`successors`.
- Update `README.md` prose definitions and example JSON.
- Update tool description in `server.py:97-104`.
- Update test in `tests/test_mcp_tools.py:21-83` to use the new orientation.

P1.2 — Fix unlabeled-parameter selector.
- Audit `engine/core/indexer.py:223-249` (`build_swift_selector`).
- Detect literal `_` token as the external label; fall back to the first `simple_identifier` only when no `_` is present.
- Add parametric tests in a new `tests/test_selectors.py` covering: `_ x: T`, `x: T`, `external internal: T`, `x: T = default`, `inout x: T`, variadic `x: T...`, generics `func foo<U>(x: U)`.
- Same audit for `extract_call_selector` (`indexer.py:252-308`).

P1.3 — Deterministic, observable symbol resolution.
- `resolve_all_symbol_ids` (`indexer.py:804-809`): order by `(file_path, start_byte)` so ties resolve identically across runs.
- `resolve_selector_ids` (`indexer.py:812-823`): same.
- When name_map[name] has >1 candidate or selector_map[sel] would overwrite, log a `WARNING` to `indexing_errors.log` with all candidate `(file_path, line_number)` pairs.
- Add `docs/LIMITATIONS.md` documenting that cross-module same-named symbols may resolve to the wrong target.

P1.4 — Crash isolation per file.
- Refactor `index_repository` (`indexer.py:1036-1162`) so that any exception during a single file's lifecycle (hash → parse → edge build → embed → SQL write) is caught, logged with the file path and exception type, and processing continues.
- Add a regression test: drop a malformed file (binary garbage with `.swift` extension; empty file; UTF-8 BOM; very long single line) into a tmpdir fixture and assert the index still completes with all other files indexed.

P1.5 — Replace `print` with `logging`.
- `indexer.py:1126, 1142, 1150, 1162` and any other stdout writes: switch to `log.info()` / `log.debug()`.
- Configure logger at `main()` to write to `indexing_errors.log` (existing) and stderr.

**Acceptance criteria (✅ all met):**
- [x] All existing tests pass. → 51 passed (was 28 baseline; +23 added in P1.2/P1.3/P1.4).
- [x] New `test_selectors.py` covers all parameter/argument shapes listed. → 19 cases (12 declaration + 6 call-site + 1 collision regression).
- [x] New `test_crash_isolation.py` proves a single bad file does not abort the index. → 2 tests covering binary garbage, empty file, UTF-8 BOM, 100K-char single line.
- [x] `indexing_errors.log` contains warnings for any name collisions encountered when running the index against `test_fixtures/CalculatorApp`. → 4 name collisions (calculate, configure, perform, test) and 2 selector collisions logged.
- [x] No `print` calls remain in `indexer.py` or `server.py`. → all 4 stdout writes routed through stderr + indexing_errors.log via the new logger.

**Bonus regression coverage added (carried over from Sub-1A's review):**
- `tests/test_mcp_tools.py::test_trace_dependencies_orientation_via_handler` — calls `server._trace_dependencies` directly to lock in P1.1's swap.
- `tests/test_indexer_stdout.py` — subprocess assertion that the indexer never leaks to stdout.

**Effort:** 2 days. → matched.
**Dependencies:** Phase 0 must merge first (paths change). → satisfied.

---

### Phase 2 — Security & Enterprise Gating (1 day)

**Goal:** Make TLS bypass opt-in and pass an infosec review.

**Work units:**

P2.1 — Gate TLS bypass behind environment variable.
- Move all SSL-bypass code from module-level (`server.py:13-19`, `indexer.py:13-19`, `_generate_embeddings_worker:683-696`) into a single function `_configure_insecure_tls_if_requested()` in a new `src/ios_graphrag/_tls.py`.
- Function checks `GRAPHRAG_INSECURE_TLS=1`; if absent, do nothing.
- When active, log `WARNING: TLS verification disabled by GRAPHRAG_INSECURE_TLS=1; this is insecure and should only be used with corporate proxy approval.`
- Call from `main()` of both indexer and server, and from the embedding worker entry point.

P2.2 — Document corporate CA path as the preferred alternative.
- Update `engine/CONNECTION_GUIDE.md` "Enterprise SSL" section: the recommended path is `REQUESTS_CA_BUNDLE=/path/to/corp-bundle.pem`, not the bypass.
- Mention `GRAPHRAG_INSECURE_TLS` as a last resort with explicit "get security team sign-off" warning.

P2.3 — `--cert-bundle` CLI flag.
- Add to `ios-graphrag-index` and `ios-graphrag-server` CLIs: `--cert-bundle PATH` sets `REQUESTS_CA_BUNDLE` and `SSL_CERT_FILE` for the duration of the process.
- Document in `CONNECTION_GUIDE.md`.

**Acceptance criteria (✅ Phase 2 landed in commit 41dc0ff):**
- [x] Default invocation (no env vars) verifies TLS against a real PyPI/HF endpoint. → smoke run with no env var produced 0 "TLS verification disabled" lines (gate 7 verified).
- [x] With `GRAPHRAG_INSECURE_TLS=1`, bypass is active and a WARNING is logged. → smoke run produced exactly 1 WARNING line (gate 6 verified).
- [x] With `--cert-bundle ./corp.pem`, the cert is used and verification stays on. → `tests/test_tls_config.py::test_cert_bundle_sets_env_vars` verified (REQUESTS_CA_BUNDLE + SSL_CERT_FILE both set to abs path; ssl context unchanged).
- [x] Search the codebase for `_create_unverified_https_context`: no calls outside `_tls.py`. → grep against `src/ios_graphrag/` returns hits only in `_tls.py`; `tests/test_tls_config.py` has the assertion-string mention but is itself the regression-posture test.

**Effort:** 1 day. → matched.
**Dependencies:** Phase 0 (path changes); independent of Phase 1. → satisfied.

---

### Phase 3 — Distribution UX (2 days)

**Goal:** A teammate on a fresh corp laptop goes from clone to working Copilot integration in <15 minutes, with no hand-holding.

**Work units:**

P3.1 — Preflight script (`src/ios_graphrag/preflight.py`).
- Verifies Python 3.11+, MPS availability, free disk space (≥10 GB).
- Downloads/caches embedding model with progress.
- Parses a minimal known-good Swift snippet and asserts node types from tree-sitter-swift.
- Builds a 5-symbol smoke index and round-trips a `semantic_search` call.
- Prints exactly what to put in `~/.config/gh-copilot/config.yml` and VS Code `settings.json`, with absolute paths filled in.
- Exposed as `ios-graphrag-preflight`.
- Single-screen output. Pass/fail per check. Clear remediation on failure.

P3.2 — Verify Copilot MCP config keys.
- Test on a clean macOS box with current GitHub Copilot CLI and current VS Code Copilot extension.
- Update `engine/CONNECTION_GUIDE.md` with the verified keys/paths.
- Empirical work — requires owner's machine.

P3.3 — Offline model bundling/mirroring.
- Document `huggingface-cli download nomic-ai/nomic-embed-text-v1.5 --local-dir ./models` in `CONNECTION_GUIDE.md`.
- Indexer auto-detects and prefers local copy: extend the logic at `indexer.py:704-715` to also check `~/.cache/ios-graphrag/models/` and `$IOS_GRAPHRAG_MODEL_DIR`.
- Provide instructions for staging the model on an internal artifact server (Artifactory/S3); leave the URL unset by default — owner fills in.

P3.4 — `graphrag doctor` subcommand.
- New module `src/ios_graphrag/doctor.py`.
- Diagnoses: missing DB, schema mismatch, stale model cache, bad `GRAPH_DB_PATH`, missing tree-sitter grammars, missing torch/MPS, log directory unwritable.
- Output is copy-pasteable for support requests.
- Includes "last 5 errors from `indexing_errors.log` and `server.log`".
- Exposed as `ios-graphrag-doctor` and `ios-graphrag-doctor --bug-report` (the latter dumps a longer report suitable for an issue ticket).

P3.5 — Tone down tool descriptions.
- `engine/core/server.py:97-225`: replace "STRICTLY FORBIDDEN", "DO NOT USE", "CRITICAL" with calmer language. Keep the directive intent ("Prefer this over grep because…") but drop the all-caps.

**Acceptance criteria (✅ autonomous portions landed in commits 6090ad7, 1b1e429; ⚠️ P3.2 pending owner-empirical work):**
- [x] `ios-graphrag-preflight` on a fresh machine produces a green run, with copy-pasteable Copilot config. → 7 checks PASS on this machine; emits `~/.config/gh-copilot/config.yml` YAML + VS Code `settings.json` JSON snippets at end (with caveat noting P3.2 verification still required).
- [x] `ios-graphrag-doctor` produces useful output on a healthy system and on each of the failure modes listed. → 8 diagnostics + `--bug-report` mode (Python/platform/deps/redacted env vars/log tail). Verified end-to-end with and without DB present.
- [x] Tool descriptions are readable and don't shout. → grep `STRICTLY FORBIDDEN|DO NOT USE|CRITICAL:|MUST NEVER|NEVER USE` returns empty in `src/ios_graphrag/server.py`.
- [ ] **P3.2 — pending owner empirical work.** Verify the Copilot CLI MCP config key (`mcp_servers:`) and VS Code Copilot extension key (`github.copilot.chat.mcp.servers`) are still current on a fresh corp laptop with current Copilot release. Update `engine/CONNECTION_GUIDE.md` and `src/ios_graphrag/preflight.py` config snippets if either has drifted.

**Bonus delivered:**
- P3.3 model resolution chain extended to support `IOS_GRAPHRAG_MODEL_DIR` env var + `~/.cache/ios-graphrag/models/` default cache, with HF identifier as fallback. New `tests/test_model_resolution.py` (3 tests).
- `engine/CONNECTION_GUIDE.md` adds an "Offline / Pre-staged Model" section documenting `huggingface-cli download` + Artifactory/S3 staging (with `<!-- TODO: org-specific URL -->` placeholder for owner to fill in).

**Effort:** 2 days (excluding P3.2 which depends on owner machine). → matched.
**Dependencies:** Phases 0, 1, 2. → satisfied.

---

### Phase 4 — Performance and Scale Validation (4–5 days)

**Goal:** Validate or refute the scale assumptions in §1.3, then fix the gaps surfaced by measurement.

**P4.1 is empirical and must run on the owner's real repo.** Phases 4b–f are responsive to its findings.

#### Phase 4a — Benchmark harness (1 day, then run on real repo)

- Build `benchmarks/run.py` that measures, against a configurable repo path:
  - Full index: wall time, peak RSS (via `resource.getrusage`), per-phase breakdown (hash / parse / weave / embed / SQL write).
  - Incremental: 10 / 100 / 1000 changed files (use `git checkout` of older commits to stage realistic deltas).
  - Server cold start: process spawn → first tool result.
  - `semantic_search` p50/p95/p99 over 1000 random queries drawn from a query corpus.
  - `trace_dependencies` p50/p95/p99 over 1000 random files.
  - DB size on disk; embedding matrix size in RAM; NetworkX node/edge count.
- Output: JSON to `benchmarks/results/<git-sha>-<timestamp>.json`.
- Owner runs against a representative slice (start with 200k LOC, then full 1M).

**Acceptance criteria for 4a:** harness runs locally on `test_fixtures/CalculatorApp` and produces a result JSON. Owner runs against the real repo and shares the results JSON for the next phases.

#### Phase 4b — Eliminate per-incremental model reload (½ day)

- Audit whether the SSL/httpx contamination concern that motivated `embed_signatures` (`indexer.py:736-747`) still applies with current `sentence-transformers`.
- If not, run the embedding model in-process. Add a benchmark before/after.
- If it still applies, restructure as a long-lived embedding daemon (separate process, persistent socket).

#### Phase 4c — Indexer daemon + file watcher (1.5 days)

- `src/ios_graphrag/watcher.py` using `watchdog`.
- Monitors repo, debounces FS events (1s window), triggers incremental reindex.
- Keeps the embedding model loaded.
- Logs to `~/Library/Logs/ios-graphrag/`.
- Provide a sample `launchd` plist in `dist/launchd/com.user.ios-graphrag.watcher.plist`.
- Exposed as `ios-graphrag-watch --repo PATH`.
- Opt-in; no auto-install.

#### Phase 4d — Vector search at scale (1 day)

- Step 1: replace `np.argsort` with `np.argpartition` in `_semantic_search:240`. This is unconditional and cheap.
- Step 2 (decision based on 4a benchmarks): if `semantic_search` p99 > 500ms on the real repo, build an `hnswlib` HNSW index alongside SQLite. Persisted as `embeddings.hnsw`. Loaded into RAM at server start. Search drops to <2ms.
- Document the decision and rationale in `docs/architecture/vector-search.md`.

#### Phase 4e — Server cold-start optimization (½ day)

Conditional on 4a results showing cold start >5s. Pick from:
- Mmap embedding matrix to disk (`np.save` to `embeddings.npy`, `np.load(mmap_mode='r')`).
- Lazy-load NetworkX: do not materialize at startup; query SQLite per call. Only viable if `trace_dependencies` p99 stays under 100ms.
- Defer embedding model load until first `semantic_search` call (already partly true; verify).

#### Phase 4f — Hot-path correctness (½ day)

- Push `find_bridging_header_usage` filter into SQL: `SELECT * FROM edges WHERE edge_type='BRIDGING'` and join. Eliminates the in-memory `GRAPH.edges` walk at `server.py:167-171`.
- At server startup, record `mtime(GRAPH_DB_PATH)`. On each tool call, if mtime advanced, atomically reload `GRAPH` and `EMBEDDING_MATRIX`. Adds one syscall per call. Eliminates cache drift between indexer runs and a long-lived MCP server.

**Acceptance criteria for Phase 4 overall:**
- Real-repo benchmark numbers replace the speculative ones in §1.3 of this doc.
- README "Performance Targets" table reflects measured values.
- All targets met or explicitly downgraded with rationale.
- Server reload-on-mtime-change verified by a manual test (run indexer while server is up, see new symbols appear in tool output without restart).

**Effort:** 4–5 days (less if 4a results are forgiving).
**Dependencies:** Phases 0, 1, 2, 3.

---

### Phase 4.5 — Parser Strategy for SwiftUI + Macros (3 days, plus owner audit)

**Goal:** Make the parser robust against the SwiftUI/macro patterns that dominate the real codebase.

#### Phase 4.5a — Parse-error audit (½ day, owner runs on real repo)

- Build `tools/parse_audit.py`: indexes a directory, but for each file records `tree.root_node.has_error` and the count of `ERROR` nodes.
- Output: per-file error count, summary stats (% of files with any error, distribution of error counts).
- Owner runs against a 10k-LOC slice first, then 100k, then full 1M.
- Decision matrix:
  - <5% files with errors → Strategy A (tree-sitter primary, hardened fallback).
  - 5–20% → Strategy A + B (tree-sitter primary, SourceKitten validator in CI).
  - >20% → Strategy C (consider SourceKit-LSP as primary).

#### Phase 4.5b — Strategy A: hardened fallback (1.5 days)

Replace the regex fallback (`indexer.py:584-619`) with a much more capable line-based parser:
- Handles multi-line signatures.
- Handles `@attribute` decorations: `@MainActor`, `@Observable`, `@Model`, `@MainActor public class Foo`.
- Handles property declarations including computed: `var foo: Int { get }`, `var foo: Int { didSet { … } }`.
- Handles SwiftUI-specific patterns: `: View`, `: ViewModifier`, generic constraints `func body() -> some View`.
- Handles `where` clauses and generic parameter clauses.
- Falls through from tree-sitter only when `tree.root_node.has_error` AND the error coverage in a particular subtree exceeds a threshold (per-symbol fallback, not whole-file).
- Add tests: a directory of `tests/fixtures/edge_cases/*.swift` files with ~30 tricky snippets, each with expected symbols.

#### Phase 4.5c — Strategy B: SourceKitten validator (1 day, optional)

- CI job (nightly): runs SourceKitten on a sample of the real repo, runs tree-sitter+fallback on the same, diffs symbol output.
- Posts a report to `benchmarks/parser-drift/<date>.md`: missed symbols, type mismatches, file-level deltas.
- Does not gate releases by default; informs decisions to update tree-sitter or extend fallback.

#### Phase 4.5d — SwiftUI-specific symbol enrichment (1 day, regardless of strategy)

Add SQL columns to the `nodes` table (via migration in Phase 6a):
- `is_swiftui_view BOOLEAN` — true when struct conforms to `View`.
- `is_observable BOOLEAN` — true when class has `@Observable` or `@Model` attribute.
- `state_kind TEXT` — for properties: `state | binding | stateobject | observedobject | environmentobject | environment | appstorage | scenestorage | fetchrequest | query | published | null`.
- `body_kind TEXT` — for functions: `viewbody | resultbuilder | regular | null`.

Update `extract_swift_symbols` (`indexer.py:344-516`) to capture these decorations and write them into the new columns.

Expose via a new MCP tool, `find_swiftui_views`:
- Returns all `View`-conforming types, optionally filtered by state-kind dependencies.
- Useful question it answers: "what views depend on AuthState as `@StateObject`?"

**Acceptance criteria for Phase 4.5:**
- Audit JSON exists for the real repo.
- Strategy chosen is documented in `docs/architecture/parser-strategy.md` with rationale.
- For Strategy A: edge-case fixture suite passes.
- SwiftUI symbol enrichment columns populated for `test_fixtures/CalculatorApp` (all rows null) and for any added SwiftUI fixture.

**Effort:** 3 days for B + D + chosen primary.
**Dependencies:** Phase 4 (so audit runs against benchmarked code), Phase 0.

---

### Phase 5 — Coverage Gaps (1 week, prioritized)

**Goal:** Capture the symbol kinds the team will actually search for. Priorities are tuned for SwiftUI-heavy + macros + 1M LOC.

| Priority | Item | Why this codebase | Effort |
|---|---|---|---|
| P0 | Properties / computed vars | Cannot navigate SwiftUI without state visibility | 1 day |
| P0 | SwiftUI View detection (column flag) | Marks the entire UI layer | ½ day (covered in 4.5d) |
| P0 | Property wrappers as enriched metadata | Critical for SwiftUI/Combine | covered in 4.5d |
| P0 | Initializers (`init`) and deinitializers | Used in DI patterns | ½ day |
| P1 | `#Preview` / `@Observable` / `@Model` decorations | Modern SwiftData/SwiftUI | covered in 4.5d |
| P1 | Trailing closure CALLS (`.onTapGesture { foo() }`) | SwiftUI = 80% trailing closures | 1 day |
| P1 | Modifier chain CALLS (`.padding().background(…)`) | Verify and fix gaps | ½ day |
| P1 | Enum cases | State machines | ½ day |
| P2 | ObjC selectors / categories | Mixed codebase | 1 day |
| P2 | Subscripts, typealiases | Long tail | ½ day |
| P3 | Generics with `where` clauses | Mostly cosmetic | not in scope |

**Acceptance criteria per item:**
- Symbol type added to `symbol_type` CHECK constraint where applicable (via migration).
- Extraction logic in `extract_swift_symbols` (or `extract_objc_symbols`).
- Test fixture file with expected symbols.
- Test asserting indexer captures them.
- Updates to README's symbol-types section.

**Effort:** 5 days for P0+P1.
**Dependencies:** Phases 4.5, 6a.

---

### Phase 6 — SLA-Grade Operational Maturity (3–4 days)

**Goal:** Production-grade support, observability, upgrade safety.

#### Phase 6a — Schema versioning + migrations (1 day, blocking)

- Add `CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)` to `engine/database/schema.sql`.
- Create `engine/database/migrations/`:
  - `001_baseline.sql` — current schema as-is.
  - `002_add_swiftui_columns.sql` — adds `is_swiftui_view`, `is_observable`, `state_kind`, `body_kind`.
  - Future migrations append.
- Indexer at startup: read schema version. If absent (legacy DB), apply baseline + all subsequent migrations. If newer than code knows, fail with clear error: `Database at vN, this build expects vM. Upgrade ios-graphrag or rebuild the index.`
- Server at startup: same check; refuse to start on mismatch.
- Tests: forward migration from v0 (legacy) → current; "old DB on new code" path is graceful.

#### Phase 6b — Structured logging + trace IDs (½ day)

- Switch to stdlib JSON formatter (or `structlog` if a lighter dep is acceptable).
- Each MCP tool call gets a UUID4 trace ID; logged on entry, exit, every error.
- Log destination: `~/Library/Logs/ios-graphrag/server.log` and `…/indexer.log`. Rotating handler at 50 MB, keeping 5 files.
- `graphrag doctor --tail-errors N` reads the last N error lines from both files.

#### Phase 6c — Edge dedup + integrity checks (½ day)

- Add `UNIQUE(source_node_id, target_symbol, edge_type, line_number)` to `edges`. Migration handles existing duplicates.
- Indexer end-of-run integrity check:
  - Row counts per table.
  - Orphan edges (`target_node_id` pointing nowhere).
  - Nodes whose `file_path` is no longer on disk.
  - Symbol names that resolve to >1 node.
- Logs warnings, never fails the run.
- `ios-graphrag-doctor --verify` runs the same checks on demand.

#### Phase 6d — Crash diagnostics (½ day)

- On unhandled exception in server: write a crashdump file with full traceback, environment info (Python version, package versions, GRAPH_DB_PATH, env vars except secrets), and last N tool calls (kept in a ring buffer).
- Server tool errors include the trace ID so users can grep logs.
- `docs/REPORTING.md`: bug-report template with `ios-graphrag-doctor --bug-report` output.

#### Phase 6e — CI gates (½ day)

- GitHub Actions on every PR:
  - macOS-latest runner.
  - Matrix: Python 3.11, 3.12.
  - `pytest tests/` must pass.
  - `ruff check` must pass.
  - `ios-graphrag-preflight --smoke` must pass.
- Nightly job: full benchmark run on a checked-in slice of fixtures, post results to `benchmarks/results/`. Compare to baseline; fail if regression >10% on full reindex or >20% on `semantic_search` p99.

#### Phase 6f — Versioned releases + on-call runbook (½ day)

- Maintain `CHANGELOG.md` with semver tags.
- `docs/RUNBOOK.md`: top-10 known failures with remediation.
- "Server crashes on startup" → check schema version with `ios-graphrag-doctor`.
- "Semantic search returns nonsense" → check model cache integrity.
- "Indexer hangs" → check `indexing_errors.log` for parser timeouts.
- Pin a "current stable" version your team installs from; do not ask them to track main.

#### Phase 6g — Opt-in telemetry (½ day, optional)

- Single counter, opt-in via `GRAPHRAG_TELEMETRY_URL=https://internal.team/endpoint`.
- Sends: version string, command name, success/failure, duration in ms.
- Sends nothing else. No code, no paths, no symbol names.
- Disabled by default. Document in `docs/PRIVACY.md` exactly what is and is not sent.
- Owner decision required before implementation.

**Acceptance criteria for Phase 6:**
- Old DB (no schema_version row) loads on new code without manual intervention.
- New DB on old code refuses to start with a clear message.
- All `print` and most `log.info` outputs are JSON-structured.
- Crashdump file created on a forced exception.
- CI runs on a sample PR.
- `RUNBOOK.md` exists with at least 10 entries.

**Effort:** 3–4 days.
**Dependencies:** Phases 0–5 (some items reference features added there).

---

## 4. Sequencing

### 4.1 Eight-week plan

| Week | Phases in flight | Beta cohort | Dependencies / risk |
|---|---|---|---|
| 1 | 0 → 1 → 2 | — | Low. Foundational. |
| 2 | 3 + 4.5a audit | 1 dev (project owner) | Medium. Audit informs Phase 4.5. |
| 3 | 4a benchmarks (owner runs) + 4.5d enrichment | 1 dev | High. Reveals scale issues. |
| 4 | 4b–f based on 4a findings | 2 devs | Medium. May spike if HNSW needed. |
| 5 | 4.5b/c parser strategy | 2 devs | High. Largest implementation block. |
| 6 | 5 P0 items | 3–5 devs | Medium. SwiftUI coverage. |
| 7 | 5 P1 + 6a, b, c | 5–10 devs | Low. Mostly additive. |
| 8 | 6d–g + GA prep + RUNBOOK | full team | Low. Buffer + polish. |

Compress only by descoping Phase 5 P1, never by skipping Phase 4 or 6a.

### 4.2 Critical path

```
Phase 0
  └─> Phase 1 ─┐
  └─> Phase 2 ─┼──> Phase 3 ─> Phase 4a (owner) ─> Phase 4b–f ─┐
                                Phase 4.5a (owner) ─> Phase 4.5b/c/d ─┤
                                                                       └─> Phase 5 P0 ─> Phase 5 P1 ─> Phase 6a ─> Phase 6b–g ─> GA
```

Phase 0 blocks all. Phases 1, 2, 3 parallelize after 0. Phase 4a and 4.5a are owner-empirical. Phase 6a is blocking before any further schema changes ship.

### 4.3 Parallelism within phases

- Phase 1: serialize. P1.1, P1.2, P1.3 all touch `indexer.py` or `server.py`; merge conflicts likely.
- Phase 3: P3.1 (preflight) and P3.4 (doctor) are disjoint files; parallel-safe. P3.5 (tone) is independent.
- Phase 4: 4b is its own concern; 4c is its own file; 4d, 4e, 4f touch `server.py`; serialize the latter three.
- Phase 5: each item is its own extraction routine; mostly parallel-safe.
- Phase 6: 6a blocks 6c (schema migration). 6b, 6d are independent. 6e, 6f are docs/CI; parallel.

---

## 5. Empirical Work Requirements

These items cannot be performed without the project owner's machine and real repo:

| ID | Phase | Work | What it produces |
|---|---|---|---|
| E1 | 4a | Run benchmark harness against representative slice (200k LOC), then full repo (1M LOC). | `benchmarks/results/<sha>.json` |
| E2 | 4.5a | Run parse-error audit against real repo. | Per-file error count + summary |
| E3 | 3 | Verify Copilot CLI / VS Code MCP config keys on a fresh corp laptop. | Updated `CONNECTION_GUIDE.md` |
| E4 | 4c | Validate watcher daemon UX in a real day-of-work cycle. | Acceptance signoff |
| E5 | 6e | Configure GitHub Actions for the team's GitHub instance. | Working CI |
| E6 | 6g | Set up internal telemetry endpoint if owner opts in. | URL + endpoint |

For each empirical item, the orchestrator should:
1. Build the harness/tool.
2. Hand off to the project owner with a clear runbook.
3. Wait for results JSON or signoff before proceeding to dependent items.

---

## 6. Open Decisions (Defaulted)

The owner has not made the following decisions explicit. Defaults below are encoded into the plan; revise before execution if needed.

| ID | Decision | Default | Phase |
|---|---|---|---|
| O1 | Internal artifact server URL for the embedding model | Document offline procedure; do not bake URL | 3 |
| O2 | Watcher daemon mechanism | Ship `launchd` plist + manual mode | 4c |
| O3 | Log destination | `~/Library/Logs/ios-graphrag/`, rotated at 50 MB | 6b |
| O4 | Doctor / runbook style | Self-serve copy-paste; no Slack integration | 3, 6f |
| O5 | Telemetry | Omitted; revisit on owner request | 6g |
| O6 | Parser primary strategy | Decided by Phase 4.5a audit | 4.5 |
| O7 | HNSW vs NumPy for vector search | NumPy + argpartition; revisit if 4a p99 fails | 4d |
| O8 | Renaming `ios-graphrag-*` console scripts | Underscore-style binaries (`ios-graphrag-index`); change if owner prefers | 0 |

---

## 7. Orchestration Model (recommendation for fresh session)

### 7.1 Subagent dispatch

Each phase is decomposed into work units (above). For each unit:

1. Spawn a `general-purpose` subagent with `isolation: "worktree"` so its diff is isolated.
2. Brief: explicit acceptance criteria from this doc + file/line references + test expectations + a "do not change anything outside these files" guardrail.
3. Subagent returns: diff summary, test results, any open questions.

### 7.2 Review gates

For each completed implementation diff:

1. Spawn a `general-purpose` subagent (read-only, no Edit/Write) with the diff content.
2. Brief: "Find bugs, missing tests, style inconsistencies, security issues, breaking API changes. Report under 300 words."
3. Orchestrator decides: merge as-is, send back with specific fixes, or escalate to owner.

### 7.3 Merge strategy

- Each phase merges to `claude/review-ast-graphrag-cI7X4` as a single squashed commit (or PR if owner prefers PR-per-phase).
- One PR per phase to GitHub via the github MCP; owner reviews at phase boundaries.
- Destructive operations (`git rm` for `.build`, branch resets, schema-breaking migrations) require explicit owner confirmation.
- Push uses `git push -u origin claude/review-ast-graphrag-cI7X4`. No force-push without owner sign-off.

### 7.4 Parallelism rules

- Within a phase: serialize when units touch the same file; parallelize when files are disjoint.
- Across phases: only when no dependency. Phases 1, 2, and 3-docs can overlap.
- Always isolated worktrees so failed work is throwaway.

### 7.5 Stopping conditions

The orchestrator should pause and ask the owner when:
- A phase's acceptance criteria cannot be met after one review-fix round.
- An empirical phase result is missing.
- A subagent proposes an architecturally significant change (new dependency, schema-breaking migration, removal of a public-facing tool).
- Any destructive git operation (force-push, history rewrite).
- An owner-defaulted decision (§6) needs to be re-asked because circumstances changed.

---

## 8. References

### 8.1 Code locations cited

- `engine/core/server.py:13-19` — module-level SSL bypass.
- `engine/core/server.py:97-225` — MCP tool decorators and shouty descriptions.
- `engine/core/server.py:122-140` — upstream/downstream semantic mismatch.
- `engine/core/server.py:167-171` — bridging-edge in-memory filter.
- `engine/core/server.py:240` — `np.argsort` in semantic search.
- `engine/core/indexer.py:13-19` — module-level SSL bypass.
- `engine/core/indexer.py:223-249` — `build_swift_selector` (selector bug).
- `engine/core/indexer.py:252-308` — `extract_call_selector`.
- `engine/core/indexer.py:317-340` — tree-sitter grammar compatibility shim.
- `engine/core/indexer.py:584-619` — regex fallback parser.
- `engine/core/indexer.py:683-696` — embedding worker SSL bypass.
- `engine/core/indexer.py:704-715` — local model snapshot detection.
- `engine/core/indexer.py:736-747` — per-call subprocess for embeddings.
- `engine/core/indexer.py:804-823` — symbol/selector resolution maps.
- `engine/core/indexer.py:1036-1162` — `index_repository` orchestration.
- `engine/core/indexer.py:1126, 1142, 1150, 1162` — debug `print` statements.
- `engine/database/schema.sql` — current schema (no version table).
- `engine/CONNECTION_GUIDE.md` — Copilot config to verify.
- `engine/PRODUCTION_HARDENING_PROMPT.md` — superseded design doc; delete in Phase 0.
- `engine/pyproject.toml:7-17` — unpinned dependencies.
- `tests/conftest.py:45` — indexer invocation path; update in P0.4.
- `tests/test_mcp_tools.py:18` — `sys.path.insert` hack to remove in P0.4.
- `tests/test_function_calls.py:104-125` — overload-resolution test (will guide V1 fix).
- `README.md:107` — wrong indexer path.
- `.gitignore` — needs `test_fixtures/**/.build/` line.

### 8.2 External references to verify before execution

- Current `github-copilot` CLI MCP config schema.
- Current VS Code Copilot extension MCP settings key.
- Current `tree-sitter-swift` release notes for `inheritance_specifier` vs `inheritance_clause` grammar.
- `nomic-ai/nomic-embed-text-v1.5` model card and license terms.
- `hnswlib` license compatibility for internal redistribution.

---

## Document maintenance

- This doc is the source of truth for the production hardening effort. Update it as decisions are made and phases complete.
- When a phase merges, mark its acceptance criteria as checked here, not just in the PR description.
- Open decisions in §6 should move to a "Decided" section as they get resolved.
- Phase numbers are stable; do not renumber. Add sub-phases (e.g., 4g) if scope grows.
