# Handoff — iOS-GraphRAG Production Hardening

**Date:** 2026-05-07
**Branch:** `claude/admiring-kirch-b49458`
**HEAD:** `8b4da97` (or `cd05a28` for the 0.2.0 stamp + add the post-stamp commits on top)
**Source-of-truth roadmap:** `docs/PRODUCTION_ROADMAP.md` (acceptance criteria checked off in-line)

This doc is everything you need to pick up the project on your personal machine. Read sections 1–3 to get oriented, then jump to section 4 for the empirical work that needs your repo.

---

## 1. State at handoff

| Metric | Value |
|---|---|
| Commits on branch | 37 |
| Tests passing | **181** (was 28 baseline) |
| Tracked files | 84 (was 1339 pre-Phase-0) |
| Schema version | 6 |
| Package version | 0.2.0 |
| Implementation/review pairs | 25 / 25, all APPROVE or APPROVE WITH NOTES |
| Console scripts | 5 (`ios-graphrag-{index, server, preflight, doctor, watch}`) |
| MCP tools | 5 (`global_codebase_search`, `swift_dependency_tracer`, `read_symbol_source`, `objc_swift_bridge_finder`, `find_swiftui_views`) |
| Symbol types | 18 |
| SwiftUI state wrappers | 17 (covers iOS 13–17 SwiftUI + Combine + SwiftData) |
| CI workflows | 2 (PR matrix + nightly with regression check) |

### Phases shipped autonomously

- **Phase 0** Repo hygiene + src-layout package
- **Phase 1** Correctness fixes (semantics, selectors, resolution determinism, crash isolation, logging)
- **Phase 2** TLS bypass gating (opt-in env var + `--cert-bundle` flag)
- **Phase 3** Distribution UX (preflight, doctor, offline model docs, tone-down) — *except P3.2*
- **Phase 4a** Benchmark harness (`benchmarks/run.py`)
- **Phase 4.5a** Parse-error audit (`tools/parse_audit.py`)
- **Phase 4b** In-process embedding model (−17% full reindex, −34% embed phase)
- **Phase 4c** Watcher daemon + launchd plist (`ios-graphrag-watch`)
- **Phase 4d.1** `np.argpartition` for top-k semantic search
- **Phase 4f** Bridging-edge SQL push + mtime auto-reload
- **Phase 4.5d** SwiftUI symbol enrichment + `find_swiftui_views` tool + 17 state-kind wrappers
- **Phase 5 P0** Properties + computed_property + initializer + deinitializer
- **Phase 5 P1** Enum cases + trailing closures + modifier chains
- **Phase 5 P2** Subscripts + typealiases + rebuilt ObjC extractor
- **Phase 6a** Schema versioning + migrations runner
- **Phase 6b** Structured JSON logging + UUID4 trace IDs + log rotation
- **Phase 6c** Edge dedup + integrity checks + `doctor --verify`
- **Phase 6d** Crashdump on excepthook + recent-calls ring buffer + `docs/REPORTING.md`
- **Phase 6e** GitHub Actions CI + ruff + preflight `--smoke`
- **Phase 6f** `CHANGELOG.md` + `docs/RUNBOOK.md` + 0.2.0 version bump
- **Bonus** End-to-end MCP integration tests over real stdio JSON-RPC
- **Bonus** Fixture baseline + CI regression check

### Five hidden bugs caught and fixed

1. **V1 selector bug never existed** — the locked tree-sitter-swift grammar emits `_` as `simple_identifier` text, so the original code accidentally produced correct selectors. Phase 1 P1.2 became a defensive refactor with parametric tests.
2. **`load_graph` was non-idempotent** — discovered while wiring `_reload_if_stale` for Phase 4f's mtime auto-reload. Fixed.
3. **`enum_class_body` was missing from the recursion set** — Phase 5 P1 uncovered 4 silently-missed enum cases in `StateMachine.State`.
4. **`extract_objc_symbols` was completely broken** against tree-sitter-objc 3.0.2 — used legacy node-type names that the grammar doesn't emit, so 0 ObjC rows were produced for any `.h`/`.m` file. Phase 5 P2 rewrote it.
5. **MCP transport spec was wrong** — Sub-INT verified the `mcp` package on this branch uses newline-delimited JSON, not Content-Length framing.

---

## 2. Get oriented on your personal machine

```bash
# Clone (if you haven't already)
git clone <your-fork-or-origin>
cd iOS-GraphRAG

# Switch to this branch
git fetch origin
git checkout claude/admiring-kirch-b49458    # or pull from the worktree

# If you haven't pushed the branch yet, see section 6.

# Install
uv sync --all-extras --dev

# Verify
uv run pytest tests/ -q                       # expect 181 passed
uv run ruff check .                           # expect "All checks passed!"
uv run ios-graphrag-preflight                 # expect 7 PASS rows
uv run ios-graphrag-doctor                    # diagnose env
```

Initial preflight pulls the embedding model (~500 MB from HuggingFace). If your corp network blocks HF, see `engine/CONNECTION_GUIDE.md` → "Offline / Pre-staged Model".

---

## 3. Architecture quick reference

```
src/ios_graphrag/
  indexer.py          # extracts symbols + edges; runs migrations; populates DB
  server.py           # MCP server with 5 tools + traced_handler decorator
  preflight.py        # 7-check installer sanity (or 4-check --smoke)
  doctor.py           # 8 diagnostics + --bug-report / --verify / --tail-errors
  watcher.py          # ios-graphrag-watch daemon (debounced incremental reindex)
  _logging.py         # JSON file + text stderr + rotation
  _migrations.py      # schema version runner; 4 migrations live in engine/database/migrations/
  _diagnostics.py     # crashdump excepthook + recent-calls ring buffer
  _integrity.py       # orphan-edge / missing-file / collision detection
  _tls.py             # opt-in TLS bypass + corp CA bundle wiring

engine/database/
  schema.sql          # baseline (preserved for reference; runtime uses migrations)
  migrations/         # 001..006 + README
    001_baseline.sql
    002_add_swiftui_columns.sql
    003_unique_edges.sql
    004_add_property_init_symbol_types.sql
    005_add_enum_case_symbol_type.sql
    006_add_subscript_typealias_symbol_types.sql

benchmarks/
  run.py              # Phase 4a harness
  baseline/           # fixture baseline JSONs + README
  check_regression.py # nightly CI regression gate
  README.md

tools/
  parse_audit.py      # Phase 4.5a parse-error audit
  README.md

dist/launchd/
  com.user.ios-graphrag.watcher.plist  # template (placeholder paths)
  README.md           # install instructions

docs/
  PRODUCTION_ROADMAP.md   # source of truth; acceptance criteria checked
  CHANGELOG.md            # at repo root
  RUNBOOK.md              # top-10 on-call failure modes
  REPORTING.md            # bug-report flow
  LIMITATIONS.md          # known limitations honestly documented
  HANDOFF.md              # this file

.github/workflows/
  ci.yml                  # PR: ruff + pytest + preflight --smoke
  nightly-benchmark.yml   # daily: benchmark + parse audit + regression check
```

---

## 4. Three empirical tasks — your machine only

### 4a. P3.2 — Verify Copilot CLI / VS Code MCP config keys

**Effort:** ~30 min on a fresh corp laptop with Copilot installed.

You already noted the Copilot CLI config is now JSON with the `mcpServers` key (Claude-Desktop-style), not the YAML `mcp_servers:` we currently document. The handoff includes both keys we currently document; both need verification.

**Step 1.** Locate the actual file path on a current Copilot CLI install:

```bash
# Best candidates (try in order):
ls -la ~/.config/gh-copilot/config.json     # JSON drop-in for the old YAML
ls -la ~/Library/Application\ Support/GitHub\ Copilot\ CLI/config.json
gh copilot config --help                    # may print the path
```

**Step 2.** Drop our snippet into the config and try:

```json
{
  "mcpServers": {
    "iOS-GraphRAG": {
      "command": "/path/to/.venv/bin/ios-graphrag-server",
      "env": {
        "GRAPH_DB_PATH": "/path/to/your/knowledge-graph.sqlite"
      }
    }
  }
}
```

Then in Copilot CLI, ask: *"List the MCP tools you have access to"* — if our 5 tools (`global_codebase_search`, `swift_dependency_tracer`, `read_symbol_source`, `objc_swift_bridge_finder`, `find_swiftui_views`) appear, the key works.

**Step 3.** Same drill for VS Code Copilot extension. Current docs use `"github.copilot.chat.mcp.servers"` — verify this is still correct via VS Code Settings UI search for "mcp" under the Copilot extension.

**Step 4.** Update these locations with the verified path/key:
- `engine/CONNECTION_GUIDE.md` lines ~60–85 (Copilot CLI section).
- `engine/CONNECTION_GUIDE.md` lines ~72–85 (VS Code section).
- `src/ios_graphrag/preflight.py` lines ~454–472 (the footer template that prints the snippet).

**Step 5.** Update `tests/test_console_scripts.py` if any test snippet now has a stale shape.

### 4b. Phase 4a — Real-repo benchmark

**Effort:** ~30–60 min on the real iOS repo.

```bash
# Run against a 200k-LOC slice first if available, then full repo:
uv run python benchmarks/run.py --repo /path/to/your/iOS/repo --output benchmarks/results/

# The harness:
# - Stashes any uncommitted changes defensively (popped at the end).
# - Tries incremental deltas at 10/100/1000 changed files (uses `git checkout HEAD~k`).
# - Times semantic_search + trace_dependencies p50/p95/p99 over 1000 random queries each.
# - Reports memory snapshots.
# - Output: benchmarks/results/<git-sha>-<timestamp>.json
```

Then **share the JSON** so I can use the real numbers to drive Phase 4b–f decisions:

```bash
# Stay tight on what's interesting:
cat benchmarks/results/*.json | jq '{
  full_index: .full_index,
  semantic_search: .semantic_search_latency_ms,
  trace_dependencies: .trace_dependencies_latency_ms,
  memory: .memory
}'
```

**What the numbers tell us:**
- `full_index.wall_seconds` > 40 min → Phase 4c daemon helps a lot.
- `semantic_search_latency_ms.p99` > 500 ms → Phase 4d.2 (HNSW index) is needed.
- `server_cold_start.spawn_to_first_result_seconds` > 5 s → Phase 4e (cold-start opt) is needed.
- `memory.embedding_matrix_bytes` > 2 GB → Phase 4e mmap is needed.

### 4c. Phase 4.5a — Real-repo parse audit

**Effort:** ~30 min, scaling-test approach.

```bash
# Recommended workflow per the roadmap:
uv run python tools/parse_audit.py --repo /path/to/repo/Modules/SmallSlice --output benchmarks/results/   # ~10k LOC
uv run python tools/parse_audit.py --repo /path/to/repo/Modules/MediumSlice --output benchmarks/results/  # ~100k LOC
uv run python tools/parse_audit.py --repo /path/to/your/iOS/repo --output benchmarks/results/             # full ~1M LOC
```

The console output ends with a recommendation: **Strategy A**, **Strategy A+B**, or **Strategy C**. That recommendation drives which Phase 4.5b/c/d tasks I run next:

| Decision | What it unlocks |
|---|---|
| **A** (`<5%` parse errors) | Phase 4.5b — hardened regex fallback. ~1.5 days. |
| **A+B** (`5–20%`) | Phase 4.5b + 4.5c — fallback + nightly SourceKitten validator. ~2.5 days. |
| **C** (`>20%`) | Phase 4.5c+ — consider SourceKit-LSP as primary parser. Architecture review needed. |

Share the JSON output, especially the `top_files` section showing which files have the highest error counts — those are the parser's known weak points.

---

## 5. What unlocks after the empirical results

| Phase | Conditional on | Effort if triggered |
|---|---|---|
| **4d.2** HNSW vector search | semantic_search p99 > 500 ms | ½ day |
| **4e** Server cold-start optimization | cold start > 5 s | ½ day |
| **4.5b** Hardened parse fallback | parse audit %, regardless of strategy | 1.5 days |
| **4.5c** SourceKitten validator (CI) | parse audit shows ≥5% errors | 1 day |
| **6g** Opt-in telemetry endpoint | your decision: yes/no/skip | ½ day |

I can execute any of these autonomously once you share the relevant input (the benchmark JSON for 4d.2/4e; the parse-audit JSON for 4.5b/c; a yes/no for 6g).

---

## 6. Push & PR

The branch hasn't been pushed yet. When ready:

```bash
# From inside the worktree (or after checking out the branch on your machine):
git push -u origin claude/admiring-kirch-b49458
```

If you want to **squash the 37 commits** before pushing (keeps the PR linear), I can do that — just say the word. Otherwise the per-phase commit history is intact and reviewers can step through.

Once pushed, open the PR with this body skeleton:

```markdown
# Production Hardening — Phases 0-6 (mostly autonomous)

Implements `docs/PRODUCTION_ROADMAP.md` Phases 0-6 (autonomous portions),
plus end-to-end MCP integration tests and an extended SwiftUI wrapper
coverage pack.

## Stats

- 28 → 181 tests
- 1339 → 84 tracked files (Phase 0 untracked 1311 build artifacts)
- 5 console scripts (`ios-graphrag-{index, server, preflight, doctor, watch}`)
- 5 MCP tools (added `find_swiftui_views`)
- 6 schema migrations applied via runner
- 17 SwiftUI state-kind wrappers detected

## Hidden bugs caught

1. `extract_objc_symbols` was completely broken against tree-sitter-objc 3.0.2 — 0 ObjC rows for any `.h`/`.m`. Rewrote.
2. `enum_class_body` missing from indexer's recursion set — 4 missed enum cases in StateMachine.State.
3. `load_graph` was non-idempotent.
4. V1 selector bug per roadmap doesn't exist in locked grammar — fix is now a defensive refactor.
5. MCP transport is newline-delimited (not Content-Length framed) on the locked `mcp` version.

## Empirical work pending (separate PRs)

- P3.2: verify Copilot CLI + VS Code MCP config keys
- Phase 4a: real-repo benchmark
- Phase 4.5a: real-repo parse audit
- Conditional Phase 4d.2 / 4e / 4.5b/c depending on the above

## Test plan

- [ ] `uv sync --all-extras --dev` clean
- [ ] `uv run pytest tests/ -q` reports 181 passed
- [ ] `uv run ruff check .` clean
- [ ] `uv run ios-graphrag-preflight` reports 7 PASS
- [ ] CI workflow passes on the matrix (Python 3.11 + 3.12 on macos-latest)
```

---

## 7. Known minor follow-ups (non-blockers)

- README.md CI badge URL still has `OWNER/REPO` placeholder. Replace with your actual GitHub path before merging.
- `dist/launchd/com.user.ios-graphrag.watcher.plist` has placeholder paths. Per-user install rewrites them; opt-in only.
- `engine/CONNECTION_GUIDE.md` Offline Model section has `<!-- TODO: org-specific Artifactory/S3 URL -->` placeholder. Fill in if your team stages the model on internal infra.
- 3 stale `worktree-agent-*` local branches from Phase 0 subagents. Run `git worktree prune` and `git branch -D worktree-agent-{a10f62644555f58f3,a4c219ff65e9cb5b9,af1b46c4a63fe9a5c}` to clean up if they bug you. Harmless otherwise.
- `tests/test_preflight_passes_on_smoke_install` is `pytest.mark.skipif`-gated on the HF model cache; it'll skip on a fresh CI runner. Correct behavior.

---

## 8. If you want to resume the loop after empirical work

When you have the benchmark JSON + parse-audit JSON, paste them in chat and say "resume Phase 4 / 4.5". I'll dispatch the next round of subagents to execute the conditional phases.

If you don't want to track me in chat — the commit history + reviews are all on the branch, and `docs/PRODUCTION_ROADMAP.md` has the acceptance checklist with everything in scope.

---

## 9. Useful one-liners for daily work

```bash
# Index a repo
uv run ios-graphrag-index --repo /path/to/repo --db ~/.cache/ios-graphrag/graph.sqlite --full

# Start the server (Copilot/Claude Desktop will spawn this)
GRAPH_DB_PATH=~/.cache/ios-graphrag/graph.sqlite uv run ios-graphrag-server

# Watch a repo for changes (incremental reindex on debounce)
uv run ios-graphrag-watch --repo /path/to/repo --db ~/.cache/ios-graphrag/graph.sqlite

# Health check
uv run ios-graphrag-doctor

# Detailed bug report (paste into GitHub issue)
uv run ios-graphrag-doctor --bug-report > bug-report.txt

# Verify integrity (orphan edges, missing files)
uv run ios-graphrag-doctor --verify

# Tail recent errors with trace_id
uv run ios-graphrag-doctor --tail-errors 30

# Find a trace_id from a tool error
uv run ios-graphrag-doctor --tail-errors 50 | grep <trace_id>

# Inspect symbol counts
sqlite3 ~/.cache/ios-graphrag/graph.sqlite \
  "SELECT symbol_type, COUNT(*) FROM nodes GROUP BY symbol_type ORDER BY 2 DESC"

# Inspect SwiftUI views
sqlite3 ~/.cache/ios-graphrag/graph.sqlite \
  "SELECT symbol_name, file_path FROM nodes WHERE is_swiftui_view=1"

# Inspect state-kind dependencies
sqlite3 ~/.cache/ios-graphrag/graph.sqlite \
  "SELECT state_kind, COUNT(*) FROM nodes WHERE state_kind IS NOT NULL GROUP BY state_kind"
```

---

That's everything. Ping me when you have empirical results or want to redirect.
