# iOS-GraphRAG Runbook

When something is broken, check this list before deep-diving. Most issues
have a known remediation that's faster than reading code.

For general bug-reporting, see `docs/REPORTING.md`. For known limitations
that aren't bugs, see `docs/LIMITATIONS.md`.

## 1. Server crashes immediately on startup

**Symptom:** `ios-graphrag-server` exits with code 1 within 1-2 seconds.

**Diagnose:**
```
uv run ios-graphrag-doctor
```
Look for `[MISSING]` or `[ERROR]` lines. Common cause: schema version mismatch.

**Remediate:**
- DB at older version → `uv run ios-graphrag-index --repo <repo> --db <db>` to apply migrations.
- DB at newer version → `uv tool upgrade ios-graphrag` (the binary expects an older schema than the data).
- DB missing → set `GRAPH_DB_PATH` correctly OR run the indexer to create one.

## 2. Semantic search returns nonsense / unrelated results

**Symptom:** `global_codebase_search` returns symbols whose names have no relationship to the query.

**Diagnose:**
```
uv run ios-graphrag-doctor
```
Check the `Embedding model cache` line. If `[STALE]`, the model snapshot is partial.

**Remediate:**
```
rm -rf ~/.cache/huggingface/hub/models--nomic-ai--nomic-embed-text-v1.5
uv run ios-graphrag-preflight   # re-downloads model
uv run ios-graphrag-index --repo <repo> --db <db> --full   # re-embeds
```

## 3. Indexer hangs (no progress for >5 min)

**Symptom:** `ios-graphrag-index --full` produces no log output for many minutes.

**Diagnose:**
```
uv run ios-graphrag-doctor --tail-errors 30
```
Look for repeated parser-timeout warnings on the same file. Tree-sitter can
hit pathological parsing on certain malformed Swift files.

**Remediate:**
- Identify the file from log output, manually exclude it (move it temporarily out of the repo).
- Re-run the indexer.
- File a bug per `docs/REPORTING.md` so the parser can be hardened.

## 4. `Database at vN, server expects vM` error

**Symptom:** Server exits with this exact message on startup.

**Diagnose:** None needed — message is self-describing.

**Remediate:**
- vN < vM (DB older than code): run the indexer once to apply migrations.
- vN > vM (DB newer): upgrade `ios-graphrag` to a version that knows the schema.

## 5. `TLS verification disabled by GRAPHRAG_INSECURE_TLS=1` warning

**Symptom:** This warning appears in logs at every startup.

**Diagnose:**
```
echo $GRAPHRAG_INSECURE_TLS
```

**Remediate:** If the bypass isn't intentional, `unset GRAPHRAG_INSECURE_TLS`. Use
`--cert-bundle /path/to/corp-bundle.pem` instead — verification stays on.

## 6. Indexer fails with `ModuleNotFoundError`

**Symptom:** `ModuleNotFoundError: No module named 'sentence_transformers'` (or similar).

**Diagnose:** Are you using the right environment?
```
which uv && uv run python -c "import sys; print(sys.executable)"
```

**Remediate:** `cd <repo>; uv sync` to install dependencies into the project venv.
The console scripts are intended to be invoked via `uv run` (or after `uv tool install`).

## 7. `huggingface-cli download` fails behind corporate proxy

**Symptom:** Network error during preflight or indexer startup.

**Diagnose:**
```
echo $REQUESTS_CA_BUNDLE
echo $HF_HUB_DISABLE_HTTPX  # 1 forces requests-based client
```

**Remediate:** Stage the model offline per `engine/CONNECTION_GUIDE.md` →
"Offline / Pre-staged Model" section. Set `IOS_GRAPHRAG_MODEL_DIR` to point at
the staged copy.

## 8. `tool call error` in server logs with no obvious cause

**Symptom:** A specific tool returns an error, but stderr only shows
`[ERROR] tool call error` with no traceback.

**Diagnose:** Find the trace_id in the error response payload. Then:
```
uv run ios-graphrag-doctor --tail-errors 50 | grep <trace_id>
```

**Remediate:** Match the failure to one of the cases above. If novel,
file per `docs/REPORTING.md` with the trace_id and the relevant log lines.

## 9. Crashdump file appeared in `~/Library/Logs/ios-graphrag/`

**Symptom:** `crashdump-<timestamp>-<pid>.txt` exists.

**Diagnose:** Open the file. The "Recent calls" section shows what the
server was doing before the crash. The "Traceback" section pinpoints the
failing frame.

**Remediate:**
- For known issues (#1-#8 above), apply the corresponding remediation.
- For novel issues: file per `docs/REPORTING.md` with the entire crashdump
  (it's already redacted).
- After resolving, the dump can be safely deleted.

## 10. Pinned to which version?

**Symptom:** Teammate is on a different version and it's misbehaving.

**Diagnose:**
```
uv run python -c "from importlib.metadata import version; print(version('ios-graph-rag'))"
```

**Remediate:** Team members should `uv tool install` from the **current
stable tag** (see `CHANGELOG.md` for the latest). Don't track `main` for
day-to-day use — main may be ahead of the validated/released set.

---

If your issue isn't here, see `docs/REPORTING.md` for the bug-report
flow. Most novel issues land here within a release.
