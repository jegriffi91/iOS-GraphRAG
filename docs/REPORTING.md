# Reporting Bugs in iOS-GraphRAG

When something goes wrong, the project ships diagnostics that produce
copy-pasteable reports. Use them rather than describing the bug from memory.

## If the server crashes

1. Find the most recent crashdump:
   ```
   ls -t ~/Library/Logs/ios-graphrag/crashdump-*.txt | head -1
   ```
2. Open a GitHub issue. Paste the entire dump (it's already redacted).
3. Include reproduction steps if you have them.

## If a tool returns an error but the server keeps running

1. Note the `trace_id` in the error response payload.
2. Run:
   ```
   uv run ios-graphrag-doctor --tail-errors 30 | grep <trace_id>
   ```
3. Paste those log lines into the issue.

## If you're not sure what's wrong (general health check)

1. Run:
   ```
   uv run ios-graphrag-doctor --bug-report > bug-report.txt
   ```
2. Open a GitHub issue and attach `bug-report.txt`.

## What's redacted

The `--bug-report` and crashdump outputs replace `/Users/<your-username>/`
with `~/` and pull only env vars whose names match `GRAPHRAG_*`, `HF_*`,
or contain `SSL` / `CA_BUNDLE`. Values aren't otherwise filtered — review
the dump before posting if your machine has unusual env vars.

## Issue template

When opening an issue, include:

- iOS-GraphRAG version (from `pyproject.toml` or `git log -1`).
- macOS version (`sw_vers -productVersion`).
- Python version (`python --version`).
- The crashdump or bug-report output (above).
- Repro steps if available.
- The exact AI client that triggered the issue (Copilot CLI, VS Code Copilot,
  Claude Desktop, …).
