"""Smoke tests for the [project.scripts] console-script entry points.

Phase 0 added two console scripts via pyproject.toml:
    ios-graphrag-index   -> ios_graphrag.indexer:main
    ios-graphrag-server  -> ios_graphrag.server:main

Both tests invoke the installed binary via `uv run` so a broken
entry-point reference (renamed module, removed main, import-time
error in the target, or a missing line in [project.scripts]) fails
loudly instead of slipping through.
"""
import os
import subprocess


def test_indexer_help_exits_clean():
    result = subprocess.run(
        ["uv", "run", "ios-graphrag-index", "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"ios-graphrag-index --help failed (rc={result.returncode}):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "usage: ios-graphrag-index" in result.stdout.lower()


def test_server_entry_point_fatal_on_missing_db(tmp_path):
    # Server main() has no --help; it loads the graph DB and calls
    # mcp.run() (blocking on stdio). To exercise the entry point
    # end-to-end without hanging, point GRAPH_DB_PATH at a missing
    # file: server logs FATAL and exits 1. This proves the full chain
    # binary -> [project.scripts] -> ios_graphrag.server:main runs.
    env = os.environ.copy()
    env["GRAPH_DB_PATH"] = str(tmp_path / "missing.sqlite")
    result = subprocess.run(
        ["uv", "run", "ios-graphrag-server"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 1, (
        f"ios-graphrag-server expected rc=1 (DB missing), got rc={result.returncode}:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "FATAL" in result.stderr, (
        f"expected FATAL in stderr, got:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
