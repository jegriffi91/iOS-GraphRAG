"""Smoke tests for the [project.scripts] console-script entry points.

Phase 0 added two console scripts via pyproject.toml:
    ios-graphrag-index   -> ios_graphrag.indexer:main
    ios-graphrag-server  -> ios_graphrag.server:main

These tests catch regressions where an entry-point reference breaks
(e.g. renamed module, removed main, import-time error in the target).
"""
import os
import subprocess
import sys


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
    assert "usage" in result.stdout.lower()


def test_server_module_imports(tmp_path):
    # server main() has no --help (it loads the DB and calls mcp.run()),
    # so we verify the module imports cleanly. Catches a broken
    # ios-graphrag-server entry-point reference or any import-time
    # failure in server.py.
    env = os.environ.copy()
    env["GRAPH_DB_PATH"] = str(tmp_path / "kg.sqlite")
    result = subprocess.run(
        [sys.executable, "-c", "import ios_graphrag.server"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"importing ios_graphrag.server failed (rc={result.returncode}):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
