"""
Indexer stdout cleanliness regression test.

Locks in P1.5: the indexer must NEVER write to stdout. The MCP server
multiplexes its stdio JSON-RPC stream over the same pid; a stray
``print()`` from the indexer (e.g. when invoked from a server-side
re-index path) would corrupt the protocol stream and crash any client
parsing it.

The previous implementation peppered ``print()`` calls through
``index_repository`` for human progress. P1.5 routed all of that
through the ``logging`` module on stderr. This test runs the indexer
as a subprocess and asserts ``stdout`` stays empty.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# Path to the fixture project under test. Mirrors ``conftest.TEST_FIXTURES_DIR``
# but recomputed locally because ``tests/`` is not a Python package
# (pytest auto-discovers it without requiring an ``__init__.py``).
_TEST_FIXTURES_DIR = Path(__file__).parent.parent / "test_fixtures" / "CalculatorApp"


def test_indexer_stdout_is_empty(tmp_path: Path):
    """Run the indexer end-to-end and assert it writes nothing to stdout.

    Uses a private ``tmp_path``-scoped DB rather than the
    session-scoped ``indexed_db_path`` fixture because we want a clean
    one-shot subprocess invocation that matches how the MCP server
    might shell out.

    Phase 6b: ``GRAPHRAG_LOG_DIR`` redirects the indexer's rotating log
    file out of ``~/Library/Logs/ios-graphrag/`` so this test does not
    pollute the user's Library directory on every run.
    """
    tmp_db = tmp_path / "stdout-clean-test.sqlite"
    project_root = Path(__file__).parent.parent

    import os as _os
    env = _os.environ.copy()
    env["GRAPHRAG_LOG_DIR"] = str(tmp_path / "logs")

    result = subprocess.run(
        [
            sys.executable,
            "-m", "ios_graphrag.indexer",
            "--repo", str(_TEST_FIXTURES_DIR),
            "--db", str(tmp_db),
            "--full",
        ],
        cwd=str(project_root),
        env=env,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, (
        f"Indexer exited non-zero ({result.returncode}). "
        f"stderr (last 2KB): {result.stderr[-2048:]!r}"
    )

    assert result.stdout == b"", (
        f"Indexer leaked output to stdout. This will corrupt the MCP "
        f"server's JSON-RPC stream if the server ever shells out to the "
        f"indexer. stdout (first 2KB): {result.stdout[:2048]!r}"
    )
