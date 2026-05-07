"""
Per-file crash isolation regression test.

Locks in P1.4: ``index_repository`` must never abort the whole run when
a single file in the tree is malformed. We build a tiny tmpdir repo
that contains one valid Swift file alongside several deliberately
hostile inputs (binary garbage, empty file, UTF-8 BOM, very long
single line) and assert:

  * the indexer subprocess exits 0 (a non-zero exit means a malformed
    file killed the run — that's the regression this guards against);
  * the on-disk SQLite DB exists and contains rows for the valid file;
  * ``indexing_errors.log`` either contains the run's structured log
    summary OR (for cases that flow through cleanly without raising
    Python exceptions, e.g. an empty file with zero symbols) the run
    completes silently — which is also acceptable behavior.

Cases that the parser handles gracefully (zero-symbol files, BOM, very
long lines) won't actually exercise the try/except wrap — they just
parse to empty symbols. That's still a useful regression: if any of
these cases starts crashing in the future, this test will catch it.
"""
from __future__ import annotations

import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest


_PROJECT_ROOT = Path(__file__).parent.parent
_VALID_SWIFT_SOURCE = (
    _PROJECT_ROOT / "test_fixtures" / "CalculatorApp" / "Sources"
    / "CalculatorApp" / "Calculator.swift"
)


def _build_hostile_repo(repo_root: Path) -> dict:
    """Populate ``repo_root`` with one valid Swift file and four
    deliberately malformed neighbors. Return a dict of label -> path so
    individual assertions can reference each input.
    """
    repo_root.mkdir(parents=True, exist_ok=True)
    sources = repo_root / "Sources"
    sources.mkdir()

    # Valid Swift file: a verbatim copy of the existing fixture's
    # Calculator.swift. Gives the indexer something to succeed on.
    valid_path = sources / "Calculator.swift"
    shutil.copy(_VALID_SWIFT_SOURCE, valid_path)

    # Binary garbage with a .swift extension. Tree-sitter usually copes
    # by producing an error tree; we just need the indexer to not crash.
    binary_path = sources / "binary_garbage.swift"
    binary_path.write_bytes(bytes(range(256)) * 4)

    # Empty file. read_file_text might return an empty bytes/str pair;
    # the parser should produce zero symbols.
    empty_path = sources / "empty.swift"
    empty_path.write_bytes(b"")

    # UTF-8 BOM at the start of an otherwise valid file. Some legacy
    # Swift sources have this. Tree-sitter Swift's tokenizer usually
    # treats the BOM as whitespace.
    bom_path = sources / "with_bom.swift"
    bom_path.write_bytes(b"\xef\xbb\xbfimport Foundation\n\nfunc bom() {}\n")

    # Single very long line, no newline. Pathological for any
    # line-based fallback parser.
    long_line_path = sources / "long_line.swift"
    long_line_path.write_bytes(b"// " + b"x" * 100_001)

    return {
        "valid": valid_path,
        "binary": binary_path,
        "empty": empty_path,
        "bom": bom_path,
        "long_line": long_line_path,
    }


def _indexer_env(tmp_path: Path) -> dict:
    """Return an env dict with GRAPHRAG_LOG_DIR pointed at the tmpdir.

    Phase 6b: keeps subprocess indexer runs from rotating their JSON
    logs into the user's ``~/Library/Logs/ios-graphrag/`` directory.
    """
    import os as _os
    env = _os.environ.copy()
    env["GRAPHRAG_LOG_DIR"] = str(tmp_path / "graphrag-logs")
    return env


def test_crash_isolation_against_malformed_files(tmp_path: Path):
    """Run the indexer over a hostile tmpdir repo and assert the run
    completes with the valid file indexed."""
    repo_root = tmp_path / "hostile_repo"
    paths = _build_hostile_repo(repo_root)

    db_path = tmp_path / "test.db"

    # Subprocess invocation matches how conftest.py drives the indexer
    # for the rest of the suite — same module entry point, same flags.
    result = subprocess.run(
        [
            sys.executable,
            "-m", "ios_graphrag.indexer",
            "--repo", str(repo_root),
            "--db", str(db_path),
            "--full",
        ],
        cwd=str(_PROJECT_ROOT),
        env=_indexer_env(tmp_path),
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, (
        f"Indexer exited non-zero ({result.returncode}) — at least one "
        f"malformed file killed the run. stderr (last 4KB): "
        f"{result.stderr[-4096:]!r}"
    )

    # The DB must exist with the schema applied and rows for the valid
    # file. If the malformed inputs aborted the run before
    # update_file_index landed, the nodes table for Calculator.swift
    # would be empty.
    assert db_path.exists(), "Indexer did not create the SQLite database"
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE file_path = ?",
            (str(paths["valid"]),),
        ).fetchone()
    finally:
        conn.close()

    assert rows[0] > 0, (
        f"Expected nodes rows for the valid file at {paths['valid']!s}; "
        f"none found. stderr: {result.stderr[-2048:]!r}"
    )

    # Sanity: stderr (where the configured logger emits) should not
    # contain a fatal stack trace from the indexer's setup phase. We
    # don't assert on stderr content beyond this — the per-file error
    # logs go to indexing_errors.log via the FileHandler attached in
    # main(), which lives at the project root and is shared with
    # other test runs. Asserting on its absolute content here would
    # flake.
    stderr_text = result.stderr.decode("utf-8", errors="replace")
    fatal_markers = ("Traceback (most recent call last):", "OSError", "MemoryError")
    forbidden = [m for m in fatal_markers if m in stderr_text]
    assert not forbidden, (
        f"Indexer stderr contains fatal-looking markers {forbidden}; "
        f"stderr (last 4KB): {stderr_text[-4096:]!r}"
    )


def test_crash_isolation_run_summary(tmp_path: Path):
    """Sanity-check that the run summary log line lands on stderr.

    Phase 1 P1.4 added a final ``log.info`` of the form
    ``"Indexing complete in Xs: N files indexed, M files failed"``.
    This lets ops grep for the indexed/failed split in CI logs.
    """
    repo_root = tmp_path / "hostile_repo"
    _build_hostile_repo(repo_root)

    db_path = tmp_path / "test.db"

    result = subprocess.run(
        [
            sys.executable,
            "-m", "ios_graphrag.indexer",
            "--repo", str(repo_root),
            "--db", str(db_path),
            "--full",
        ],
        cwd=str(_PROJECT_ROOT),
        env=_indexer_env(tmp_path),
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    stderr_text = result.stderr.decode("utf-8", errors="replace")
    assert "files indexed" in stderr_text and "files failed" in stderr_text, (
        f"Expected run summary 'N files indexed, M files failed' on "
        f"stderr; got (last 4KB): {stderr_text[-4096:]!r}"
    )
