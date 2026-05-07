"""Tests for the Phase 6b ``ios-graphrag-doctor --tail-errors N`` flag.

Coverage:

1. With a synthetic ``indexer.log`` of mixed JSON levels, only the last
   N WARNING+/ERROR records appear, INFO is filtered out.
2. With ``GRAPHRAG_LOG_DIR`` pointing at an empty tmpdir, both files
   produce the friendly "(no log file at ...)" stub instead of crashing.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent


def _make_record(level: str, message: str, trace_id: str | None = None) -> str:
    record = {
        "ts": "2026-05-07T12:34:56.789",
        "level": level,
        "logger": "ios_graphrag.indexer",
        "message": message,
    }
    if trace_id is not None:
        record["trace_id"] = trace_id
    return json.dumps(record)


def test_tail_errors_from_indexer_log(tmp_path: Path):
    """Synthetic indexer.log -> only WARNING+/ERROR last-N records appear.

    Writes 5 NDJSON records (mixed INFO/WARNING/ERROR), runs
    ``ios-graphrag-doctor --tail-errors 3``, asserts only the last 3
    non-INFO records appear in order.
    """
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    indexer_log = log_dir / "indexer.log"
    server_log = log_dir / "server.log"

    indexer_log.write_text(
        "\n".join(
            [
                _make_record("INFO", "ignored info 1"),
                _make_record("WARNING", "warn alpha"),
                _make_record("ERROR", "err beta", trace_id="aaaa1111"),
                _make_record("INFO", "ignored info 2"),
                _make_record("WARNING", "warn gamma"),
                _make_record("ERROR", "err delta", trace_id="bbbb2222"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    # Empty server.log so the missing-file branch isn't exercised here.
    server_log.write_text("", encoding="utf-8")

    env = os.environ.copy()
    env["GRAPHRAG_LOG_DIR"] = str(log_dir)

    result = subprocess.run(
        ["uv", "run", "ios-graphrag-doctor", "--tail-errors", "3"],
        cwd=str(_PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"doctor rc={result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )

    out = result.stdout
    # Header for each file
    assert "Last 3 errors from indexer.log" in out
    assert "Last 3 errors from server.log" in out

    # The last 3 non-INFO records, in order: err beta, warn gamma, err delta.
    # Verifying ordering plus that INFO lines were filtered.
    assert "err beta" in out
    assert "warn gamma" in out
    assert "err delta" in out
    assert "ignored info 1" not in out
    assert "ignored info 2" not in out
    # The first WARNING ("warn alpha") was earlier than the last 3.
    assert "warn alpha" not in out

    # trace_id should surface for records that have one.
    assert "trace_id=bbbb2222" in out


def test_tail_errors_handles_missing_log(tmp_path: Path):
    """Empty log dir -> friendly missing-file stub for both files."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    env = os.environ.copy()
    env["GRAPHRAG_LOG_DIR"] = str(empty_dir)

    result = subprocess.run(
        ["uv", "run", "ios-graphrag-doctor", "--tail-errors", "5"],
        cwd=str(_PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0
    out = result.stdout
    # Both files should print the no-log-file stub.
    assert "(no log file at" in out
    assert "indexer.log" in out
    assert "server.log" in out
