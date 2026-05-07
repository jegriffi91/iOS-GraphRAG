"""Tests for the Phase 6b centralized logging module.

Coverage:

1. ``setup_logging`` creates the configured log directory and file.
2. The JSON file handler emits parseable NDJSON with caller-attached
   ``extra=`` fields preserved.
3. The stderr handler emits human-readable text (NOT JSON) so the
   indexer's interactive output and ``test_indexer_stdout`` invariants
   stay intact.
4. ``setup_logging`` is idempotent: calling it twice still leaves the
   root logger with exactly two handlers (stderr + rotating file).
"""
from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import pytest

from ios_graphrag import _logging


@pytest.fixture(autouse=True)
def _isolate_root_logger():
    """Snapshot/restore the root logger across each test.

    ``setup_logging`` deliberately mutates the root logger's handlers, so
    each test has to start from a known state and not bleed into others.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    yield
    for handler in list(root.handlers):
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    for handler in saved_handlers:
        root.addHandler(handler)
    root.setLevel(saved_level)


def test_setup_logging_creates_log_dir_and_file(tmp_path: Path, monkeypatch):
    """The log directory + per-component file are created on demand.

    ``GRAPHRAG_LOG_DIR`` redirects the file out of the user's
    ``~/Library/Logs/ios-graphrag/`` so tests stay hermetic.
    """
    log_dir = tmp_path / "logs"
    monkeypatch.setenv("GRAPHRAG_LOG_DIR", str(log_dir))

    log_file = _logging.setup_logging("test")

    assert log_dir.is_dir(), "setup_logging should create the log dir"
    assert log_file == log_dir / "test.log"
    assert log_file.exists(), "RotatingFileHandler should open the file eagerly"


def test_json_formatter_roundtrip(tmp_path: Path, monkeypatch):
    """A record with extras should round-trip through JsonLineFormatter.

    Verifies the file handler emits one JSON object per record with the
    base fields (``ts``, ``level``, ``logger``, ``message``) plus any
    ``extra=`` fields the caller attached (here: ``trace_id`` and
    ``duration_ms``).
    """
    log_dir = tmp_path / "logs"
    monkeypatch.setenv("GRAPHRAG_LOG_DIR", str(log_dir))

    log_file = _logging.setup_logging("rt")
    log = logging.getLogger("ios_graphrag.test.rt")
    log.info(
        "tool call end",
        extra={"trace_id": "abc12345", "duration_ms": 42.5, "tool": "fake"},
    )

    # RotatingFileHandler buffers in memory until flush; force one.
    for handler in logging.getLogger().handlers:
        handler.flush()

    raw = log_file.read_text(encoding="utf-8")
    lines = [line for line in raw.splitlines() if line.strip()]
    assert lines, f"expected at least one JSON line in {log_file}, got {raw!r}"
    last = json.loads(lines[-1])
    assert last["level"] == "INFO"
    assert last["logger"] == "ios_graphrag.test.rt"
    assert last["message"] == "tool call end"
    assert last["trace_id"] == "abc12345"
    assert last["duration_ms"] == 42.5
    assert last["tool"] == "fake"
    # Sanity: ts must be ISO-8601 with millisecond precision
    # (e.g. "2026-05-07T12:34:56.789"). %f is not honored by strftime
    # against a struct_time, so the formatter computes ms manually --
    # this assertion catches regressions to the literal "%f" form.
    import re as _re
    assert "ts" in last
    assert _re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}$", last["ts"]), (
        f"ts is not ISO-8601 with milliseconds: {last['ts']!r}"
    )


def test_text_formatter_on_stderr(tmp_path: Path, monkeypatch, capfd):
    """stderr handler must emit human-readable text, not JSON.

    The MCP harness captures stderr and surfaces it in error contexts;
    the indexer's stdout-clean test passes because logs go to stderr.
    Both consumers expect text, not JSON.
    """
    log_dir = tmp_path / "logs"
    monkeypatch.setenv("GRAPHRAG_LOG_DIR", str(log_dir))

    _logging.setup_logging("text")
    log = logging.getLogger("ios_graphrag.test.text")
    log.warning("hello text on stderr")

    for handler in logging.getLogger().handlers:
        handler.flush()

    captured = capfd.readouterr()
    assert "hello text on stderr" in captured.err
    # Whatever stderr emitted MUST not be JSON. If anyone tried to
    # parse a stderr line as JSON it would deserialize cleanly.
    for line in captured.err.splitlines():
        if "hello text on stderr" not in line:
            continue
        # Confirm the line is NOT a JSON object: text formatter prefix is
        # "<timestamp> [LEVEL] logger: message" — not a JSON {...}.
        assert not line.lstrip().startswith("{"), (
            f"stderr line should be text, not JSON: {line!r}"
        )
        # And confirm the human-readable shape is present.
        assert "[WARNING]" in line
        assert "ios_graphrag.test.text" in line


def test_setup_logging_is_idempotent(tmp_path: Path, monkeypatch):
    """Calling setup_logging twice must not stack duplicate handlers.

    Worker subprocesses re-import this module + re-call setup_logging;
    the contract is exactly two handlers (stderr + rotating file).
    """
    log_dir = tmp_path / "logs"
    monkeypatch.setenv("GRAPHRAG_LOG_DIR", str(log_dir))

    _logging.setup_logging("idem")
    _logging.setup_logging("idem")

    root = logging.getLogger()
    assert len(root.handlers) == 2, (
        f"expected exactly 2 root handlers (stderr + file); got "
        f"{len(root.handlers)}: {root.handlers!r}"
    )
    handler_types = {type(h).__name__ for h in root.handlers}
    assert "StreamHandler" in handler_types
    assert "RotatingFileHandler" in handler_types
