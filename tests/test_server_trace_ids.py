"""Tests for the Phase 6b MCP tool trace ID decorator.

Coverage:

1. A successful tool call emits ``tool call begin`` and ``tool call end``
   log records sharing the same ``trace_id``.
2. A failed tool call emits a ``tool call error`` record carrying the
   ``trace_id`` and ``duration_ms``.

We exercise the actual handlers (``_trace_dependencies``) with the
indexed fixture DB. ``caplog`` captures records on the root logger; the
decorator passes ``trace_id`` and friends via ``extra=`` so they live on
the LogRecord and are accessible in the captured records.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest


def test_handler_logs_trace_id_on_entry_and_exit(
    indexed_db_path: Path, caplog
):
    """A normal tool call logs begin + end records sharing one trace_id."""
    from ios_graphrag import server

    server.GRAPH.clear()
    server.NODE_META.clear()
    with patch.dict(os.environ, {"GRAPH_DB_PATH": str(indexed_db_path)}):
        server.load_graph(str(indexed_db_path))

        # Pick an arbitrary file path that's actually in the graph so the
        # handler returns a non-error result.
        conn = sqlite3.connect(str(indexed_db_path))
        try:
            row = conn.execute(
                "SELECT file_path FROM nodes LIMIT 1"
            ).fetchone()
        finally:
            conn.close()
        assert row is not None
        file_path = row[0]

        with caplog.at_level(logging.INFO, logger=""):
            result = server._trace_dependencies(file_path=file_path)

    assert isinstance(result, dict)
    # Decorator surfaces the trace_id back into the response payload.
    assert "trace_id" in result
    payload_trace = result["trace_id"]
    assert isinstance(payload_trace, str) and len(payload_trace) == 8

    begin_records = [r for r in caplog.records if r.message == "tool call begin"]
    end_records = [r for r in caplog.records if r.message == "tool call end"]
    assert begin_records, "expected at least one 'tool call begin' record"
    assert end_records, "expected at least one 'tool call end' record"

    # Find the matching pair for THIS call by trace_id.
    matching_begin = next(
        (r for r in begin_records if getattr(r, "trace_id", None) == payload_trace),
        None,
    )
    matching_end = next(
        (r for r in end_records if getattr(r, "trace_id", None) == payload_trace),
        None,
    )
    assert matching_begin is not None, (
        f"no 'tool call begin' record carries trace_id={payload_trace!r}"
    )
    assert matching_end is not None, (
        f"no 'tool call end' record carries trace_id={payload_trace!r}"
    )
    assert matching_begin.tool == "swift_dependency_tracer"
    assert matching_end.tool == "swift_dependency_tracer"
    # duration_ms is attached to the end record only.
    assert hasattr(matching_end, "duration_ms")
    assert isinstance(matching_end.duration_ms, (int, float))


def test_handler_logs_trace_id_on_error(indexed_db_path: Path, caplog):
    """A handler that raises must log a 'tool call error' record with trace_id.

    We force an error by passing byte ranges that exceed the file's
    length to ``_read_symbol``. Actually, _read_symbol returns a dict
    with ``error`` for FileNotFoundError, but otherwise raises -- so we
    induce an unhandled exception via a PermissionError-like path.
    The simplest reliable failure is to mock ``builtins.open`` to raise.
    """
    from ios_graphrag import server

    server.GRAPH.clear()
    server.NODE_META.clear()
    with patch.dict(os.environ, {"GRAPH_DB_PATH": str(indexed_db_path)}):
        server.load_graph(str(indexed_db_path))

    # Patch ``open`` inside the server module so _read_symbol's open()
    # call raises something non-FileNotFoundError -> falls past the
    # try/except -> propagates -> the decorator logs and re-raises.
    def _bad_open(*args, **kwargs):
        raise OSError("synthetic disk failure")

    with caplog.at_level(logging.INFO, logger=""):
        with patch("ios_graphrag.server.open", _bad_open, create=True):
            with pytest.raises(OSError):
                server._read_symbol(
                    file_path="/tmp/does-not-matter",
                    start_byte=0,
                    end_byte=10,
                )

    error_records = [r for r in caplog.records if r.message == "tool call error"]
    assert error_records, "expected a 'tool call error' record"
    rec = error_records[-1]
    assert getattr(rec, "trace_id", None), "error record missing trace_id"
    assert isinstance(rec.trace_id, str) and len(rec.trace_id) == 8
    assert hasattr(rec, "duration_ms")
    assert isinstance(rec.duration_ms, (int, float))
    assert rec.tool == "read_symbol_source"
