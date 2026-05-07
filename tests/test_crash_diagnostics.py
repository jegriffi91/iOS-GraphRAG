"""Phase 6d crash diagnostics tests.

Coverage:

1. The ring buffer caps at ``GRAPHRAG_RECENT_CALLS_LIMIT`` and keeps
   only the most recent N entries.
2. Concurrent ``_record_call`` from multiple threads is safe and
   honours the cap.
3. Invoking the excepthook writes a crashdump file under the
   configured log dir; the file contains the expected sections.
4. Home paths in tool args are redacted to ``~`` before they hit disk.
5. A non-writable log dir does not propagate the crash-write failure;
   the original exception's chain remains intact.
6. Handled exceptions inside a tool handler (caught by the
   ``_traced_handler`` decorator) do NOT trigger a crashdump.

Tests are designed to be hermetic: each one points
``GRAPHRAG_LOG_DIR`` at a tmpdir to keep the user's
``~/Library/Logs/ios-graphrag/`` directory clean.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Ring buffer behaviour
# ---------------------------------------------------------------------------


def test_recent_calls_ring_buffer_caps_at_limit(monkeypatch):
    """Manually appending 5 entries to a buffer of size 3 keeps the last 3."""
    monkeypatch.setenv("GRAPHRAG_RECENT_CALLS_LIMIT", "3")
    from ios_graphrag import _diagnostics

    # Rebuild the buffer under the new cap. The module reads the env
    # var at import time; tests need an explicit reset to honour a
    # mid-run override.
    _diagnostics._reset_recent_calls()
    assert _diagnostics._snapshot_recent_calls() == []

    for i in range(5):
        _diagnostics._record_call(
            trace_id=f"trace{i:03d}",
            tool="dummy_tool",
            tool_args={"i": i},
            status="ok",
            duration_ms=1.0 + i,
        )

    snapshot = _diagnostics._snapshot_recent_calls()
    assert len(snapshot) == 3, f"buffer cap=3 should keep 3 entries, got {len(snapshot)}"
    # deque.maxlen evicts oldest first, so we keep entries 2,3,4.
    trace_ids = [e["trace_id"] for e in snapshot]
    assert trace_ids == ["trace002", "trace003", "trace004"]
    # Reset under the default cap so subsequent tests aren't poisoned.
    monkeypatch.delenv("GRAPHRAG_RECENT_CALLS_LIMIT", raising=False)
    _diagnostics._reset_recent_calls()


def test_recent_calls_thread_safe(monkeypatch):
    """10 threads x 100 appends each: no exceptions, length capped."""
    monkeypatch.setenv("GRAPHRAG_RECENT_CALLS_LIMIT", "20")
    from ios_graphrag import _diagnostics

    _diagnostics._reset_recent_calls()

    errors: list[BaseException] = []

    def worker(worker_id: int) -> None:
        try:
            for i in range(100):
                _diagnostics._record_call(
                    trace_id=f"w{worker_id}-{i}",
                    tool="dummy_tool",
                    tool_args={"worker_id": worker_id, "i": i},
                    status="ok",
                    duration_ms=0.1,
                )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert errors == [], f"thread workers raised: {errors}"
    snapshot = _diagnostics._snapshot_recent_calls()
    # Cap is the only invariant we can assert; the exact ordering of the
    # 20 winners is non-deterministic across schedulers.
    assert len(snapshot) <= 20
    assert len(snapshot) == 20  # 1000 appends > 20 cap, must be saturated.

    monkeypatch.delenv("GRAPHRAG_RECENT_CALLS_LIMIT", raising=False)
    _diagnostics._reset_recent_calls()


# ---------------------------------------------------------------------------
# Crashdump hook
# ---------------------------------------------------------------------------


def _synthesize_exc_info():
    """Build a real ``(type, value, tb)`` triple without raising at
    excepthook level. The hook function takes exc_info as args, so we
    just need a representative tb -- raising and catching is enough.
    """
    try:
        raise RuntimeError("synthetic crash for tests")
    except RuntimeError as exc:
        return type(exc), exc, exc.__traceback__


def test_crashdump_writes_on_excepthook(tmp_path, monkeypatch):
    """Calling the excepthook with synthetic exc_info writes a dump file."""
    monkeypatch.setenv("GRAPHRAG_LOG_DIR", str(tmp_path))
    from ios_graphrag import _diagnostics

    _diagnostics._reset_recent_calls()
    # Drop a couple of recent calls so the dump's "Recent calls" section
    # has content rather than the "(no recent calls)" stub.
    _diagnostics._record_call(
        trace_id="abc12345",
        tool="global_codebase_search",
        tool_args={"query": "AuthState"},
        status="ok",
        duration_ms=12.5,
    )
    _diagnostics._record_call(
        trace_id="def67890",
        tool="read_symbol_source",
        tool_args={"file_path": "/tmp/foo.swift", "start_byte": 0, "end_byte": 100},
        status="error",
        duration_ms=3.4,
        error="OSError: synthetic",
    )

    exc_info = _synthesize_exc_info()
    # Call the hook directly. We bypass install_crashdump_hook() so the
    # "previously installed hook" branch doesn't fire (which would call
    # sys.__excepthook__ -> stderr noise during pytest output).
    _diagnostics._crashdump_excepthook(*exc_info)

    dumps = sorted(tmp_path.glob("crashdump-*.txt"))
    assert len(dumps) == 1, f"expected exactly one crashdump under {tmp_path}, got {dumps}"
    text = dumps[0].read_text(encoding="utf-8")

    # Section markers and content sanity checks.
    assert "=== iOS-GraphRAG crashdump ===" in text
    assert "=== Traceback ===" in text
    assert "RuntimeError" in text
    assert "synthetic crash for tests" in text
    assert "=== Process info ===" in text
    assert "=== Dependency versions ===" in text
    assert "=== Environment ===" in text
    assert "=== Recent calls ===" in text
    # Both recorded calls should appear by trace_id.
    assert "abc12345" in text
    assert "def67890" in text


def test_crashdump_redacts_home_paths(tmp_path, monkeypatch):
    """A recent call's args containing the user's HOME must be redacted."""
    monkeypatch.setenv("HOME", "/Users/testuser")
    monkeypatch.setenv("GRAPHRAG_LOG_DIR", str(tmp_path))
    from ios_graphrag import _diagnostics

    _diagnostics._reset_recent_calls()
    # File path inside HOME -- exact prefix the redactor must catch.
    _diagnostics._record_call(
        trace_id="redact01",
        tool="read_symbol_source",
        tool_args={
            "file_path": "/Users/testuser/dev/MyApp/Sources/Auth.swift",
            "start_byte": 100,
            "end_byte": 250,
        },
        status="ok",
        duration_ms=2.0,
    )

    exc_info = _synthesize_exc_info()
    _diagnostics._crashdump_excepthook(*exc_info)

    dumps = sorted(tmp_path.glob("crashdump-*.txt"))
    assert len(dumps) == 1
    text = dumps[0].read_text(encoding="utf-8")

    assert "/Users/testuser/" not in text, (
        "redaction failed: raw HOME path leaked into the crashdump"
    )
    # The redacted form should be present.
    assert "~/dev/MyApp/Sources/Auth.swift" in text


def test_crashdump_swallows_write_errors(tmp_path, monkeypatch):
    """A non-writable log dir must not raise; original exc is undisturbed."""
    # Create a read-only directory; pointing GRAPHRAG_LOG_DIR at it makes
    # the dump write fail with PermissionError. The hook must swallow it.
    ro_dir = tmp_path / "readonly"
    ro_dir.mkdir()
    ro_dir.chmod(0o400)  # r-- on owner; no write/execute.
    try:
        monkeypatch.setenv("GRAPHRAG_LOG_DIR", str(ro_dir))
        from ios_graphrag import _diagnostics

        _diagnostics._reset_recent_calls()
        exc_info = _synthesize_exc_info()
        # Capture the original exc to compare after the hook runs.
        original_exc = exc_info[1]
        original_args = original_exc.args
        # The hook must not raise. Run it; assert no propagation.
        _diagnostics._crashdump_excepthook(*exc_info)
        # The original exception's chain must remain intact -- the hook
        # only chains BaseExceptions through the previously-installed
        # excepthook, which is sys.__excepthook__ here. We confirm the
        # exception object itself wasn't mutated.
        assert original_exc.args == original_args
        # No dump file should have been written (write failed silently).
        # Some macOS configs allow root-equivalent writes; this assertion
        # is best-effort but represents the common case.
        dumps = list(ro_dir.glob("crashdump-*.txt"))
        assert dumps == [], f"unexpected dump in read-only dir: {dumps}"
    finally:
        # Restore mode so pytest can clean up the tmp dir.
        ro_dir.chmod(0o700)


def test_handler_error_does_not_write_crashdump(tmp_path, monkeypatch, indexed_db_path: Path):
    """A handled exception inside a tool handler must NOT write a dump.

    Crashdumps are reserved for excepthook-level (uncaught) failures;
    the ``_traced_handler`` decorator catches handler exceptions and
    logs them via the structured logger -- those go to indexer.log /
    server.log, not to a separate crashdump file.
    """
    monkeypatch.setenv("GRAPHRAG_LOG_DIR", str(tmp_path))
    from ios_graphrag import _diagnostics, server

    _diagnostics._reset_recent_calls()

    server.GRAPH.clear()
    server.NODE_META.clear()
    with patch.dict(os.environ, {"GRAPH_DB_PATH": str(indexed_db_path)}):
        server.load_graph(str(indexed_db_path))

    # Drive ``_read_symbol`` into raising via a patched ``open``. The
    # decorator catches the OSError, logs "tool call error", records
    # the call into the buffer, and re-raises -- but excepthook never
    # fires under pytest's stack, so no crashdump is written.
    def _bad_open(*args, **kwargs):
        raise OSError("synthetic disk failure")

    with patch("ios_graphrag.server.open", _bad_open, create=True):
        with pytest.raises(OSError):
            server._read_symbol(
                file_path="/tmp/does-not-matter",
                start_byte=0,
                end_byte=10,
            )

    # The error must NOT have produced a crashdump file.
    dumps = list(tmp_path.glob("crashdump-*.txt"))
    assert dumps == [], f"handled tool error should not write a crashdump; found {dumps}"

    # But the failed call SHOULD have been recorded in the ring buffer
    # (so a follow-on excepthook-level crash could still see the lead-up).
    snapshot = _diagnostics._snapshot_recent_calls()
    error_calls = [e for e in snapshot if e.get("status") == "error"]
    assert error_calls, "expected the failed _read_symbol call in the buffer"
    assert error_calls[-1]["tool"] == "read_symbol_source"
    assert "OSError" in (error_calls[-1].get("error") or "")
