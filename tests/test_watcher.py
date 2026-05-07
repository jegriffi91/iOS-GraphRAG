"""Tests for the Phase 4c file-system watcher daemon.

The watcher itself binds to FSEvents on macOS, which is opaque and slow
to drive end-to-end in unit tests (events arrive on a separate thread,
debouncing introduces real wall-clock waits, and a full
``index_repository`` call needs the embedding model). To stay cheap and
hermetic, these tests cover:

1. The ``DebouncedReindexer`` directly: it's just a timer + lock + set, so
   we manipulate it on the test thread and assert call counts.
2. The ``_Handler`` class: it's a thin filter on watchdog events. We
   feed it fake event objects and assert which calls reach the
   debouncer.
3. The ``main()`` entry point's argument validation: a non-existent
   repo path is the only branch worth exercising via subprocess (rest
   of main spins up a real Observer).

``indexer.index_repository`` is mocked everywhere so no model is loaded.
"""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ios_graphrag import watcher

# Short debounce keeps the timer tests under ~250 ms total.
TEST_DEBOUNCE = 0.05


def _wait_for(condition, timeout: float = 1.0, interval: float = 0.005) -> bool:
    """Spin until ``condition()`` returns truthy, or ``timeout`` elapses.

    Used in place of bare sleeps so the tests pass on slow runners
    without bloating their happy-path duration.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(interval)
    return False


def test_debouncer_collapses_within_window(tmp_path: Path) -> None:
    """A burst of events inside the debounce window must trigger ONE reindex."""
    debouncer = watcher.DebouncedReindexer(
        repo=tmp_path, db=tmp_path / "x.db", debounce=TEST_DEBOUNCE
    )
    with mock.patch.object(watcher.indexer, "index_repository") as mocked:
        for i in range(5):
            debouncer.on_change(f"{tmp_path}/file{i}.swift")
        # Wait for the timer thread to fire the flush.
        assert _wait_for(lambda: mocked.called, timeout=1.0), (
            "debouncer flush never ran within 1s"
        )
        # Give any straggler timers a moment to misfire if the cancel
        # logic is broken.
        time.sleep(TEST_DEBOUNCE * 2)
        assert mocked.call_count == 1, (
            f"expected exactly one reindex, got {mocked.call_count}"
        )
    debouncer.stop()


def test_debouncer_resets_window_on_new_event(tmp_path: Path) -> None:
    """Events arriving DURING the wait must extend the window."""
    debouncer = watcher.DebouncedReindexer(
        repo=tmp_path, db=tmp_path / "x.db", debounce=TEST_DEBOUNCE
    )
    with mock.patch.object(watcher.indexer, "index_repository") as mocked:
        debouncer.on_change(f"{tmp_path}/a.swift")
        # Wait less than the window, then send another event. The flush
        # must NOT fire from the first timer.
        time.sleep(TEST_DEBOUNCE / 2)
        debouncer.on_change(f"{tmp_path}/b.swift")
        time.sleep(TEST_DEBOUNCE / 2)
        # At this point ~one full window has passed since the first
        # event, but only half a window since the second. If reset
        # works, no flush yet.
        assert mocked.call_count == 0, (
            "second event should have reset the timer; flush fired early"
        )
        # Now wait the full window (plus slack) for the second timer.
        assert _wait_for(lambda: mocked.called, timeout=1.0)
        assert mocked.call_count == 1
    debouncer.stop()


def test_debouncer_does_not_call_after_stop(tmp_path: Path) -> None:
    """``stop()`` cancels the pending timer; no reindex must run."""
    debouncer = watcher.DebouncedReindexer(
        repo=tmp_path, db=tmp_path / "x.db", debounce=TEST_DEBOUNCE
    )
    with mock.patch.object(watcher.indexer, "index_repository") as mocked:
        debouncer.on_change(f"{tmp_path}/a.swift")
        debouncer.stop()
        # Wait well past when the flush would have fired.
        time.sleep(TEST_DEBOUNCE * 4)
        assert mocked.call_count == 0


def test_handler_filters_by_extension(tmp_path: Path) -> None:
    """``on_modified`` for a non-Swift file must NOT reach the debouncer."""
    debouncer = watcher.DebouncedReindexer(
        repo=tmp_path, db=tmp_path / "x.db", debounce=TEST_DEBOUNCE
    )
    handler = watcher._Handler(debouncer, watcher.DEFAULT_EXTENSIONS)

    spy = mock.MagicMock(wraps=debouncer.on_change)
    debouncer.on_change = spy  # type: ignore[assignment]

    # Non-source file: should be filtered out.
    handler.on_modified(
        SimpleNamespace(is_directory=False, src_path=str(tmp_path / "build.log"))
    )
    # Directory event for a Swift-named dir: also filtered.
    handler.on_modified(
        SimpleNamespace(is_directory=True, src_path=str(tmp_path / "Sources.swift"))
    )
    assert spy.call_count == 0

    # Real Swift file: should pass through.
    handler.on_modified(
        SimpleNamespace(is_directory=False, src_path=str(tmp_path / "Foo.swift"))
    )
    assert spy.call_count == 1
    debouncer.stop()


def test_handler_handles_moved_events(tmp_path: Path) -> None:
    """``on_moved`` uses ``dest_path`` (post-rename file) for filtering."""
    debouncer = watcher.DebouncedReindexer(
        repo=tmp_path, db=tmp_path / "x.db", debounce=TEST_DEBOUNCE
    )
    handler = watcher._Handler(debouncer, watcher.DEFAULT_EXTENSIONS)

    seen: list[str] = []
    debouncer.on_change = lambda p: seen.append(p)  # type: ignore[assignment]

    # Renamed FROM .txt TO .swift — dest is eligible.
    handler.on_moved(
        SimpleNamespace(
            is_directory=False,
            src_path=str(tmp_path / "Foo.txt"),
            dest_path=str(tmp_path / "Foo.swift"),
        )
    )
    # Renamed FROM .swift TO .txt — dest is NOT eligible, so handler
    # ignores it (we only reindex when the destination is a source file).
    handler.on_moved(
        SimpleNamespace(
            is_directory=False,
            src_path=str(tmp_path / "Bar.swift"),
            dest_path=str(tmp_path / "Bar.txt"),
        )
    )

    assert seen == [str(tmp_path / "Foo.swift")]
    # Lock-free reset — no real timer was scheduled because we replaced
    # ``on_change`` above with a list-append.
    debouncer._timer = None
    debouncer.stop()


def test_main_rejects_non_directory_repo(tmp_path: Path) -> None:
    """``ios-graphrag-watch --repo <missing>`` must exit 1."""
    missing = tmp_path / "does-not-exist"
    db = tmp_path / "graph.sqlite"
    result = subprocess.run(
        [
            "uv",
            "run",
            "ios-graphrag-watch",
            "--repo",
            str(missing),
            "--db",
            str(db),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 1, (
        f"expected exit 1 for missing repo dir, got {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    # Error path should mention the missing directory.
    combined = result.stdout + result.stderr
    assert "is not a directory" in combined, (
        f"expected 'is not a directory' in output, got:\n{combined}"
    )


def test_debouncer_stop_is_idempotent(tmp_path: Path) -> None:
    """Calling ``stop()`` twice must not raise."""
    debouncer = watcher.DebouncedReindexer(
        repo=tmp_path, db=tmp_path / "x.db", debounce=TEST_DEBOUNCE
    )
    debouncer.stop()
    debouncer.stop()  # second call: noop


def test_debouncer_thread_safety_under_concurrent_events(tmp_path: Path) -> None:
    """Hammer ``on_change`` from multiple threads; assert no exceptions and
    exactly one reindex once activity stops.

    Models the realistic case where watchdog posts events from its own
    thread while a developer's editor saves a stack of buffers in
    quick succession.
    """
    debouncer = watcher.DebouncedReindexer(
        repo=tmp_path, db=tmp_path / "x.db", debounce=TEST_DEBOUNCE
    )
    with mock.patch.object(watcher.indexer, "index_repository") as mocked:
        threads = []
        for tid in range(8):
            t = threading.Thread(
                target=lambda i=tid: [
                    debouncer.on_change(f"{tmp_path}/t{i}-{j}.swift") for j in range(20)
                ]
            )
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert _wait_for(lambda: mocked.called, timeout=1.0)
        time.sleep(TEST_DEBOUNCE * 2)
        # Even with 160 events from 8 threads, exactly one reindex must
        # run (the cancel-and-restart timer collapses them).
        assert mocked.call_count == 1
    debouncer.stop()
