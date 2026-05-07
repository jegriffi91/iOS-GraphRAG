"""Tests for the Phase 4f.2 mtime-based auto-reload in
``ios_graphrag.server``.

Behavior under test:
1. ``load_graph`` records the DB file's mtime in ``server._LOADED_MTIME``.
2. ``_reload_if_stale`` compares the current mtime against the recorded
   one and, if it advanced, calls ``load_graph`` again and emits an INFO
   log. If unchanged, it is a silent no-op.

These tests use the indexed Calculator fixture from ``conftest.py``, copy
the resulting DB into ``tmp_path`` so each test owns its own file (so
mtime mutation doesn't leak across the session-scoped fixture), and then
manipulate the mtime via ``os.utime``.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from ios_graphrag import server


@pytest.fixture
def isolated_db(indexed_db_path: Path, tmp_path: Path) -> Path:
    """Copy the session-scoped indexed DB into a per-test path so we can
    safely mutate its mtime without poisoning sibling tests."""
    dst = tmp_path / "knowledge-graph.sqlite"
    shutil.copyfile(indexed_db_path, dst)
    return dst


def test_server_reloads_on_mtime_change(
    isolated_db: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Touch the DB so its mtime advances, then call ``_reload_if_stale``.

    Expectations:
    - The ``Database mtime changed`` INFO line is emitted.
    - ``GRAPH`` ends up populated from the (re-loaded) DB.
    - ``_LOADED_MTIME`` advances to the new value.
    """
    with patch.dict(os.environ, {"GRAPH_DB_PATH": str(isolated_db)}):
        # Initial load. We assert the load picked up nodes so the test
        # has a real signal to compare against post-reload, instead of
        # validating "still empty -> still empty".
        server.load_graph(str(isolated_db))
        initial_node_count = server.GRAPH.number_of_nodes()
        assert initial_node_count > 0, (
            "indexed Calculator fixture should have nodes; got empty graph"
        )
        initial_mtime = server._LOADED_MTIME

        # Advance mtime by 2s. ``os.utime(None)`` to current time would
        # work too but is racy on filesystems with second-resolution
        # mtimes — pinning an explicit future timestamp avoids the race.
        future = initial_mtime + 2.0
        os.utime(isolated_db, (future, future))

        with caplog.at_level(logging.INFO, logger=server.log.name):
            server._reload_if_stale()

    # The reload-triggered log line is the contract surface.
    matching = [rec for rec in caplog.records if "Database mtime changed" in rec.getMessage()]
    assert matching, (
        f"expected 'Database mtime changed' INFO line, got "
        f"{[r.getMessage() for r in caplog.records]}"
    )

    # Mtime tracker advanced to the new value.
    assert server._LOADED_MTIME == pytest.approx(future, abs=1e-3), (
        f"_LOADED_MTIME did not advance; expected ~{future}, got {server._LOADED_MTIME}"
    )

    # Graph is populated from the same DB (content is unchanged) — same
    # node count is the simplest invariant.
    assert server.GRAPH.number_of_nodes() == initial_node_count


def test_server_no_reload_when_mtime_unchanged(
    isolated_db: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Repeated ``_reload_if_stale`` calls with no mtime change must be
    silent no-ops: no reload log line, ``GRAPH`` object identity stable.
    """
    with patch.dict(os.environ, {"GRAPH_DB_PATH": str(isolated_db)}):
        server.load_graph(str(isolated_db))
        assert server.GRAPH.number_of_nodes() > 0
        graph_id_before = id(server.GRAPH)
        mtime_before = server._LOADED_MTIME

        with caplog.at_level(logging.INFO, logger=server.log.name):
            server._reload_if_stale()
            server._reload_if_stale()
            server._reload_if_stale()

    # No reload log line.
    reload_messages = [
        rec for rec in caplog.records if "Database mtime changed" in rec.getMessage()
    ]
    assert not reload_messages, (
        f"unexpected reload log on stable mtime: {[r.getMessage() for r in reload_messages]}"
    )

    # ``GRAPH`` is the same object — ``load_graph`` was not invoked. We
    # use ``GRAPH.clear()`` + add inside ``load_graph``, so the container
    # identity stays stable but the contents would be replaced — guarding
    # against that by checking ``_LOADED_MTIME`` parity.
    assert id(server.GRAPH) == graph_id_before
    assert server._LOADED_MTIME == mtime_before


def test_reload_if_stale_tolerates_missing_db(
    isolated_db: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """If ``GRAPH_DB_PATH`` points at a path that no longer exists, the
    helper must not raise — it returns silently and lets the next real
    DB access surface the error with full context."""
    with patch.dict(os.environ, {"GRAPH_DB_PATH": str(isolated_db)}):
        server.load_graph(str(isolated_db))
        baseline_mtime = server._LOADED_MTIME

    # Point at a non-existent path. ``_reload_if_stale`` reads
    # ``GRAPH_DB_PATH`` at call time, so the env override here matters.
    missing = isolated_db.parent / "does-not-exist.sqlite"
    with patch.dict(os.environ, {"GRAPH_DB_PATH": str(missing)}):
        with caplog.at_level(logging.INFO, logger=server.log.name):
            server._reload_if_stale()  # must not raise

    # Tracker should NOT have been clobbered to 0.0 by the OSError path.
    assert server._LOADED_MTIME == baseline_mtime
