"""Tests for the schema-version migration runner (Phase 6a).

Coverage:

1. Fresh empty DB -> migrations apply, version reaches the highest
   known migration, the SwiftUI columns are present.
2. Legacy DB (pre-Phase-6a tables, no schema_version row) -> the
   baseline migration is a no-op on existing tables, the version
   stamp lands, and the SwiftUI columns are added.
3. Newer DB (schema_version > what the code knows) -> the runner
   refuses with SchemaMismatchError instead of silently degrading.
4. Idempotency: running ``apply_migrations`` twice in a row leaves
   the DB unchanged on the second call.
5. Server: mismatched schema_version causes the binary to exit 1
   with a clear remediation message, before ``mcp.run()`` blocks.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
from pathlib import Path

import pytest

from ios_graphrag import _migrations

# Pre-Phase-6a schema. Used by test_apply_migrations_to_legacy_db to
# build a DB that simulates one created by an older ios-graphrag build:
# all tables exist but there is no schema_version table at all.
_LEGACY_SCHEMA_SQL = """
CREATE TABLE file_hashes (
    path TEXT PRIMARY KEY,
    hash TEXT NOT NULL
);

CREATE TABLE nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    symbol_name TEXT NOT NULL,
    symbol_type TEXT NOT NULL,
    language TEXT NOT NULL,
    start_byte INTEGER NOT NULL,
    end_byte INTEGER NOT NULL,
    line_number INTEGER,
    signature TEXT NOT NULL,
    selector TEXT
);

CREATE TABLE edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_node_id INTEGER NOT NULL,
    target_node_id INTEGER,
    target_symbol TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    line_number INTEGER,
    FOREIGN KEY (source_node_id) REFERENCES nodes(id),
    FOREIGN KEY (target_node_id) REFERENCES nodes(id)
);

CREATE TABLE extension_map (
    extension_node_id INTEGER PRIMARY KEY,
    canonical_node_id INTEGER,
    FOREIGN KEY (extension_node_id) REFERENCES nodes(id),
    FOREIGN KEY (canonical_node_id) REFERENCES nodes(id)
);

CREATE TABLE node_embeddings (
    node_id INTEGER PRIMARY KEY,
    embedding BLOB NOT NULL,
    FOREIGN KEY (node_id) REFERENCES nodes(id)
);
"""

# Columns the Phase 6a 002 migration must add. Asserted by tests 1 and 2.
SWIFTUI_COLUMNS = {"is_swiftui_view", "is_observable", "state_kind", "body_kind"}


def _node_columns(conn: sqlite3.Connection) -> set[str]:
    return {row[1] for row in conn.execute("PRAGMA table_info(nodes)")}


def test_apply_migrations_to_fresh_db():
    """Empty DB -> all migrations applied -> latest version with SwiftUI cols.

    The exact "latest" number is whatever ``max_known_version()`` reports;
    asserting against that constant keeps the test stable as new migrations
    land. Phase 6c added 003, so today the latest is 3, but the test
    intentionally avoids hard-coding that.
    """
    conn = sqlite3.connect(":memory:")
    try:
        new_version = _migrations.apply_migrations(conn)
        latest = _migrations.max_known_version()
        assert new_version == latest
        assert latest >= 2, "baseline + SwiftUI migrations must always exist"
        assert _migrations.current_version(conn) == latest

        cols = _node_columns(conn)
        assert SWIFTUI_COLUMNS <= cols, f"missing SwiftUI columns; got {sorted(cols)}"

        # Every migration must stamp its version into schema_version.
        rows = conn.execute("SELECT version FROM schema_version ORDER BY version").fetchall()
        assert [r[0] for r in rows] == list(range(1, latest + 1))
    finally:
        conn.close()


def test_apply_migrations_to_legacy_db():
    """Legacy DB (tables but no schema_version) -> latest + SwiftUI columns.

    Simulates upgrading an existing user's DB: all the pre-6a tables
    exist, there is no schema_version table, and no rows have been
    populated. The runner must promote it to the latest known version
    without erroring on duplicate CREATE TABLE statements
    (001_baseline uses IF NOT EXISTS) and must add the SwiftUI columns
    introduced in 002.
    """
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(_LEGACY_SCHEMA_SQL)
        # Sanity: legacy DB has nodes/edges but no schema_version.
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "nodes" in tables
        assert "edges" in tables
        assert "schema_version" not in tables
        assert _migrations.current_version(conn) == 0

        new_version = _migrations.apply_migrations(conn)
        assert new_version == _migrations.max_known_version()

        cols = _node_columns(conn)
        assert SWIFTUI_COLUMNS <= cols
    finally:
        conn.close()


def test_refuses_to_run_on_newer_db():
    """schema_version=99 with no matching migration -> SchemaMismatchError."""
    conn = sqlite3.connect(":memory:")
    try:
        # Hand-craft a schema_version row claiming a version far ahead
        # of any migration on disk. The runner must refuse rather than
        # silently downgrade or no-op.
        conn.executescript(
            "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, "
            "applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);"
            "INSERT INTO schema_version (version) VALUES (99);"
        )
        with pytest.raises(_migrations.SchemaMismatchError) as exc_info:
            _migrations.apply_migrations(conn)
        msg = str(exc_info.value)
        assert "v99" in msg
        assert "Upgrade" in msg or "rebuild" in msg
    finally:
        conn.close()


def test_idempotent_application():
    """Two consecutive apply_migrations calls leave the DB unchanged."""
    conn = sqlite3.connect(":memory:")
    try:
        latest = _migrations.max_known_version()
        v1 = _migrations.apply_migrations(conn)
        rows1 = conn.execute("SELECT version FROM schema_version ORDER BY version").fetchall()

        v2 = _migrations.apply_migrations(conn)
        rows2 = conn.execute("SELECT version FROM schema_version ORDER BY version").fetchall()

        assert v1 == v2 == latest
        assert rows1 == rows2, "second apply_migrations must not re-stamp version rows"
    finally:
        conn.close()


def test_server_refuses_to_start_on_mismatch(tmp_path: Path):
    """Older DB on newer code: server exits 1 with remediation message.

    Builds a DB at v1 (only 001_baseline applied), runs the server
    binary, and asserts it logs an error directing the user to rerun
    the indexer and exits 1 -- never reaches mcp.run(). Asserts the
    output names both v1 (the stale DB) and the latest known version
    so the user can see exactly what's expected.
    """
    db_path = tmp_path / "stale.sqlite"
    conn = sqlite3.connect(str(db_path))
    try:
        # Apply only 001 by hand-running its body, leaving us at v1
        # while the code knows about higher versions.
        baseline = _migrations.MIGRATIONS_DIR / "001_baseline.sql"
        _migrations._apply_one(conn, 1, baseline)
        assert _migrations.current_version(conn) == 1
    finally:
        conn.close()

    latest = _migrations.max_known_version()

    env = os.environ.copy()
    env["GRAPH_DB_PATH"] = str(db_path)
    # Stop the indexer's import-time HF cache poke from slowing the
    # test; not required, but keeps the subprocess fast.
    env.setdefault("HF_HUB_OFFLINE", "1")
    result = subprocess.run(
        ["uv", "run", "ios-graphrag-server"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 1, (
        f"server should exit 1 on schema mismatch, got rc={result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    combined = result.stderr + result.stdout
    assert "v1" in combined and f"v{latest}" in combined, (
        f"expected version mismatch (v1 vs v{latest}) in output, got:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "ios-graphrag-index" in combined or "Run" in combined, (
        f"expected remediation pointer in output, got:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
