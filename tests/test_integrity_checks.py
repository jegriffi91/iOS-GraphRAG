"""Tests for the Phase 6c integrity-check helpers and migration 003.

Coverage:

1. ``run_integrity_checks`` against a freshly-indexed clean DB returns
   sane values (orphans=0, missing=[], non-zero row counts).
2. Manually-injected orphan edges are detected.
3. A node whose ``file_path`` is deleted from disk shows up in
   ``missing_files``.
4. Migration 003 dedupes pre-existing duplicate edges (run against a
   pre-Phase-6c schema with three identical rows; only one survives).
5. After migration 003 has run, attempting to insert a duplicate edge
   raises ``sqlite3.IntegrityError`` with "UNIQUE constraint failed".
6. ``ios-graphrag-doctor --verify`` exits 1 with "orphan" in its output
   when the DB has orphan edges.
"""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from ios_graphrag import _integrity, _migrations, indexer


_PROJECT_ROOT = Path(__file__).parent.parent
_TEST_FIXTURES = _PROJECT_ROOT / "test_fixtures" / "CalculatorApp"


# Pre-Phase-6c (i.e. pre-003) schema. We deliberately include schema_version
# stamped at v2 so apply_migrations sees this as a v2 DB and runs only 003,
# matching the upgrade path real users will take.
_PRE_003_SCHEMA_SQL = """
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
    selector TEXT,
    is_swiftui_view BOOLEAN,
    is_observable BOOLEAN,
    state_kind TEXT,
    body_kind TEXT
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

CREATE TABLE schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO schema_version (version) VALUES (1);
INSERT INTO schema_version (version) VALUES (2);
"""


@pytest.fixture
def fresh_indexed_db(tmp_path: Path) -> Path:
    """Index the CalculatorApp fixture into a new tmp DB.

    Module-scoped would be cheaper, but we want each test that relies
    on this fixture to start from a clean slate so mutations in one
    test (e.g. deleting a source file from disk) don't leak.
    """
    db_path = tmp_path / "integrity-test.sqlite"
    # Copy the fixture into tmp_path so deletion tests don't touch the
    # repo's checked-in fixture files.
    import shutil
    fixture_copy = tmp_path / "CalculatorApp"
    shutil.copytree(_TEST_FIXTURES, fixture_copy, symlinks=False)

    indexer.index_repository(
        repo_root=str(fixture_copy),
        db_path=str(db_path),
        full_reindex=True,
    )
    return db_path


def test_run_integrity_on_clean_db(fresh_indexed_db: Path) -> None:
    """A freshly-indexed DB has no orphan edges or missing files."""
    conn = sqlite3.connect(str(fresh_indexed_db))
    try:
        result = _integrity.run_integrity_checks(conn)
    finally:
        conn.close()

    counts = result["row_counts"]
    assert counts["nodes"] > 0, "indexer should produce some nodes"
    assert counts["edges"] > 0, "indexer should produce some edges"
    assert counts["file_hashes"] > 0, "file_hashes should be populated"

    assert result["orphan_edges"] == 0, (
        f"clean DB has unexpected orphans: {result}"
    )
    assert result["missing_files"] == [], (
        f"clean DB reports missing files: {result['missing_files']}"
    )


def test_orphan_edges_detected(fresh_indexed_db: Path) -> None:
    """Manually inserting an edge with a bogus target_node_id is flagged."""
    conn = sqlite3.connect(str(fresh_indexed_db))
    try:
        # Pick any existing node id as the source so the FK is
        # satisfied (target FK is permissive in SQLite without
        # PRAGMA foreign_keys=ON).
        row = conn.execute("SELECT id FROM nodes LIMIT 1").fetchone()
        assert row is not None
        source_id = row[0]
        # 999_999 is far above any AUTOINCREMENT id the indexer would
        # mint for the small test fixture.
        conn.execute(
            "INSERT INTO edges "
            "(source_node_id, target_node_id, target_symbol, edge_type, line_number) "
            "VALUES (?, ?, ?, ?, ?)",
            (source_id, 999_999, "BogusTarget", "CALLS", 42),
        )
        conn.commit()

        result = _integrity.run_integrity_checks(conn)
    finally:
        conn.close()

    assert result["orphan_edges"] >= 1, (
        f"expected orphan_edges>=1, got {result}"
    )


def test_missing_file_paths_detected(fresh_indexed_db: Path, tmp_path: Path) -> None:
    """Deleting a source file from disk surfaces in missing_files."""
    conn = sqlite3.connect(str(fresh_indexed_db))
    try:
        rows = conn.execute(
            "SELECT DISTINCT file_path FROM nodes ORDER BY file_path LIMIT 1"
        ).fetchall()
    finally:
        conn.close()
    assert rows, "fixture should produce at least one node with a file_path"
    target_path = rows[0][0]

    # Sanity: the file should exist before we delete it.
    assert os.path.exists(target_path), (
        f"file_path in DB not on disk yet: {target_path}"
    )

    os.unlink(target_path)

    conn = sqlite3.connect(str(fresh_indexed_db))
    try:
        result = _integrity.run_integrity_checks(conn)
    finally:
        conn.close()

    assert target_path in result["missing_files"], (
        f"deleted path {target_path} not surfaced in {result['missing_files']}"
    )


def test_unique_edges_migration_dedupes_existing(tmp_path: Path) -> None:
    """003 deletes duplicate edges (keeps min rowid) before the index lands.

    Builds a DB stamped at v2, inserts three identical edge tuples
    (target_node_id is set so the unique-key columns are fully
    populated), runs ``apply_migrations`` to trigger 003, and asserts
    the count drops to 1.
    """
    db_path = tmp_path / "pre003.sqlite"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(_PRE_003_SCHEMA_SQL)
        # Need a node so source_node_id is real (FK is unenforced
        # without PRAGMA foreign_keys=ON, but this matches reality).
        conn.execute(
            "INSERT INTO nodes "
            "(file_path, symbol_name, symbol_type, language, start_byte, "
            "end_byte, line_number, signature) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("/tmp/Foo.swift", "Foo", "class", "swift", 0, 10, 1, "class Foo"),
        )
        node_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        # Insert THREE identical edges (everything that participates in
        # the new unique key matches: source, target_symbol, edge_type,
        # line_number).
        for _ in range(3):
            conn.execute(
                "INSERT INTO edges "
                "(source_node_id, target_node_id, target_symbol, edge_type, line_number) "
                "VALUES (?, ?, ?, ?, ?)",
                (node_id, None, "Bar", "CALLS", 5),
            )
        conn.commit()

        # Sanity: we really inserted three before the migration runs.
        before = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        assert before == 3

        new_version = _migrations.apply_migrations(conn)
        assert new_version >= 3, "migration 003 should have applied"

        after = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        assert after == 1, (
            f"expected dedup to leave 1 row, got {after}"
        )
    finally:
        conn.close()


def test_unique_edges_blocks_future_duplicates(tmp_path: Path) -> None:
    """Once 003 is applied, duplicate inserts raise IntegrityError.

    Builds a fresh DB, runs all migrations (lands at v3), inserts one
    edge, and asserts the same tuple a second time raises
    ``sqlite3.IntegrityError`` with the SQLite "UNIQUE constraint
    failed" prefix.
    """
    db_path = tmp_path / "v3.sqlite"
    conn = sqlite3.connect(str(db_path))
    try:
        new_version = _migrations.apply_migrations(conn)
        assert new_version >= 3, "expected schema to reach v3"

        # Build a real node for the FK + unique-key source side.
        conn.execute(
            "INSERT INTO nodes "
            "(file_path, symbol_name, symbol_type, language, start_byte, "
            "end_byte, line_number, signature) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("/tmp/Bar.swift", "Bar", "class", "swift", 0, 10, 1, "class Bar"),
        )
        node_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        conn.execute(
            "INSERT INTO edges "
            "(source_node_id, target_node_id, target_symbol, edge_type, line_number) "
            "VALUES (?, ?, ?, ?, ?)",
            (node_id, None, "Quux", "CALLS", 7),
        )

        with pytest.raises(sqlite3.IntegrityError) as exc_info:
            conn.execute(
                "INSERT INTO edges "
                "(source_node_id, target_node_id, target_symbol, edge_type, line_number) "
                "VALUES (?, ?, ?, ?, ?)",
                (node_id, None, "Quux", "CALLS", 7),
            )
        assert "UNIQUE constraint failed" in str(exc_info.value), (
            f"expected UNIQUE constraint failure, got {exc_info.value!r}"
        )
    finally:
        conn.close()


def test_doctor_verify_exits_1_on_orphan_edges(fresh_indexed_db: Path) -> None:
    """Subprocess ``ios-graphrag-doctor --verify`` against an orphan-laden DB."""
    conn = sqlite3.connect(str(fresh_indexed_db))
    try:
        row = conn.execute("SELECT id FROM nodes LIMIT 1").fetchone()
        assert row is not None
        source_id = row[0]
        conn.execute(
            "INSERT INTO edges "
            "(source_node_id, target_node_id, target_symbol, edge_type, line_number) "
            "VALUES (?, ?, ?, ?, ?)",
            (source_id, 999_999, "BogusTarget", "CALLS", 42),
        )
        conn.commit()
    finally:
        conn.close()

    env = os.environ.copy()
    env["GRAPH_DB_PATH"] = str(fresh_indexed_db)
    result = subprocess.run(
        ["uv", "run", "ios-graphrag-doctor", "--verify"],
        cwd=str(_PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 1, (
        f"expected rc=1 on orphan edges, got rc={result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "orphan" in combined.lower(), (
        f"expected 'orphan' in output, got:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
