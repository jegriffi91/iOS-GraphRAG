"""Database schema migration runner for iOS-GraphRAG (Phase 6a).

Migrations live in ``engine/database/migrations/<NNN>_<description>.sql``.
``apply_migrations()`` reads the current ``schema_version`` (treating
absence of the table as version 0), applies any unapplied migrations in
numeric order inside a transaction, and refuses to run if the DB is at a
NEWER version than the code knows about.

Design notes:

- The runner is the single entry point for schema changes. The indexer
  calls it on every run; the server only reads ``current_version`` and
  refuses to start on mismatch (the server never writes schema changes,
  to avoid a race with a concurrent indexer process).
- Migration files are immutable once shipped. Fix a mistake by adding
  a new migration, never by editing an existing one.
- Each migration body must end with
  ``INSERT OR REPLACE INTO schema_version (version) VALUES (N);`` so the
  version stamp commits atomically with the rest of the migration body
  inside the same transaction.
- ``001_baseline.sql`` uses ``CREATE TABLE IF NOT EXISTS`` so it is a
  safe no-op on legacy DBs (which already have the tables but no
  ``schema_version`` row).
- ``ALTER TABLE ADD COLUMN`` errors on duplicate columns. The runner
  catches ``sqlite3.OperationalError`` whose message contains
  "duplicate column" and treats the migration as already applied -- this
  recovers gracefully from a previously-partial migration.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path
from typing import List, Tuple

log = logging.getLogger(__name__)


class SchemaMismatchError(RuntimeError):
    """Raised when DB ``schema_version`` exceeds the highest known migration."""


# Repo layout: <repo>/src/ios_graphrag/_migrations.py and
# <repo>/engine/database/migrations/. parents[2] of this file is the repo root.
MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "engine" / "database" / "migrations"
MIGRATION_FILENAME_RE = re.compile(r"^(\d{3})_[\w_]+\.sql$")


def discover_migrations() -> List[Tuple[int, Path]]:
    """Return ``[(version, path)]`` for every migration file, sorted by version.

    Returns an empty list if the migrations directory is absent or empty.
    Filenames must match ``^\\d{3}_[\\w_]+\\.sql$`` -- anything else (a
    README, a temporary editor file) is ignored.
    """
    items: List[Tuple[int, Path]] = []
    if not MIGRATIONS_DIR.is_dir():
        return items
    for path in MIGRATIONS_DIR.iterdir():
        match = MIGRATION_FILENAME_RE.match(path.name)
        if match:
            items.append((int(match.group(1)), path))
    return sorted(items, key=lambda t: t[0])


def current_version(conn: sqlite3.Connection) -> int:
    """Return the DB's current schema version. Zero if absent.

    "Absent" means either (a) ``schema_version`` table does not exist, or
    (b) the table exists but is empty. Both legacy DBs and freshly-created
    DBs without a baseline applied resolve to 0.
    """
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
    ).fetchone()
    if row is None:
        return 0
    cur = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    return cur[0] or 0


def _is_duplicate_column_error(exc: sqlite3.OperationalError) -> bool:
    """True if ``exc`` is sqlite's ``duplicate column name: <name>`` error.

    SQLite raises ``OperationalError`` with a message like
    ``duplicate column name: is_swiftui_view`` when ``ALTER TABLE ADD
    COLUMN`` fires on a column that already exists. We treat that as
    "the column we wanted is already present" and let the migration
    proceed past it -- this matters for partial-failure recovery (e.g.
    a previous migration crashed between the ALTER and the version
    stamp).
    """
    return "duplicate column" in str(exc).lower()


def _split_sql_statements(sql: str) -> List[str]:
    """Split a migration SQL file into individual statements.

    Strategy: drop whole-line ``--`` comments first, then split on
    ``;``. Whole-line stripping is required because comment prose
    legitimately contains semicolons (``"These columns are nullable;
    the current indexer..."``) which would otherwise terminate a
    pseudo-statement. After comment removal the migration files only
    contain DDL with no string literals or trigger bodies, so naive
    semicolon splitting is sound.

    We cannot use ``executescript`` because it issues an implicit
    COMMIT, which would break the atomic version-stamp guarantee.
    """
    sql_only = "\n".join(line for line in sql.splitlines() if not line.strip().startswith("--"))
    statements: List[str] = []
    for raw in sql_only.split(";"):
        body = raw.strip()
        if body:
            statements.append(body)
    return statements


def _apply_one(conn: sqlite3.Connection, version: int, path: Path) -> None:
    """Apply a single migration file inside a transaction.

    SQLite's ``executescript`` issues an implicit COMMIT, which would
    break the atomic-version-stamp guarantee, so we manage the
    transaction ourselves with explicit BEGIN/COMMIT/ROLLBACK and
    execute each statement separately. Duplicate-column
    ``OperationalError`` instances are caught and treated as
    already-applied so partial-failure recovery is automatic.
    """
    statements = _split_sql_statements(path.read_text(encoding="utf-8"))
    conn.execute("BEGIN")
    try:
        for stmt in statements:
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError as exc:
                if _is_duplicate_column_error(exc):
                    log.info(
                        "migration %03d: skipping already-applied statement (%s)",
                        version,
                        exc,
                    )
                    continue
                raise
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def apply_migrations(conn: sqlite3.Connection) -> int:
    """Apply unapplied migrations in numeric order. Return the new version.

    - If no migration files exist, return the current version unchanged.
    - If the DB is at a higher version than the highest known migration,
      raise :class:`SchemaMismatchError`.
    - Otherwise, run each migration whose version exceeds the current
      version. Each migration body runs inside its own transaction so
      the ``INSERT INTO schema_version`` lands atomically with the DDL.
    """
    migrations = discover_migrations()
    if not migrations:
        return current_version(conn)

    max_known = max(version for version, _ in migrations)
    cur = current_version(conn)

    if cur > max_known:
        raise SchemaMismatchError(
            f"Database at v{cur}, code knows up to v{max_known}. "
            "Upgrade ios-graphrag or rebuild the index."
        )

    for version, path in migrations:
        if version > cur:
            log.info("applying migration %03d (%s)", version, path.name)
            _apply_one(conn, version, path)

    return current_version(conn)


def max_known_version() -> int:
    """Return the highest migration version known to this code.

    Convenience for the server's startup check, which needs to compare
    the DB's version against what this build supports without running
    any migrations.
    """
    migrations = discover_migrations()
    return max((version for version, _ in migrations), default=0)
