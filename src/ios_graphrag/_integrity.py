"""Database integrity checks (Phase 6c).

Each check is fail-soft: a failure inside one check is logged but never
aborts the rest of the run. The indexer calls :func:`run_integrity_checks`
at the end of every indexing run for visibility; ``ios-graphrag-doctor
--verify`` runs the same checks on demand for ad-hoc verification.

Returns a structured dict so the doctor CLI can render its own report
format without reparsing log strings.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from typing import Dict, List, Tuple

log = logging.getLogger(__name__)


def _safe(check_name: str, fn):
    """Run ``fn`` and convert any exception into a logged warning + None.

    Integrity checks are diagnostic; a failure inside one check (e.g. a
    missing table on a corrupt DB) must never abort the indexing run or
    other checks running alongside it. We log the exception so the user
    sees what went wrong without losing the rest of the report.
    """
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "integrity check %r failed: %s: %s",
            check_name,
            type(exc).__name__,
            exc,
        )
        return None


def _row_counts(conn: sqlite3.Connection) -> Dict[str, int]:
    """Return ``{table: count}`` for the three tables we care about."""
    counts: Dict[str, int] = {}
    for table in ("nodes", "edges", "file_hashes"):
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        counts[table] = int(row[0]) if row and row[0] is not None else 0
    return counts


def _orphan_edge_count(conn: sqlite3.Connection) -> Tuple[int, int]:
    """Return ``(orphan_count, resolved_count)``.

    An "orphan" is an edge whose ``target_node_id`` is non-NULL but does
    not appear in ``nodes.id``. Edges with ``target_node_id`` IS NULL are
    NOT orphans -- those are deliberately-unresolved edges (the indexer
    leaves cross-module references unresolved by design when no matching
    declaration is found).
    """
    orphan_row = conn.execute(
        """
        SELECT COUNT(*) FROM edges e
        WHERE e.target_node_id IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM nodes n WHERE n.id = e.target_node_id)
        """
    ).fetchone()
    resolved_row = conn.execute(
        "SELECT COUNT(*) FROM edges WHERE target_node_id IS NOT NULL"
    ).fetchone()
    return (
        int(orphan_row[0]) if orphan_row and orphan_row[0] is not None else 0,
        int(resolved_row[0]) if resolved_row and resolved_row[0] is not None else 0,
    )


def _missing_file_paths(conn: sqlite3.Connection) -> List[str]:
    """Return distinct ``file_path`` values that no longer exist on disk.

    Sorted for determinism so the same DB always reports paths in the
    same order, which keeps the doctor's report stable across runs.
    """
    rows = conn.execute("SELECT DISTINCT file_path FROM nodes ORDER BY file_path").fetchall()
    missing: List[str] = []
    for (path,) in rows:
        if path and not os.path.exists(path):
            missing.append(path)
    return missing


def _symbol_collisions(conn: sqlite3.Connection) -> int:
    """Count distinct ``symbol_name`` values that appear in >1 node.

    The P1.3 resolver in ``indexer.py`` already handles these
    deterministically; this is purely visibility -- if the count is
    high it usually means cross-module name reuse (common in Swift
    extension-heavy codebases) which the user may want to know about.
    """
    row = conn.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT symbol_name FROM nodes
            GROUP BY symbol_name HAVING COUNT(*) > 1
        )
        """
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def run_integrity_checks(conn: sqlite3.Connection) -> Dict[str, object]:
    """Run all integrity checks and return a structured result dict.

    Shape::

        {
            "row_counts": {"nodes": int, "edges": int, "file_hashes": int},
            "orphan_edges": int,
            "resolved_edges": int,
            "missing_files": list[str],
            "symbol_collisions": int,
        }

    Any sub-check that errors logs a warning and contributes a sentinel
    value to the dict (``0`` for counts, ``[]`` for lists) so callers
    can render a partial report without crashing on missing keys.
    """
    counts = _safe("row_counts", lambda: _row_counts(conn)) or {}
    orphan_pair = _safe("orphan_edges", lambda: _orphan_edge_count(conn)) or (0, 0)
    orphan_edges, resolved_edges = orphan_pair
    missing = _safe("missing_files", lambda: _missing_file_paths(conn)) or []
    collisions = _safe("symbol_collisions", lambda: _symbol_collisions(conn)) or 0

    return {
        "row_counts": counts,
        "orphan_edges": orphan_edges,
        "resolved_edges": resolved_edges,
        "missing_files": missing,
        "symbol_collisions": collisions,
    }


def log_integrity_check(conn: sqlite3.Connection) -> Dict[str, object]:
    """Run integrity checks and emit log lines via the configured logger.

    Returns the same dict :func:`run_integrity_checks` does so callers
    can re-use the result without re-running. Logging levels:

    - Row counts and symbol collisions: INFO (always emitted; routine).
    - Orphan edges: WARNING when count > 0 (real bug -- the resolver
      pointed at a node id that no longer exists).
    - Missing file paths: WARNING when any are present (the source file
      was deleted/renamed since the last index, the user should run a
      ``--full`` reindex).

    Fail-soft: this function never raises. If something goes wrong
    inside a single check, ``run_integrity_checks`` already swallowed
    the exception with a logged warning; this wrapper just emits the
    summary lines on top.
    """
    result = run_integrity_checks(conn)

    counts = result.get("row_counts", {}) or {}
    log.info(
        "integrity: row counts -- nodes=%d edges=%d file_hashes=%d",
        counts.get("nodes", 0),
        counts.get("edges", 0),
        counts.get("file_hashes", 0),
    )

    orphan_edges = int(result.get("orphan_edges", 0) or 0)
    resolved_edges = int(result.get("resolved_edges", 0) or 0)
    if orphan_edges > 0:
        log.warning(
            "integrity: %d orphan edges detected (target_node_id points "
            "at a non-existent node); %d edges resolved cleanly",
            orphan_edges,
            resolved_edges,
        )

    missing = list(result.get("missing_files") or [])
    if missing:
        # Show the first 5 so the log line stays bounded; the full list
        # is in the dict for callers that want all of them.
        sample = ", ".join(missing[:5])
        log.warning(
            "integrity: %d node file_path(s) missing on disk; sample: %s",
            len(missing),
            sample,
        )

    collisions = int(result.get("symbol_collisions", 0) or 0)
    log.info(
        "integrity: %d symbol name(s) resolve to multiple nodes "
        "(P1.3 resolver picks deterministically; collisions logged separately)",
        collisions,
    )

    return result
