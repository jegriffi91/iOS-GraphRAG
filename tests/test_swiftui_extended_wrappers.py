"""SwiftUI 5+ wrapper coverage extension.

Phase 4.5d's initial ``_STATE_KIND_BY_ATTRIBUTE`` map covered 11
SwiftUI-1.0 / Combine wrappers. Real-world iOS codebases targeting
current OS revs depend on six newer wrappers introduced in iOS 15-17:

* ``@FocusState``                — focus binding (iOS 15+)
* ``@Bindable``                  — paired with ``@Observable`` (iOS 17+)
* ``@GestureState``              — auto-resetting gesture state
* ``@ScaledMetric``              — Dynamic Type scaling
* ``@Namespace``                 — matched geometry effects
* ``@AccessibilityFocusState``   — accessibility focus

Each test queries the indexed test DB for a row produced by the
``AdvancedView`` struct appended to
``test_fixtures/CalculatorApp/Sources/CalculatorApp/SwiftUIPatterns.swift``
and asserts the correct ``state_kind`` value.

The tests live in this dedicated file (rather than extending
``test_phase45d_swiftui.py``) so the git history cleanly shows the
SwiftUI 5+ coverage added in one place.
"""

from __future__ import annotations

import sqlite3


def _query(
    conn: sqlite3.Connection,
    *,
    symbol_name: str,
    symbol_type: str | None = None,
    file_substring: str = "SwiftUIPatterns.swift",
):
    """Return the single matching row, or None.

    Mirrors ``test_phase45d_swiftui._query`` so the two suites stay
    consistent. The fixture's symbol names are unique within
    SwiftUIPatterns.swift, so (name, optional type, file substring)
    yields at most one row.
    """
    sql = "SELECT * FROM nodes WHERE symbol_name = ? AND file_path LIKE ?"
    params: list = [symbol_name, f"%{file_substring}%"]
    if symbol_type is not None:
        sql += " AND symbol_type = ?"
        params.append(symbol_type)
    rows = conn.execute(sql, params).fetchall()
    assert len(rows) <= 1, (
        f"expected at most 1 row for {symbol_name!r}/{symbol_type!r}, "
        f"got {len(rows)}: {[dict(r) for r in rows]}"
    )
    return rows[0] if rows else None


def test_extracts_focusstate(db_connection: sqlite3.Connection) -> None:
    """``@FocusState private var focus`` -> ``state_kind='focusstate'``."""
    row = _query(db_connection, symbol_name="focus", symbol_type="property")
    assert row is not None, "focus not indexed as 'property'"
    assert row["state_kind"] == "focusstate", (
        f"focus should have state_kind='focusstate', got {row['state_kind']!r}"
    )


def test_extracts_bindable(db_connection: sqlite3.Connection) -> None:
    """``@Bindable var model: FocusModel`` -> ``state_kind='bindable'``.

    @Bindable (iOS 17+) wraps an @Observable instance for two-way
    binding. Tree-sitter parses it identically to other simple
    attribute wrappers (``@Bindable`` <-> ``@State``); no special AST
    handling is needed.
    """
    row = _query(db_connection, symbol_name="model", symbol_type="property")
    assert row is not None, "model not indexed as 'property'"
    assert row["state_kind"] == "bindable", (
        f"model should have state_kind='bindable', got {row['state_kind']!r}"
    )


def test_extracts_gesturestate(db_connection: sqlite3.Connection) -> None:
    """``@GestureState private var dragOffset`` -> ``state_kind='gesturestate'``."""
    row = _query(db_connection, symbol_name="dragOffset", symbol_type="property")
    assert row is not None, "dragOffset not indexed as 'property'"
    assert row["state_kind"] == "gesturestate", (
        f"dragOffset should have state_kind='gesturestate', got {row['state_kind']!r}"
    )


def test_extracts_scaledmetric(db_connection: sqlite3.Connection) -> None:
    """``@ScaledMetric private var iconSize`` -> ``state_kind='scaledmetric'``."""
    row = _query(db_connection, symbol_name="iconSize", symbol_type="property")
    assert row is not None, "iconSize not indexed as 'property'"
    assert row["state_kind"] == "scaledmetric", (
        f"iconSize should have state_kind='scaledmetric', got {row['state_kind']!r}"
    )


def test_extracts_namespace(db_connection: sqlite3.Connection) -> None:
    """``@Namespace private var heroNamespace`` -> ``state_kind='namespace'``."""
    row = _query(db_connection, symbol_name="heroNamespace", symbol_type="property")
    assert row is not None, "heroNamespace not indexed as 'property'"
    assert row["state_kind"] == "namespace", (
        f"heroNamespace should have state_kind='namespace', got {row['state_kind']!r}"
    )


def test_extracts_accessibilityfocusstate(db_connection: sqlite3.Connection) -> None:
    """``@AccessibilityFocusState private var a11yFocus`` -> ``state_kind='accessibilityfocusstate'``."""
    row = _query(db_connection, symbol_name="a11yFocus", symbol_type="property")
    assert row is not None, "a11yFocus not indexed as 'property'"
    assert row["state_kind"] == "accessibilityfocusstate", (
        f"a11yFocus should have state_kind='accessibilityfocusstate', "
        f"got {row['state_kind']!r}"
    )


def test_advanced_view_is_swiftui_view(db_connection: sqlite3.Connection) -> None:
    """``struct AdvancedView: View`` should have ``is_swiftui_view=1``.

    Confirms View detection still fires on the new struct alongside
    LoginView and ProfileBadge from the original Phase 4.5d fixture.
    """
    row = _query(db_connection, symbol_name="AdvancedView", symbol_type="struct")
    assert row is not None, "AdvancedView not indexed"
    assert row["is_swiftui_view"] == 1, (
        f"AdvancedView should have is_swiftui_view=1, got {row['is_swiftui_view']!r}"
    )


def test_focusmodel_is_observable(db_connection: sqlite3.Connection) -> None:
    """``@Observable class FocusModel`` should have ``is_observable=1``.

    Class-level @Observable still fires alongside the existing
    AuthState case. Guards against the extension pack accidentally
    breaking class-level macro detection.
    """
    row = _query(db_connection, symbol_name="FocusModel", symbol_type="class")
    assert row is not None, "FocusModel not indexed"
    assert row["is_observable"] == 1, (
        f"FocusModel should have is_observable=1, got {row['is_observable']!r}"
    )


def test_find_swiftui_views_filter_by_focusstate(
    indexed_db_path, monkeypatch
) -> None:
    """``_find_swiftui_views(state_kind='focusstate')`` returns only AdvancedView.

    LoginView and ProfileBadge from the original fixture do NOT use
    @FocusState; the filter must isolate the new wrapper to the new
    struct. Exercises the GROUP_CONCAT/HAVING substring match against
    one of the newly-added state kinds. Mirrors the calling pattern in
    ``test_find_swiftui_views.py`` (direct module call after pointing
    GRAPH_DB_PATH at the fixture DB).
    """
    from ios_graphrag import server

    monkeypatch.setenv("GRAPH_DB_PATH", str(indexed_db_path))
    server.load_graph(str(indexed_db_path))

    result = server._find_swiftui_views(state_kind="focusstate")

    assert result["count"] == 1, (
        f"expected exactly 1 view with focusstate, got {result['count']}: {result['views']}"
    )
    names = {v["swift_class"] for v in result["views"]}
    assert names == {"AdvancedView"}, (
        f"expected AdvancedView to be the only focusstate-using view, got {names}"
    )
