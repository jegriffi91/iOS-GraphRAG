"""Phase 4.5d -- SwiftUI symbol enrichment.

Each test queries the indexed test DB for a row produced by
``test_fixtures/CalculatorApp/Sources/CalculatorApp/SwiftUIPatterns.swift``
and asserts the expected value lands in one of the four migration-002
columns: ``is_swiftui_view``, ``is_observable``, ``state_kind``,
``body_kind``.

Coverage map:

  * ``is_swiftui_view``: tested on LoginView + ProfileBadge (View
    conformance).
  * ``is_observable``: tested on AuthState (@Observable). Counter is
    explicitly NOT marked observable -- @Published is property-level,
    not class-level.
  * ``state_kind``: one test per supported wrapper -- @State, @Binding,
    @StateObject, @ObservedObject, @Environment, @AppStorage, @Published.
  * ``body_kind``: one test for ``var body: some View`` (viewbody) and
    one for ``@ViewBuilder func`` (resultbuilder).
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

    The SwiftUIPatterns fixture is unique enough that
    (symbol_name, optional symbol_type, file substring) yields exactly
    one row. ``symbol_type=None`` lets tests query rows where the type
    is unimportant (e.g. is_observable on a class).
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


def test_extracts_view_struct(db_connection: sqlite3.Connection) -> None:
    """Structs that conform to ``View`` get ``is_swiftui_view=1``.

    Both LoginView and ProfileBadge in the fixture declare ``: View``;
    both should be flagged. Counter, AuthState, and Recipe (the non-View
    types in the fixture) must remain NULL/0 -- this guards against the
    detector over-firing on every class/struct.
    """
    for name in ("LoginView", "ProfileBadge"):
        row = _query(db_connection, symbol_name=name, symbol_type="struct")
        assert row is not None, f"{name} not indexed"
        assert row["is_swiftui_view"] == 1, (
            f"{name} should have is_swiftui_view=1, got {row['is_swiftui_view']!r}"
        )

    # AuthState is @Observable but NOT a View. Don't false-positive.
    auth = _query(db_connection, symbol_name="AuthState", symbol_type="class")
    assert auth is not None
    assert not auth["is_swiftui_view"], (
        f"AuthState (a class, not a View) wrongly flagged is_swiftui_view={auth['is_swiftui_view']!r}"
    )


def test_extracts_observable_class(db_connection: sqlite3.Connection) -> None:
    """``@Observable`` flips ``is_observable=1`` on the class.

    Policy: ``is_observable`` reflects ONLY class-level @Observable /
    @Model attributes. Classes with @Published members but no class-level
    macro must remain unmarked -- @Published is captured separately on
    the property via ``state_kind``. Counter has @Published var count
    but no @Observable, so its ``is_observable`` must be NULL/0.
    """
    auth = _query(db_connection, symbol_name="AuthState", symbol_type="class")
    assert auth is not None
    assert auth["is_observable"] == 1, (
        f"AuthState (@Observable class) should have is_observable=1, got {auth['is_observable']!r}"
    )

    # Counter has @Published members but no class-level @Observable / @Model.
    counter = _query(db_connection, symbol_name="Counter", symbol_type="class")
    assert counter is not None
    assert not counter["is_observable"], (
        f"Counter (only @Published, no @Observable) should NOT have is_observable=1, "
        f"got {counter['is_observable']!r}"
    )


def test_extracts_state_kind_state(db_connection: sqlite3.Connection) -> None:
    """``@State private var password`` -> ``state_kind='state'``."""
    row = _query(db_connection, symbol_name="password", symbol_type="property")
    assert row is not None, "password not indexed as 'property'"
    assert row["state_kind"] == "state", (
        f"password should have state_kind='state', got {row['state_kind']!r}"
    )


def test_extracts_state_kind_stateobject(db_connection: sqlite3.Connection) -> None:
    """``@StateObject var auth`` -> ``state_kind='stateobject'``."""
    row = _query(db_connection, symbol_name="auth", symbol_type="property")
    assert row is not None, "auth not indexed as 'property'"
    assert row["state_kind"] == "stateobject", (
        f"auth should have state_kind='stateobject', got {row['state_kind']!r}"
    )


def test_extracts_state_kind_binding(db_connection: sqlite3.Connection) -> None:
    """``@Binding var isVisible`` -> ``state_kind='binding'``."""
    row = _query(db_connection, symbol_name="isVisible", symbol_type="property")
    assert row is not None, "isVisible not indexed as 'property'"
    assert row["state_kind"] == "binding", (
        f"isVisible should have state_kind='binding', got {row['state_kind']!r}"
    )


def test_extracts_state_kind_environment(db_connection: sqlite3.Connection) -> None:
    """``@Environment(\\.colorScheme) var colorScheme`` -> ``state_kind='environment'``.

    The Environment wrapper carries a key-path argument
    (``\\.colorScheme``) and is more complex AST-wise than parameterless
    wrappers. This test guards against a regression where the argument
    list breaks the type_identifier walk.
    """
    row = _query(db_connection, symbol_name="colorScheme", symbol_type="property")
    assert row is not None, "colorScheme not indexed as 'property'"
    assert row["state_kind"] == "environment", (
        f"colorScheme should have state_kind='environment', got {row['state_kind']!r}"
    )


def test_extracts_state_kind_appstorage(db_connection: sqlite3.Connection) -> None:
    """``@AppStorage(\"rememberMe\") var remember`` -> ``state_kind='appstorage'``."""
    row = _query(db_connection, symbol_name="remember", symbol_type="property")
    assert row is not None, "remember not indexed as 'property'"
    assert row["state_kind"] == "appstorage", (
        f"remember should have state_kind='appstorage', got {row['state_kind']!r}"
    )


def test_extracts_state_kind_published(db_connection: sqlite3.Connection) -> None:
    """``@Published var count`` -> ``state_kind='published'``.

    Combine's @Published is recognized as a property-level wrapper even
    though it's not from SwiftUI proper. Documents the design choice
    that ``state_kind`` is broader than "SwiftUI state" -- it covers any
    property wrapper that participates in iOS reactivity.
    """
    row = _query(db_connection, symbol_name="count", symbol_type="property")
    assert row is not None, "count not indexed as 'property'"
    assert row["state_kind"] == "published", (
        f"count should have state_kind='published', got {row['state_kind']!r}"
    )


def test_extracts_body_kind_viewbody(db_connection: sqlite3.Connection) -> None:
    """Both Views' ``body: some View`` properties get ``body_kind='viewbody'``.

    The fixture has two Views; both declare ``var body: some View`` as
    a computed property. We expect TWO rows with symbol_name='body' and
    body_kind='viewbody'.
    """
    rows = db_connection.execute(
        "SELECT * FROM nodes WHERE symbol_name = 'body' "
        "AND symbol_type = 'computed_property' "
        "AND file_path LIKE '%SwiftUIPatterns.swift%' "
        "AND body_kind = 'viewbody'"
    ).fetchall()
    assert len(rows) == 2, (
        f"expected 2 'body' viewbody rows (LoginView + ProfileBadge), "
        f"got {len(rows)}: {[dict(r) for r in rows]}"
    )


def test_extracts_body_kind_resultbuilder(db_connection: sqlite3.Connection) -> None:
    """``@ViewBuilder func makeButtonRow`` -> ``body_kind='resultbuilder'``.

    The function-level @ViewBuilder annotation flags the function as a
    SwiftUI result-builder participant. body_kind is set on functions
    too, not just computed properties.
    """
    row = _query(
        db_connection, symbol_name="makeButtonRow", symbol_type="function"
    )
    assert row is not None, "makeButtonRow not indexed as 'function'"
    assert row["body_kind"] == "resultbuilder", (
        f"makeButtonRow should have body_kind='resultbuilder', got {row['body_kind']!r}"
    )
