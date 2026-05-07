"""Phase 5 P0 — properties + initializers + deinitializers.

Each test queries the indexed test database for a specific symbol_name +
file_path + symbol_type combination from the Phase 5 fixture
``PropertiesAndInit.swift``. The fixture intentionally packs every
detection path into one file so a single indexing run covers them all:

    - stored properties (``id``, ``balance``, ``total``)
    - simple-body computed (``displayName``, ``taxIncluded``)
    - explicit get/set computed (``formatted``)
    - lazy stored (``heavy``)
    - static stored (``sharedTaxRate``)
    - plain initializer (``init(id:balance:)``)
    - failable initializer (``init?(legacyString:)``)
    - deinitializer (``deinit``)

Tests use the session-scoped ``db_connection`` fixture so they piggyback
on the single indexing run that other test modules also share, keeping
the suite fast.
"""

from __future__ import annotations

import sqlite3


def _query_symbol(
    conn: sqlite3.Connection,
    *,
    symbol_name: str,
    symbol_type: str,
    file_substring: str = "PropertiesAndInit.swift",
):
    """Return the single matching row, or None.

    The PropertiesAndInit fixture is unique enough that
    (symbol_name, symbol_type, file_path-substring) yields exactly one
    row when the extractor fires. We guard against accidental dup rows
    by asserting len <= 1 in the helper.
    """
    rows = conn.execute(
        "SELECT * FROM nodes WHERE symbol_name = ? AND symbol_type = ? "
        "AND file_path LIKE ?",
        (symbol_name, symbol_type, f"%{file_substring}%"),
    ).fetchall()
    assert len(rows) <= 1, (
        f"expected at most 1 row for {symbol_name!r}/{symbol_type!r}, "
        f"got {len(rows)}: {[dict(r) for r in rows]}"
    )
    return rows[0] if rows else None


def test_extracts_stored_property(db_connection: sqlite3.Connection) -> None:
    """Stored ``let`` and ``var`` declarations land as ``symbol_type='property'``.

    Covers the simplest case: ``public let id: Int`` and
    ``public var balance: Double`` inside HouseholdAccount, plus
    ``public let total: Double`` inside the Receipt struct (proves the
    extractor fires inside a struct body, not only inside class bodies).
    """
    for name in ("id", "balance", "total"):
        row = _query_symbol(db_connection, symbol_name=name, symbol_type="property")
        assert row is not None, f"stored property {name!r} not indexed as 'property'"


def test_extracts_computed_property_simple_body(db_connection: sqlite3.Connection) -> None:
    """Simple-body computed properties (``var x: T { return ... }``) land
    as ``symbol_type='computed_property'``.

    ``displayName`` is on a class, ``taxIncluded`` is on a struct — both
    must be classified the same regardless of declaring kind.
    """
    for name in ("displayName", "taxIncluded"):
        row = _query_symbol(
            db_connection, symbol_name=name, symbol_type="computed_property"
        )
        assert row is not None, (
            f"simple-body computed {name!r} not indexed as 'computed_property'"
        )


def test_extracts_computed_property_with_get_set(db_connection: sqlite3.Connection) -> None:
    """Explicit get/set form (``var x: T { get { ... } set { ... } }``)
    is classified as computed.

    The Tree-sitter Swift grammar emits the SAME ``computed_property``
    node for both the simple-body and the explicit-getter forms, so this
    test mainly guards against a regression where the extractor's
    discriminator narrows to "must contain a return statement" or some
    other body-shape heuristic.
    """
    row = _query_symbol(
        db_connection, symbol_name="formatted", symbol_type="computed_property"
    )
    assert row is not None, "get/set computed 'formatted' not indexed as 'computed_property'"


def test_extracts_lazy_property(db_connection: sqlite3.Connection) -> None:
    """``lazy var foo: T = {...}()`` is classified as a stored property.

    Despite the closure body, ``lazy`` is a stored property modifier in
    Swift -- the closure runs once on first access and the result is
    cached. The Tree-sitter grammar reflects this: there is no
    ``computed_property`` child, only an ``=`` assignment. So we expect
    ``symbol_type='property'``.
    """
    row = _query_symbol(db_connection, symbol_name="heavy", symbol_type="property")
    assert row is not None, "lazy stored property 'heavy' not indexed as 'property'"


def test_extracts_static_property(db_connection: sqlite3.Connection) -> None:
    """``static var foo: T = ...`` is a (type-level) stored property.

    The 'static' modifier doesn't change the grammar's
    property_declaration shape, so we expect the same 'property'
    classification as instance-level stored properties.
    """
    row = _query_symbol(
        db_connection, symbol_name="sharedTaxRate", symbol_type="property"
    )
    assert row is not None, "static property 'sharedTaxRate' not indexed as 'property'"


def test_extracts_initializer(db_connection: sqlite3.Connection) -> None:
    """A plain ``init(...)`` lands as ``symbol_type='initializer'`` with
    a Swift selector.

    The selector format mirrors function selectors: ``init(label:label:)``.
    For HouseholdAccount's ``init(id:balance:)`` we expect
    ``init(id:balance:)`` exactly.
    """
    rows = db_connection.execute(
        "SELECT * FROM nodes WHERE symbol_name = 'init' "
        "AND symbol_type = 'initializer' "
        "AND file_path LIKE '%PropertiesAndInit.swift%' "
        "AND selector = 'init(id:balance:)'"
    ).fetchall()
    assert len(rows) == 1, (
        f"expected one init(id:balance:) row, got {len(rows)}: "
        f"{[dict(r) for r in rows]}"
    )
    row = rows[0]
    assert row["selector"] == "init(id:balance:)"
    assert row["language"] == "swift"


def test_extracts_failable_initializer(db_connection: sqlite3.Connection) -> None:
    """A failable ``init?(...)`` is still classified as 'initializer'.

    The ``?`` is not part of the selector form -- selectors capture
    parameter labels only -- so the failable init shows up with
    selector ``init(legacyString:)``. The variant flag (failable vs
    non-failable) is intentionally NOT modeled in this phase; that's
    deferred per the roadmap.
    """
    rows = db_connection.execute(
        "SELECT * FROM nodes WHERE symbol_name = 'init' "
        "AND symbol_type = 'initializer' "
        "AND file_path LIKE '%PropertiesAndInit.swift%' "
        "AND selector = 'init(legacyString:)'"
    ).fetchall()
    assert len(rows) == 1, (
        f"expected one init(legacyString:) row, got {len(rows)}: "
        f"{[dict(r) for r in rows]}"
    )


def test_extracts_deinitializer(db_connection: sqlite3.Connection) -> None:
    """``deinit`` lands as ``symbol_type='deinitializer'`` with no selector.

    Deinit takes no parameters and is unique within its declaring type,
    so the symbol_name is literally 'deinit' and selector is NULL.
    """
    row = _query_symbol(db_connection, symbol_name="deinit", symbol_type="deinitializer")
    assert row is not None, "deinit not indexed as 'deinitializer'"
    # Selector should be NULL (deinit has no call-site form).
    assert row["selector"] is None, (
        f"deinit should have NULL selector, got {row['selector']!r}"
    )
