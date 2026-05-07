"""Phase 5 P2 -- subscripts, typealiases, and ObjC class/method/property/
protocol/category coverage.

Two fixtures are exercised:

    * ``SubscriptsAndAliases.swift`` -- 2 file-scope typealiases
      (``OperationFn``, ``DigitMap``), one nested typealias (``Pair``
      inside ``Coordinate``), and 2 subscripts on ``HistoryStorage``
      with distinct selectors.
    * ``ObjCBridge/CalculatorMath.h`` + ``.m`` -- ``@interface``,
      ``@property``, ``-`` and ``+`` methods, ``@category (Trig)``, and
      ``@protocol``. The header and the implementation each contribute
      their own rows; the @category appears in both files.

Tests use the session-scoped ``db_connection`` fixture so they piggyback
on the single indexing run that other test modules share.
"""

from __future__ import annotations

import sqlite3


# ---------------------------------------------------------------------------
# Swift: typealias + subscript
# ---------------------------------------------------------------------------
def test_extracts_typealias(db_connection: sqlite3.Connection) -> None:
    """``typealias OperationFn = ...`` lands as ``symbol_type='typealias'``.

    Three typealiases must be present:
        * ``OperationFn`` and ``DigitMap`` at file scope,
        * ``Pair`` nested inside the ``Coordinate`` struct.
    Each gets a NULL selector (typealiases have no call-site form).
    """
    for name in ("OperationFn", "DigitMap", "Pair"):
        rows = db_connection.execute(
            "SELECT * FROM nodes WHERE symbol_name = ? AND symbol_type = 'typealias' "
            "AND file_path LIKE '%SubscriptsAndAliases.swift%'",
            (name,),
        ).fetchall()
        assert len(rows) == 1, (
            f"expected one typealias row for {name!r}, got {len(rows)}: "
            f"{[dict(r) for r in rows]}"
        )
        assert rows[0]["selector"] is None, (
            f"typealias {name!r} should have NULL selector, got {rows[0]['selector']!r}"
        )
        assert rows[0]["language"] == "swift"


def test_extracts_subscript(db_connection: sqlite3.Connection) -> None:
    """``subscript(key:)`` and ``subscript(index:withDefault:)`` both land
    as ``symbol_type='subscript'`` with the literal symbol name
    ``subscript`` and distinct selectors.

    Selector format mirrors function selectors: parameter labels joined
    with colons, surrounded by ``subscript(...)``. Overloads are
    disambiguated via the selector column (matches how Swift overloaded
    functions are handled in Phase 5 P0).
    """
    rows = db_connection.execute(
        "SELECT selector FROM nodes WHERE symbol_name = 'subscript' "
        "AND symbol_type = 'subscript' "
        "AND file_path LIKE '%SubscriptsAndAliases.swift%' "
        "ORDER BY selector"
    ).fetchall()
    selectors = [r["selector"] for r in rows]
    assert selectors == ["subscript(index:withDefault:)", "subscript(key:)"], (
        f"expected both subscript selectors, got {selectors}"
    )


# ---------------------------------------------------------------------------
# Objective-C: class, method, property, protocol, category
# ---------------------------------------------------------------------------
def test_extracts_objc_class(db_connection: sqlite3.Connection) -> None:
    """``@interface CalculatorMath : NSObject`` lands as
    ``symbol_type='objc_class'``.

    The header and .m each contribute a row; the test asserts at least
    one with the expected language. ``inherits`` is not exposed in the
    nodes table directly but the row's existence is what matters here.
    """
    rows = db_connection.execute(
        "SELECT * FROM nodes WHERE symbol_name = 'CalculatorMath' "
        "AND symbol_type = 'objc_class'"
    ).fetchall()
    assert len(rows) >= 1, f"expected >=1 objc_class row for CalculatorMath, got {len(rows)}"
    for row in rows:
        assert row["language"] == "objc", (
            f"objc_class row should have language='objc', got {row['language']!r}"
        )


def test_extracts_objc_method_selector(db_connection: sqlite3.Connection) -> None:
    """``- (double)addValue:(double)a toValue:(double)b`` produces a
    method row whose ``selector`` is ``addValue:toValue:``.

    The canonical ObjC selector form joins each colon-terminated keyword;
    the symbol_name carries the same value because ObjC has no
    name-vs-call-site distinction the way Swift does.
    """
    rows = db_connection.execute(
        "SELECT * FROM nodes WHERE selector = 'addValue:toValue:' "
        "AND symbol_type = 'objc_method'"
    ).fetchall()
    assert len(rows) >= 1, (
        f"expected >=1 objc_method row with selector 'addValue:toValue:', got {len(rows)}"
    )
    for row in rows:
        assert row["symbol_name"] == "addValue:toValue:", (
            f"symbol_name should mirror the selector, got {row['symbol_name']!r}"
        )


def test_extracts_objc_property(db_connection: sqlite3.Connection) -> None:
    """``@property NSString *name`` and ``@property NSInteger precision``
    both land as ``symbol_type='objc_property'``.

    The property name is read from the inner ``identifier`` of the
    ``struct_declarator`` (with ``pointer_declarator`` for object-typed
    properties; bare identifier for primitive scalars).
    """
    for name in ("name", "precision"):
        row = db_connection.execute(
            "SELECT * FROM nodes WHERE symbol_name = ? "
            "AND symbol_type = 'objc_property'",
            (name,),
        ).fetchone()
        assert row is not None, f"objc_property {name!r} not indexed"
        assert row["language"] == "objc"


def test_extracts_objc_category(db_connection: sqlite3.Connection) -> None:
    """``@interface CalculatorMath (Trig)`` lands as
    ``symbol_type='category'`` with the parenthesized name ``Trig`` as
    ``symbol_name``.

    The host class is captured separately as an ``objc_class`` row; the
    category is its own row tied to that host (extends column).
    """
    rows = db_connection.execute(
        "SELECT * FROM nodes WHERE symbol_name = 'Trig' "
        "AND symbol_type = 'category'"
    ).fetchall()
    assert len(rows) >= 1, f"expected >=1 category row for 'Trig', got {len(rows)}"


def test_extracts_objc_class_method(db_connection: sqlite3.Connection) -> None:
    """``+ (instancetype)sharedInstance`` (a class method) is extracted
    as an ``objc_method`` row with selector ``sharedInstance``.

    Class-vs-instance method (the leading ``+`` vs ``-``) is not
    modeled in this phase -- the canonical selector is enough to
    disambiguate within a class because Swift selectors and ObjC
    selectors live in disjoint namespaces.
    """
    rows = db_connection.execute(
        "SELECT * FROM nodes WHERE selector = 'sharedInstance' "
        "AND symbol_type = 'objc_method'"
    ).fetchall()
    assert len(rows) >= 1, (
        f"expected >=1 objc_method row with selector 'sharedInstance', got {len(rows)}"
    )


def test_extracts_objc_protocol(db_connection: sqlite3.Connection) -> None:
    """``@protocol CalculatorMathProtocol <NSObject>`` lands as
    ``symbol_type='objc_protocol'``.

    The protocol's adopted protocols (``<NSObject>``) are stored in the
    ``conforms`` field but only the row's existence is asserted here --
    conformance edge tests live in the graph-structure suite.
    """
    row = db_connection.execute(
        "SELECT * FROM nodes WHERE symbol_name = 'CalculatorMathProtocol' "
        "AND symbol_type = 'objc_protocol'"
    ).fetchone()
    assert row is not None, "objc_protocol 'CalculatorMathProtocol' not indexed"
    assert row["language"] == "objc"
