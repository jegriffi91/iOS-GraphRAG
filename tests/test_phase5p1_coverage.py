"""Phase 5 P1 — enum cases + trailing closure CALLS + modifier chain CALLS.

The fixture ``EnumAndChains.swift`` packs every detection path into one
file so a single indexing run covers them all:

    - bare enum cases (``add``, ``subtract``, ``multiply``, ``divide``)
    - enum cases with associated values (``custom(name:fn:)``,
      ``configured(power:)``)
    - raw-value enum cases (``idle``, ``loading``, ``error`` of
      ``LoadingState: String``)
    - a SwiftUI-style modifier chain inside ``ChainExample.process()``
      that produces CALLS edges to ``Text``, ``padding``, ``background``,
      and ``onTapGesture``
    - a trailing closure on ``onTapGesture`` whose inner body invokes
      ``handleTap()`` and ``state.update()`` -- both must produce CALLS
      edges with ``process`` as the source.

Tests use the session-scoped ``db_connection`` fixture so they piggyback
on the single indexing run that other test modules also share, keeping
the suite fast.
"""

from __future__ import annotations

import sqlite3


def test_extracts_enum_case_simple(db_connection: sqlite3.Connection) -> None:
    """Bare cases (no associated value, no raw value) land as
    ``symbol_type='enum_case'`` with selector NULL.

    Covers ``CalculatorOperation.add`` and ``.subtract`` from the new
    fixture, plus ``LoadingState.idle`` (which is raw-valued -- still
    classified as ``enum_case``). Duplicate ``idle`` rows from
    ``StateMachine.State`` exist in the index, so we filter by file.
    """
    for name in ("add", "subtract", "idle"):
        rows = db_connection.execute(
            "SELECT * FROM nodes WHERE symbol_name = ? AND symbol_type = 'enum_case' "
            "AND file_path LIKE '%EnumAndChains.swift%'",
            (name,),
        ).fetchall()
        assert len(rows) == 1, (
            f"expected one enum_case row for {name!r}, got {len(rows)}: "
            f"{[dict(r) for r in rows]}"
        )


def test_extracts_enum_case_with_associated_values(
    db_connection: sqlite3.Connection,
) -> None:
    """``case custom(name: String, fn: ...)`` produces selector
    ``custom(name:fn:)``.

    The selector mirrors function selectors so the Phase 3 selector-based
    resolver can disambiguate associated-value cases from same-named
    bare cases without bespoke code.
    """
    row = db_connection.execute(
        "SELECT * FROM nodes WHERE symbol_name = 'custom' "
        "AND symbol_type = 'enum_case' "
        "AND file_path LIKE '%EnumAndChains.swift%'"
    ).fetchone()
    assert row is not None, "associated-value case 'custom' not indexed"
    assert row["selector"] == "custom(name:fn:)", (
        f"expected selector 'custom(name:fn:)', got {row['selector']!r}"
    )


def test_extracts_enum_case_raw_value(db_connection: sqlite3.Connection) -> None:
    """Raw-valued cases (``case idle = "idle"``) are classified as
    ``enum_case`` regardless of the trailing literal.

    The Tree-sitter Swift grammar puts the raw value after a ``=``
    sibling of the case identifier, which our extractor explicitly
    ignores in ``_extract_enum_case_names`` -- this test guards against
    a regression where the extractor fell through to treating the raw
    literal as a second case name.
    """
    for name in ("idle", "loading", "error"):
        rows = db_connection.execute(
            "SELECT * FROM nodes WHERE symbol_name = ? AND symbol_type = 'enum_case' "
            "AND file_path LIKE '%EnumAndChains.swift%'",
            (name,),
        ).fetchall()
        assert len(rows) == 1, (
            f"expected one enum_case row for raw-valued {name!r}, got {len(rows)}"
        )
        # Raw values do not produce a selector (no call-site form).
        assert rows[0]["selector"] is None, (
            f"raw-valued case {name!r} should have NULL selector, "
            f"got {rows[0]['selector']!r}"
        )


def test_extracts_modifier_chain_calls(db_connection: sqlite3.Connection) -> None:
    """``Text("Hello").padding().background(Color.red).onTapGesture { ... }``
    produces CALLS edges from ``process`` to each modifier in the chain.

    The Tree-sitter Swift grammar represents modifier chains as nested
    ``call_expression > navigation_expression > call_expression`` and
    the existing recursive walker in ``extract_swift_symbols`` already
    descends through every level of nesting, so each modifier call
    produces its own edge. This test pins that behavior.
    """
    expected_targets = ["padding()", "background(_:)", "onTapGesture()"]
    rows = db_connection.execute(
        """
        SELECT e.target_symbol FROM edges e
        JOIN nodes src ON e.source_node_id = src.id
        WHERE src.symbol_name = 'process'
        AND e.edge_type = 'CALLS'
        AND src.file_path LIKE '%EnumAndChains.swift%'
        """
    ).fetchall()
    actual_targets = {r["target_symbol"] for r in rows}
    for expected in expected_targets:
        assert expected in actual_targets, (
            f"missing modifier-chain CALLS edge: process -> {expected}; "
            f"got {sorted(actual_targets)}"
        )


def test_extracts_trailing_closure_inner_calls(
    db_connection: sqlite3.Connection,
) -> None:
    """The body of ``.onTapGesture { handleTap(); state.update() }`` is
    walked recursively, so ``process`` gets CALLS edges to both inner
    invocations.

    The Tree-sitter Swift grammar represents the trailing closure as a
    ``call_suffix > lambda_literal`` whose ``statements`` child contains
    the inner ``call_expression`` nodes. The extractor's recursive walk
    visits every descendant of ``function_body`` so the ``process``
    function captures the inner calls without bespoke code -- this test
    pins that behavior.
    """
    rows = db_connection.execute(
        """
        SELECT e.target_symbol FROM edges e
        JOIN nodes src ON e.source_node_id = src.id
        WHERE src.symbol_name = 'process'
        AND e.edge_type = 'CALLS'
        AND src.file_path LIKE '%EnumAndChains.swift%'
        """
    ).fetchall()
    targets = {r["target_symbol"] for r in rows}
    assert "handleTap()" in targets, (
        f"missing trailing-closure inner CALLS edge: process -> handleTap(); "
        f"got {sorted(targets)}"
    )
    assert "update()" in targets, (
        f"missing trailing-closure inner CALLS edge: process -> update(); "
        f"got {sorted(targets)}"
    )


def test_enum_cases_count_in_calculator_app(db_connection: sqlite3.Connection) -> None:
    """The new EnumAndChains.swift fixture contributes exactly 9 enum_case
    rows: 6 from CalculatorOperation plus 3 from LoadingState.

    A separate count check pins the project-wide total at 13 (the 9
    above plus 4 from ``StateMachine.State`` that the Phase 5 P1
    extractor now picks up where it previously did not). Both numbers
    are pinned so accidental fixture changes show up as a test failure
    rather than silently shifting downstream assertions.
    """
    fixture_count = db_connection.execute(
        "SELECT COUNT(*) AS cnt FROM nodes WHERE symbol_type = 'enum_case' "
        "AND file_path LIKE '%EnumAndChains.swift%'"
    ).fetchone()["cnt"]
    assert fixture_count == 9, (
        f"expected 9 enum_case rows in EnumAndChains.swift, got {fixture_count}"
    )

    project_total = db_connection.execute(
        "SELECT COUNT(*) AS cnt FROM nodes WHERE symbol_type = 'enum_case'"
    ).fetchone()["cnt"]
    assert project_total == 13, (
        f"expected 13 enum_case rows project-wide (9 from EnumAndChains.swift + "
        f"4 from StateMachine.State), got {project_total}"
    )
