"""
Tests for CALLS edges in the graph.

Validates:
- Function call chains are captured
- CALLS edges are created from caller to callee
- Documents current limitation: name-based linking fails for overloads
"""

import sqlite3


class TestFunctionCallChains:
    """Tests for function call chain tracing via CALLS edges."""

    def test_state_machine_start_calls_validate(self, db_connection: sqlite3.Connection):
        """Verify start() has CALLS edge to validate()."""
        edge = db_connection.execute("""
            SELECT e.* FROM edges e
            JOIN nodes src ON e.source_node_id = src.id
            WHERE src.symbol_name = 'start'
            AND e.edge_type = 'CALLS'
            AND e.target_symbol LIKE 'validate%'
        """).fetchone()

        assert edge is not None, "Missing CALLS edge: start -> validate"

    def test_state_machine_start_calls_process(self, db_connection: sqlite3.Connection):
        """Verify start() has CALLS edge to process()."""
        edge = db_connection.execute("""
            SELECT e.* FROM edges e
            JOIN nodes src ON e.source_node_id = src.id
            WHERE src.symbol_name = 'start'
            AND e.edge_type = 'CALLS'
            AND e.target_symbol LIKE 'process%'
        """).fetchone()

        assert edge is not None, "Missing CALLS edge: start -> process"

    def test_state_machine_process_calls_transform(self, db_connection: sqlite3.Connection):
        """Verify process() has CALLS edge to transform()."""
        edge = db_connection.execute("""
            SELECT e.* FROM edges e
            JOIN nodes src ON e.source_node_id = src.id
            WHERE src.symbol_name = 'process'
            AND e.edge_type = 'CALLS'
            AND e.target_symbol LIKE 'transform%'
        """).fetchone()

        assert edge is not None, "Missing CALLS edge: process -> transform"

    def test_state_machine_process_calls_complete(self, db_connection: sqlite3.Connection):
        """Verify process() has CALLS edge to complete()."""
        edge = db_connection.execute("""
            SELECT e.* FROM edges e
            JOIN nodes src ON e.source_node_id = src.id
            WHERE src.symbol_name = 'process'
            AND e.edge_type = 'CALLS'
            AND e.target_symbol LIKE 'complete%'
        """).fetchone()

        assert edge is not None, "Missing CALLS edge: process -> complete"


class TestFactoryCallChains:
    """Tests for factory pattern call chains."""

    def test_create_operation_calls_create_addition(self, db_connection: sqlite3.Connection):
        """Verify createOperation() has CALLS edge to createAddition()."""
        edge = db_connection.execute("""
            SELECT e.* FROM edges e
            JOIN nodes src ON e.source_node_id = src.id
            WHERE src.symbol_name = 'createOperation'
            AND e.edge_type = 'CALLS'
            AND e.target_symbol LIKE 'createAddition%'
        """).fetchone()

        assert edge is not None, "Missing CALLS edge: createOperation -> createAddition"

    def test_engine_evaluate_calls_factory(self, db_connection: sqlite3.Connection):
        """Verify evaluate() has CALLS edge to createOperation()."""
        edge = db_connection.execute("""
            SELECT e.* FROM edges e
            JOIN nodes src ON e.source_node_id = src.id
            WHERE src.symbol_name = 'evaluate'
            AND e.edge_type = 'CALLS'
            AND e.target_symbol LIKE 'createOperation%'
        """).fetchone()

        assert edge is not None, "Missing CALLS edge: evaluate -> createOperation"


class TestOverloadedFunctionCalls:
    """
    Tests for overloaded function call resolution.

    Phase 3 implements selector-based linking, which correctly distinguishes
    between overloaded functions with the same name but different argument labels
    (e.g., test(_:) vs test(something:)).
    """

    def test_call_unlabeled_targets_correct_overload(self, db_connection: sqlite3.Connection):
        """
        Verify callUnlabeled() CALLS edge points to test(_:), not test(something:).

        Uses selector-based matching for accurate overload resolution.
        """
        # Get the callUnlabeled function's CALLS edge - now uses full selector
        edge = db_connection.execute("""
            SELECT e.*, tgt.file_path, tgt.signature, tgt.selector FROM edges e
            JOIN nodes src ON e.source_node_id = src.id
            LEFT JOIN nodes tgt ON e.target_node_id = tgt.id
            WHERE src.symbol_name = 'callUnlabeled'
            AND e.edge_type = 'CALLS'
            AND e.target_symbol = 'test(_:)'
        """).fetchone()

        assert edge is not None, "Missing CALLS edge: callUnlabeled -> test(_:)"

        # Verify it points to the correct overload via selector
        assert edge["selector"] == "test(_:)", (
            f"Wrong overload linked. Expected test(_:), got: {edge['selector']}"
        )

    def test_call_something_targets_correct_overload(self, db_connection: sqlite3.Connection):
        """
        Verify callSomething() CALLS edge points to test(something:).
        """
        edge = db_connection.execute("""
            SELECT e.*, tgt.signature, tgt.selector FROM edges e
            JOIN nodes src ON e.source_node_id = src.id
            LEFT JOIN nodes tgt ON e.target_node_id = tgt.id
            WHERE src.symbol_name = 'callSomething'
            AND e.edge_type = 'CALLS'
            AND e.target_symbol = 'test(something:)'
        """).fetchone()

        assert edge is not None, "Missing CALLS edge: callSomething -> test(something:)"
        assert edge["selector"] == "test(something:)", (
            f"Wrong overload linked. Expected test(something:), got: {edge['selector']}"
        )

    def test_overload_edges_are_distinct(self, db_connection: sqlite3.Connection):
        """
        Verify that callUnlabeled, callSomething, and callSomethingElse
        each link to DIFFERENT target nodes.

        Selector-based linking ensures each overload is correctly identified.
        """
        selectors_to_check = [
            ("callUnlabeled", "test(_:)"),
            ("callSomething", "test(something:)"),
            ("callSomethingElse", "test(somethingElse:)"),
        ]
        target_node_ids = []

        for caller, expected_selector in selectors_to_check:
            edge = db_connection.execute(
                """
                SELECT e.target_node_id FROM edges e
                JOIN nodes src ON e.source_node_id = src.id
                WHERE src.symbol_name = ?
                AND e.edge_type = 'CALLS'
                AND e.target_symbol = ?
            """,
                (caller, expected_selector),
            ).fetchone()

            assert edge is not None, f"Missing CALLS edge: {caller} -> {expected_selector}"
            if edge and edge["target_node_id"]:
                target_node_ids.append(edge["target_node_id"])

        # All three should have edges
        assert len(target_node_ids) == 3, f"Expected 3 edges, found {len(target_node_ids)}"

        # All three should point to DIFFERENT nodes
        unique_targets = set(target_node_ids)
        assert len(unique_targets) == 3, (
            f"Expected 3 distinct target nodes, but got {len(unique_targets)}. "
            f"Target node IDs: {target_node_ids}"
        )


class TestCallEdgeCount:
    """Tests to verify expected number of CALLS edges."""

    def test_calls_edges_created(self, db_connection: sqlite3.Connection):
        """Verify CALLS edges exist in the database."""
        count = db_connection.execute(
            "SELECT COUNT(*) as cnt FROM edges WHERE edge_type = 'CALLS'"
        ).fetchone()["cnt"]

        assert count > 0, "No CALLS edges found in the database"

    def test_no_self_calls(self, db_connection: sqlite3.Connection):
        """Verify functions don't have CALLS edges to themselves (except recursion)."""
        # The only legitimate self-call is factorial -> factorial
        self_calls = db_connection.execute("""
            SELECT src.symbol_name, e.target_symbol FROM edges e
            JOIN nodes src ON e.source_node_id = src.id
            WHERE e.edge_type = 'CALLS'
            AND src.symbol_name = e.target_symbol
            AND src.symbol_name != 'factorial'
        """).fetchall()

        assert len(self_calls) == 0, (
            f"Found unexpected self-calls: {[r['symbol_name'] for r in self_calls]}"
        )
