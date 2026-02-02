"""
Tests for verifying the graph structure after indexing.

Validates:
- Nodes are created for classes, protocols, structs, functions
- INHERITS edges for class inheritance
- CONFORMS edges for protocol conformance
- Extension map entries
"""
import sqlite3
from pathlib import Path


class TestNodeCreation:
    """Tests that nodes are correctly created during indexing."""
    
    def test_calculator_class_exists(self, db_connection: sqlite3.Connection):
        """Verify Calculator class is indexed."""
        row = db_connection.execute(
            "SELECT * FROM nodes WHERE symbol_name = 'Calculator' AND symbol_type = 'class'"
        ).fetchone()
        
        assert row is not None, "Calculator class not found in index"
        assert "Calculator.swift" in row["file_path"]
    
    def test_scientific_calculator_exists(self, db_connection: sqlite3.Connection):
        """Verify ScientificCalculator class is indexed."""
        row = db_connection.execute(
            "SELECT * FROM nodes WHERE symbol_name = 'ScientificCalculator' AND symbol_type = 'class'"
        ).fetchone()
        
        assert row is not None, "ScientificCalculator class not found in index"
    
    def test_operation_protocol_exists(self, db_connection: sqlite3.Connection):
        """Verify Operation protocol is indexed."""
        row = db_connection.execute(
            "SELECT * FROM nodes WHERE symbol_name = 'Operation' AND symbol_type = 'protocol'"
        ).fetchone()
        
        assert row is not None, "Operation protocol not found in index"
    
    def test_operation_conformers_exist(self, db_connection: sqlite3.Connection):
        """Verify structs conforming to Operation are indexed."""
        structs = ["Addition", "Subtraction", "Multiplication", "Division"]
        for struct_name in structs:
            row = db_connection.execute(
                "SELECT * FROM nodes WHERE symbol_name = ? AND symbol_type = 'struct'",
                (struct_name,)
            ).fetchone()
            assert row is not None, f"{struct_name} struct not found in index"
    
    def test_functions_are_indexed(self, db_connection: sqlite3.Connection):
        """Verify functions are indexed."""
        # Check for a known function
        row = db_connection.execute(
            "SELECT * FROM nodes WHERE symbol_name = 'calculate' AND symbol_type = 'function'"
        ).fetchone()
        
        assert row is not None, "calculate function not found in index"


class TestInheritanceEdges:
    """Tests for INHERITS edge type."""
    
    def test_scientific_inherits_calculator(self, db_connection: sqlite3.Connection):
        """Verify INHERITS edge from ScientificCalculator to Calculator."""
        edge = db_connection.execute("""
            SELECT e.* FROM edges e
            JOIN nodes src ON e.source_node_id = src.id
            WHERE src.symbol_name = 'ScientificCalculator'
            AND e.edge_type = 'INHERITS'
            AND e.target_symbol = 'Calculator'
        """).fetchone()
        
        assert edge is not None, "Missing INHERITS edge: ScientificCalculator -> Calculator"


class TestConformanceEdges:
    """Tests for CONFORMS edge type."""
    
    def test_addition_conforms_to_operation(self, db_connection: sqlite3.Connection):
        """Verify CONFORMS edge from Addition to Operation."""
        edge = db_connection.execute("""
            SELECT e.* FROM edges e
            JOIN nodes src ON e.source_node_id = src.id
            WHERE src.symbol_name = 'Addition'
            AND e.edge_type = 'CONFORMS'
            AND e.target_symbol = 'Operation'
        """).fetchone()
        
        assert edge is not None, "Missing CONFORMS edge: Addition -> Operation"
    
    def test_all_operations_conform(self, db_connection: sqlite3.Connection):
        """Verify all operation structs conform to Operation protocol."""
        structs = ["Addition", "Subtraction", "Multiplication", "Division"]
        for struct_name in structs:
            edge = db_connection.execute("""
                SELECT e.* FROM edges e
                JOIN nodes src ON e.source_node_id = src.id
                WHERE src.symbol_name = ?
                AND e.edge_type = 'CONFORMS'
                AND e.target_symbol = 'Operation'
            """, (struct_name,)).fetchone()
            
            assert edge is not None, f"Missing CONFORMS edge: {struct_name} -> Operation"


class TestNodeMetadata:
    """Tests for node metadata fields."""
    
    def test_signature_captured(self, db_connection: sqlite3.Connection):
        """Verify function signatures are captured."""
        row = db_connection.execute(
            "SELECT signature FROM nodes WHERE symbol_name = 'calculate' LIMIT 1"
        ).fetchone()
        
        assert row is not None
        assert "func calculate" in row["signature"]
    
    def test_byte_ranges_valid(self, db_connection: sqlite3.Connection):
        """Verify byte ranges are valid (start < end)."""
        invalid = db_connection.execute(
            "SELECT * FROM nodes WHERE start_byte >= end_byte"
        ).fetchall()
        
        assert len(invalid) == 0, f"Found {len(invalid)} nodes with invalid byte ranges"
    
    def test_line_numbers_present(self, db_connection: sqlite3.Connection):
        """Verify line numbers are present for all nodes."""
        missing = db_connection.execute(
            "SELECT * FROM nodes WHERE line_number IS NULL"
        ).fetchall()
        
        assert len(missing) == 0, f"Found {len(missing)} nodes without line numbers"
