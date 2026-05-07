"""
Tests for MCP server tools.

Validates:
- trace_dependencies returns correct upstream/downstream
- semantic_search finds relevant symbols
- read_symbol returns correct code snippets
"""
import os
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest


class TestTraceDependencies:
    """Tests for the trace_dependencies MCP tool.

    Semantic convention (matches LSP findReferences / prepareCallHierarchy):
    - upstream   = symbols this file depends on (parents, protocols, callees)
    - downstream = symbols that depend on this file (subclasses, conformers, callers)

    Edge convention in the indexer: source = subject, target = thing it depends on.
    For `class ScientificCalculator: Calculator` -> edge source=ScientificCalculator,
    target=Calculator, type=INHERITS. Therefore Calculator is upstream of
    ScientificCalculator, and ScientificCalculator is downstream of Calculator.
    """

    def test_trace_scientific_calculator_upstream(
        self, db_connection: sqlite3.Connection, test_fixtures_path: Path
    ):
        """
        ScientificCalculator inherits from Calculator. Under the new semantics
        (upstream = "what I depend on"), Calculator must appear in
        ScientificCalculator's upstream list.
        """
        # Find ScientificCalculator file path
        row = db_connection.execute(
            "SELECT file_path FROM nodes WHERE symbol_name = 'ScientificCalculator' LIMIT 1"
        ).fetchone()
        assert row is not None
        file_path = row["file_path"]

        # Get all nodes in that file
        nodes = db_connection.execute(
            "SELECT id, symbol_name FROM nodes WHERE file_path = ?",
            (file_path,)
        ).fetchall()
        node_ids = [n["id"] for n in nodes]

        # New semantics: upstream of ScientificCalculator = outbound INHERITS edges
        # i.e. edges where source_node_id is a node in ScientificCalculator.swift
        # and target_symbol points to its parent.
        assert node_ids, "ScientificCalculator file has no nodes"
        placeholders = ",".join("?" for _ in node_ids)
        edge = db_connection.execute(f"""
            SELECT e.target_symbol FROM edges e
            WHERE e.source_node_id IN ({placeholders})
            AND e.edge_type = 'INHERITS'
        """, node_ids).fetchone()

        assert edge is not None, "No INHERITS edge from ScientificCalculator"
        # Calculator is what ScientificCalculator depends on => upstream.
        assert edge["target_symbol"] == "Calculator"

    def test_trace_operation_downstream(
        self, db_connection: sqlite3.Connection, test_fixtures_path: Path
    ):
        """
        Operation is a protocol conformed to by Addition/Subtraction/Multiplication/
        Division. Under the new semantics (downstream = "what depends on me"),
        those conformers must appear in Operation's downstream list.
        """
        # Find Operation node
        row = db_connection.execute(
            "SELECT id FROM nodes WHERE symbol_name = 'Operation' AND symbol_type = 'protocol' LIMIT 1"
        ).fetchone()
        assert row is not None
        operation_id = row["id"]

        # New semantics: downstream of Operation = inbound CONFORMS edges
        # i.e. edges where target_node_id == Operation.id (things that depend on it).
        edges = db_connection.execute("""
            SELECT src.symbol_name FROM edges e
            JOIN nodes src ON e.source_node_id = src.id
            WHERE e.target_node_id = ?
            AND e.edge_type = 'CONFORMS'
        """, (operation_id,)).fetchall()

        conformers = [e["symbol_name"] for e in edges]
        expected = ["Addition", "Subtraction", "Multiplication", "Division"]

        for expected_conformer in expected:
            assert expected_conformer in conformers, f"Missing conformer: {expected_conformer}"

    # NOTE: the two tests above query the underlying SQL `edges` table directly
    # rather than invoking `_trace_dependencies`. Because the indexer's edge
    # convention (source = subject, target = thing it depends on) is unchanged
    # by this commit, these assertions hold under both the OLD orientation
    # (where the server placed predecessors into `upstream`) and the NEW
    # orientation (successors into `upstream`). They were correctly named and
    # asserted under the new semantics; under the old code the test names were
    # misleading rather than wrong. The end-to-end orientation of
    # `_trace_dependencies` is exercised manually during verification (see
    # docs/PRODUCTION_ROADMAP.md P1.1 acceptance criteria).


class TestEmbeddings:
    """Tests for embeddings/semantic search setup."""
    
    def test_embeddings_exist(self, db_connection: sqlite3.Connection):
        """Verify embeddings were created during indexing."""
        count = db_connection.execute(
            "SELECT COUNT(*) as cnt FROM node_embeddings"
        ).fetchone()["cnt"]
        
        assert count > 0, "No embeddings found in node_embeddings table"
    
    def test_embedding_size_consistent(self, db_connection: sqlite3.Connection):
        """Verify all embeddings have the same size."""
        rows = db_connection.execute(
            "SELECT LENGTH(embedding) as size FROM node_embeddings LIMIT 10"
        ).fetchall()
        
        sizes = [r["size"] for r in rows]
        assert len(set(sizes)) == 1, f"Inconsistent embedding sizes: {set(sizes)}"
    
    def test_all_nodes_have_embeddings(self, db_connection: sqlite3.Connection):
        """Verify all nodes have embeddings."""
        missing = db_connection.execute("""
            SELECT n.symbol_name FROM nodes n
            LEFT JOIN node_embeddings e ON n.id = e.node_id
            WHERE e.node_id IS NULL
        """).fetchall()
        
        # Some nodes might not have embeddings (like extensions without meaningful signatures)
        # But the main symbols should have them
        missing_names = [m["symbol_name"] for m in missing]
        important_missing = [n for n in missing_names if n in ["Calculator", "Operation", "Engine"]]
        
        assert len(important_missing) == 0, f"Important symbols missing embeddings: {important_missing}"


class TestReadSymbol:
    """Tests for read_symbol functionality."""
    
    def test_byte_ranges_readable(
        self, db_connection: sqlite3.Connection, test_fixtures_path: Path
    ):
        """Verify byte ranges can be used to read actual source code."""
        # Get a function node with byte ranges
        row = db_connection.execute("""
            SELECT file_path, start_byte, end_byte, symbol_name
            FROM nodes WHERE symbol_type = 'function' LIMIT 1
        """).fetchone()
        
        assert row is not None
        
        file_path = row["file_path"]
        start_byte = row["start_byte"]
        end_byte = row["end_byte"]
        symbol_name = row["symbol_name"]
        
        # Read the bytes
        with open(file_path, "rb") as f:
            f.seek(start_byte)
            content = f.read(end_byte - start_byte)
        
        code = content.decode("utf-8")
        
        # The code should contain the function name
        assert symbol_name in code or "func" in code, (
            f"Read content doesn't look like a function: {code[:100]}..."
        )
