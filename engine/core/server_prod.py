import os
import sys
import sqlite3
import logging
import traceback
from pathlib import Path
from typing import Dict, List

import networkx as nx
import numpy as np
from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer

DB_DEFAULT = "knowledge-graph.sqlite"
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

# --- STARTUP LOGGING ---
# Writes to a log file alongside the DB so startup crashes are diagnosable.
# Critical for enterprise/work-PC setups where Copilot CLI gives no useful
# error output when an MCP server crashes during the stdio handshake.
_db_path_for_log = os.getenv("GRAPH_DB_PATH", DB_DEFAULT)
_log_path = str(Path(_db_path_for_log).parent / "server_prod.log")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(_log_path),
        logging.StreamHandler(sys.stderr),
    ],
)
log = logging.getLogger(__name__)

mcp = FastMCP("iOS-GraphRAG-Prod")

GRAPH = nx.DiGraph()
NODE_META: Dict[int, Dict[str, str]] = {}
MODEL = None


def load_graph(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    for row in conn.execute("SELECT id, file_path, symbol_name, symbol_type FROM nodes"):
        node_id, path, name, symbol_type = row
        GRAPH.add_node(node_id, path=path, name=name, type=symbol_type)
        NODE_META[node_id] = {"path": path, "name": name, "type": symbol_type}

    for row in conn.execute("SELECT source_node_id, target_node_id, edge_type FROM edges"):
        source_id, target_id, edge_type = row
        if target_id is None:
            continue
        GRAPH.add_edge(source_id, target_id, type=edge_type)
    conn.close()


def ensure_model() -> SentenceTransformer:
    global MODEL
    if MODEL is None:
        MODEL = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
        try:
            import torch

            if torch.backends.mps.is_available():
                MODEL.to("mps")
        except Exception:
            pass
    return MODEL


@mcp.tool()
def trace_dependencies(file_path: str) -> dict:
    """
    Trace the inheritance, protocol conformance, and extension relationships for a Swift/ObjC file.
    
    USE THIS INSTEAD OF: grep, find, or manual file navigation to understand how a file
    relates to the rest of the codebase.
    
    RETURNS:
    - upstream: Classes/protocols this file inherits from or conforms to
    - downstream: Classes that inherit from or depend on symbols in this file  
    - extensions: Files that extend types defined in this file
    
    EXAMPLE: trace_dependencies("/path/to/MyViewController.swift")
    """
    nodes = [n for n, data in GRAPH.nodes(data=True) if data.get("path") == file_path]
    if not nodes:
        return {"error": f"File not found in index: {file_path}"}

    result = {
        "target": {
            "path": file_path,
            "symbols": [GRAPH.nodes[n]["name"] for n in nodes],
        },
        "upstream": [],
        "downstream": [],
        "extensions": [],
    }

    for node in nodes:
        for pred in GRAPH.predecessors(node):
            edge_data = GRAPH.edges[pred, node]
            result["upstream"].append(
                {
                    "path": GRAPH.nodes[pred].get("path"),
                    "symbol": GRAPH.nodes[pred].get("name"),
                    "edge_type": edge_data.get("type"),
                }
            )
        for succ in GRAPH.successors(node):
            edge_data = GRAPH.edges[node, succ]
            result["downstream"].append(
                {
                    "path": GRAPH.nodes[succ].get("path"),
                    "symbol": GRAPH.nodes[succ].get("name"),
                    "edge_type": edge_data.get("type"),
                }
            )

    conn = sqlite3.connect(os.getenv("GRAPH_DB_PATH", DB_DEFAULT))
    extensions = conn.execute(
        """
        SELECT n.file_path
        FROM extension_map em
        JOIN nodes n ON n.id = em.extension_node_id
        JOIN nodes cn ON cn.id = em.canonical_node_id
        WHERE cn.file_path = ?
        """,
        (file_path,),
    ).fetchall()
    conn.close()
    result["extensions"] = [row[0] for row in extensions]
    return result


@mcp.tool()
def find_bridging_header_usage() -> dict:
    """
    Find all Swift classes that inherit from Objective-C classes (bridging header usage).
    
    USE THIS INSTEAD OF: grep -r "@objc" or searching for bridging header imports.
    
    This is essential for understanding Swift/ObjC interoperability in mixed codebases.
    """
    bridging_edges = [
        (u, v, d)
        for u, v, d in GRAPH.edges(data=True)
        if d.get("type") == "BRIDGING"
    ]
    return {
        "count": len(bridging_edges),
        "bridging_classes": [
            {
                "swift_class": GRAPH.nodes[u].get("name"),
                "swift_file": GRAPH.nodes[u].get("path"),
                "objc_parent": GRAPH.nodes[v].get("name"),
                "objc_file": GRAPH.nodes[v].get("path"),
            }
            for u, v, _ in bridging_edges
        ],
    }


@mcp.tool()
def read_symbol(file_path: str, start_byte: int, end_byte: int) -> dict:
    """
    Read the LIVE source code for a specific symbol using its byte range.
    
    USE THIS INSTEAD OF: cat, head, tail, or reading entire files.
    
    The byte ranges come from semantic_search results. This reads exactly the code
    you need without loading entire files, which is faster and uses less context.
    
    EXAMPLE: After semantic_search returns a result with node_id, use trace_dependencies
    to get the file_path and byte ranges, then call this to read the actual code.
    """
    try:
        with open(file_path, "rb") as handle:
            handle.seek(start_byte)
            content = handle.read(end_byte - start_byte)
        try:
            code = content.decode("utf-8")
        except UnicodeDecodeError:
            code = content.decode("latin-1")
        return {
            "file": file_path,
            "range": {"start": start_byte, "end": end_byte},
            "code": code,
        }
    except FileNotFoundError:
        return {
            "error": "FILE_DELETED",
            "message": f"File no longer exists: {file_path}. Index may be stale.",
        }


@mcp.tool()
def semantic_search(query: str, top_k: int = 10) -> dict:
    """
    Find code by MEANING, not just text matching. Uses AI embeddings for semantic similarity.
    
    USE THIS INSTEAD OF: grep -r "pattern", find . -name "*.swift", or browsing files.
    
    This tool understands concepts, not just keywords. For example:
    - "authentication flow" finds login/logout/session code
    - "network request handling" finds API clients and URLSession usage
    - "user profile management" finds profile-related ViewControllers and models
    
    RETURNS: Top K results ranked by semantic similarity, with file paths and symbol names.
    
    WORKFLOW:
    1. semantic_search("your concept") -> get file paths and node IDs
    2. trace_dependencies(file_path) -> understand relationships
    3. read_symbol(...) -> read the actual implementation
    """
    model = ensure_model()
    query_embedding = model.encode([query], convert_to_numpy=True)[0]
    query_norm = np.linalg.norm(query_embedding) + 1e-10

    conn = sqlite3.connect(os.getenv("GRAPH_DB_PATH", DB_DEFAULT))
    rows = conn.execute(
        """
        SELECT n.id, n.file_path, n.symbol_name, e.embedding
        FROM nodes n
        JOIN node_embeddings e ON n.id = e.node_id
        """
    ).fetchall()
    conn.close()

    results: List[Dict[str, object]] = []
    for node_id, path, name, emb_blob in rows:
        embedding = np.frombuffer(emb_blob, dtype=np.float32)
        score = float(np.dot(query_embedding, embedding) / ((np.linalg.norm(embedding) * query_norm) + 1e-10))
        results.append((score, path, name, node_id))

    results.sort(reverse=True, key=lambda x: x[0])
    return {
        "query": query,
        "results": [
            {"score": score, "file": path, "symbol": name, "node_id": node_id}
            for score, path, name, node_id in results[:top_k]
        ],
    }


def main() -> None:
    try:
        db_path = os.getenv("GRAPH_DB_PATH", DB_DEFAULT)
        log.info(f"Starting iOS-GraphRAG MCP server")
        log.info(f"Python: {sys.executable}")
        log.info(f"DB path: {db_path}")
        log.info(f"DB exists: {Path(db_path).exists()}")
        log.info(f"Log file: {_log_path}")

        if not Path(db_path).exists():
            log.error(
                f"FATAL: knowledge-graph.sqlite not found at '{db_path}'.\n"
                f"Set the GRAPH_DB_PATH env var to the absolute path of your .sqlite file.\n"
                f"See engine/CONNECTION_GUIDE.md for setup instructions."
            )
            sys.exit(1)

        load_graph(db_path)
        log.info("Graph loaded. Starting MCP stdio server...")
        mcp.run()
    except Exception:
        log.error("Unhandled exception during server startup:\n" + traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

