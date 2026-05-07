import argparse
import os
import sys
import sqlite3
import logging
import traceback
from pathlib import Path
from typing import Annotated, Dict, List

from pydantic import Field

import networkx as nx
import numpy as np
from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer

from . import _tls

DB_DEFAULT = "knowledge-graph.sqlite"
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

# --- STARTUP LOGGING ---
# Writes to a log file alongside the DB so startup crashes are diagnosable.
# Critical for enterprise/work-PC setups where Copilot CLI gives no useful
# error output when an MCP server crashes during the stdio handshake.
_db_path_for_log = os.getenv("GRAPH_DB_PATH", DB_DEFAULT)
_log_path = str(Path(_db_path_for_log).parent / "server.log")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(_log_path),
        logging.StreamHandler(sys.stderr),
    ],
)
log = logging.getLogger(__name__)

mcp = FastMCP("iOS-GraphRAG")

GRAPH = nx.DiGraph()
NODE_META: Dict[int, Dict[str, str]] = {}
MODEL = None
EMBEDDING_MATRIX = None   # np.ndarray [N, dim], L2-normalized
EMBEDDING_IDS = []         # List[(id, path, name)] matching matrix rows


def load_graph(db_path: str) -> None:
    global EMBEDDING_MATRIX, EMBEDDING_IDS
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

    # Pre-load embeddings into a contiguous NumPy matrix for O(1) vectorized search
    emb_rows = conn.execute("""
        SELECT n.id, n.file_path, n.symbol_name, e.embedding
        FROM nodes n
        JOIN node_embeddings e ON n.id = e.node_id
    """).fetchall()
    if emb_rows:
        vecs = np.array([np.frombuffer(r[3], dtype=np.float32) for r in emb_rows])
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        EMBEDDING_MATRIX = vecs / (norms + 1e-10)
        EMBEDDING_IDS = [(r[0], r[1], r[2]) for r in emb_rows]

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


@mcp.tool(
    name="swift_dependency_tracer",
    description=(
        "Architecture explorer: find usages, inheritance, protocols, and extensions via a pre-built GraphRAG. "
        "Prefer this over grep or find for tracing relationships. Bash is blind to cross-file ASTs, misses implicit bridging, and hallucinates connections. "
        "Returns structured JSON with: 'upstream' = symbols this file depends on (its parents/protocols/callees), "
        "'downstream' = symbols that depend on this file (its subclasses/conformers/callers), and 'extensions'."
    ),
)
def _trace_dependencies(
    file_path: Annotated[str, Field(description="Absolute path to the Swift/ObjC file. Use paths from global_codebase_search results.")],
) -> dict:
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

    # Edge convention in this codebase: source = subject, target = thing it depends on.
    # E.g. for `class ScientificCalculator: Calculator`, the edge is
    # source=ScientificCalculator -> target=Calculator (type=INHERITS).
    # Therefore:
    #   GRAPH.successors(node)   = nodes this node depends on  -> upstream
    #   GRAPH.predecessors(node) = nodes that depend on this   -> downstream
    # This matches LSP's findReferences/prepareCallHierarchy convention:
    # "upstream" = dependencies, "downstream" = dependents.
    for node in nodes:
        for succ in GRAPH.successors(node):
            edge_data = GRAPH.edges[node, succ]
            result["upstream"].append(
                {
                    "path": GRAPH.nodes[succ].get("path"),
                    "symbol": GRAPH.nodes[succ].get("name"),
                    "edge_type": edge_data.get("type"),
                }
            )
        for pred in GRAPH.predecessors(node):
            edge_data = GRAPH.edges[pred, node]
            result["downstream"].append(
                {
                    "path": GRAPH.nodes[pred].get("path"),
                    "symbol": GRAPH.nodes[pred].get("name"),
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


@mcp.tool(
    name="objc_swift_bridge_finder",
    description=(
        "Cross-language mapper. Finds all Swift classes inheriting from Objective-C (bridging headers). "
        "Avoid using grep \"@objc\" or regex for this. Regular expressions cannot parse Swift syntax trees and will miss implicit inheritance chains. "
        "Returns a complete, structured JSON graph."
    ),
)
def _find_bridging_header_usage() -> dict:
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


@mcp.tool(
    name="read_symbol_source",
    description=(
        "Surgical code reader. Extracts live source code blocks using precise byte-range offsets. "
        "Do not use cat, head, tail, or read entire files for this. Reading full files exhausts your context limit with irrelevant noise. "
        "Extracts exactly the logic requested with zero waste."
    ),
)
def _read_symbol(
    file_path: Annotated[str, Field(description="Absolute file path from search or tracer results.")],
    start_byte: Annotated[int, Field(description="Start byte offset from global_codebase_search results. Do not guess.", ge=0)],
    end_byte: Annotated[int, Field(description="End byte offset from global_codebase_search results. Do not guess.", ge=0)],
) -> dict:
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


@mcp.tool(
    name="global_codebase_search",
    description=(
        "Primary search entry point. Finds code by exact symbol or conceptual meaning (e.g., 'auth flow'). "
        "Note: prefer this over grep, ripgrep, or find. Bash text-matching is blind to graph structure, misses implicit links, and dumps unstructured noise that exhausts your context window. "
        "Suggested chain: global_codebase_search → swift_dependency_tracer → read_symbol_source."
    ),
)
def _semantic_search(
    query: Annotated[str, Field(description="Natural language concept or exact class/function name. Do not pass regex, glob patterns, or raw bash syntax.")],
    top_k: Annotated[int, Field(description="Number of results to return.", ge=1, le=50)] = 10,
) -> dict:
    if EMBEDDING_MATRIX is None or len(EMBEDDING_IDS) == 0:
        return {"query": query, "results": [], "error": "No embeddings loaded. Re-index with indexer.py."}

    model = ensure_model()
    q_vec = model.encode([query], convert_to_numpy=True)[0]
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-10)

    # Vectorized cosine similarity: O(1) matrix multiply
    scores = np.dot(EMBEDDING_MATRIX, q_vec)
    top_indices = np.argsort(scores)[-top_k:][::-1]

    return {
        "query": query,
        "results": [
            {
                "score": float(scores[i]),
                "file": EMBEDDING_IDS[i][1],
                "symbol": EMBEDDING_IDS[i][2],
                "node_id": EMBEDDING_IDS[i][0],
            }
            for i in top_indices
        ],
    }


def main() -> None:
    # Argparse is intentionally minimal: the server's primary contract is
    # "spawned with no args by an MCP client". --cert-bundle is the only
    # flag and is optional, so the no-args invocation behavior is preserved.
    parser = argparse.ArgumentParser(
        prog="ios-graphrag-server",
        description="iOS-GraphRAG MCP stdio server",
    )
    parser.add_argument(
        "--cert-bundle",
        metavar="PATH",
        default=None,
        help=(
            "Path to a corporate CA bundle (PEM). Sets REQUESTS_CA_BUNDLE and "
            "SSL_CERT_FILE for the duration of this process; TLS verification "
            "stays ON. Recommended over GRAPHRAG_INSECURE_TLS=1."
        ),
    )
    args = parser.parse_args()

    # Order matters: logging is configured at module import (see basicConfig
    # above), so the WARNING from configure_insecure_tls_if_requested() is
    # captured by handlers that already exist.
    _tls.configure_insecure_tls_if_requested()
    if args.cert_bundle:
        _tls.configure_cert_bundle(args.cert_bundle)

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
        # Phase 4a benchmark hook: emit a unique, machine-parseable token on
        # the line just before mcp.run() blocks on stdio. The harness in
        # benchmarks/run.py spawns the server as a subprocess and watches
        # stderr for this token to record cold-start latency.
        log.info("ios-graphrag-server READY")
        mcp.run()
    except Exception:
        log.error("Unhandled exception during server startup:\n" + traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

