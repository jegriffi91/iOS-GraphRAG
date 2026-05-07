import argparse
import functools
import os
import sys
import sqlite3
import logging
import time
import traceback
import uuid
from pathlib import Path
from typing import Annotated, Any, Callable, Dict, List

from pydantic import Field

import networkx as nx
import numpy as np
from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer

from . import _migrations, _tls

DB_DEFAULT = "knowledge-graph.sqlite"
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

# Phase 6b: log handlers are configured in ``main()`` via
# :func:`_logging.setup_logging`. Tests and other importers see a default
# logger; the file/stderr handlers attach when the CLI entry point runs.
log = logging.getLogger(__name__)


def _traced_handler(*, tool_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Wrap an MCP tool handler with per-call trace IDs.

    Every invocation gets a fresh 8-char UUID4 prefix (8 chars is plenty for
    grep / jq filtering without bloating log lines). Lifecycle:

      1. Entry: ``log.info("tool call begin", extra={trace_id, tool, args})``.
      2. Body runs.
      3a. Success: ``log.info("tool call end", extra={trace_id, duration_ms})``.
          If the result is a dict, ``trace_id`` is also injected so callers
          can grep logs from the response payload.
      3b. Error: ``log.exception(...)`` with ``trace_id`` and ``duration_ms``
          attached, then re-raise. The trace_id is included in the error
          payload (when the handler returns a dict with an ``error`` key) or
          the re-raised exception's ``args``.

    Decorated functions keep their signature — :mod:`mcp.server.fastmcp`
    introspects the wrapped function via :func:`functools.wraps` for tool
    schema generation.
    """

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*call_args: Any, **call_kwargs: Any) -> Any:
            trace_id = uuid.uuid4().hex[:8]
            # ``call_args`` are positional; FastMCP always invokes via
            # kwargs for typed parameters, so ``args_payload`` should
            # normally capture the full call. We log positional args
            # defensively in case a future call site bypasses kwargs.
            #
            # Field name note: ``args`` is reserved on
            # :class:`logging.LogRecord` (it's the format-string args
            # tuple), so we surface the call inputs as ``tool_args`` to
            # avoid the ``Attempt to overwrite 'args'`` KeyError.
            args_payload: Dict[str, Any] = dict(call_kwargs)
            if call_args:
                args_payload["_positional"] = list(call_args)
            log.info(
                "tool call begin",
                extra={
                    "trace_id": trace_id,
                    "tool": tool_name,
                    "tool_args": args_payload,
                },
            )
            t0 = time.monotonic()
            try:
                result = fn(*call_args, **call_kwargs)
            except Exception:
                duration_ms = round((time.monotonic() - t0) * 1000, 3)
                log.exception(
                    "tool call error",
                    extra={
                        "trace_id": trace_id,
                        "tool": tool_name,
                        "duration_ms": duration_ms,
                    },
                )
                raise
            duration_ms = round((time.monotonic() - t0) * 1000, 3)
            log.info(
                "tool call end",
                extra={
                    "trace_id": trace_id,
                    "tool": tool_name,
                    "duration_ms": duration_ms,
                },
            )
            # If the handler returned a dict-shaped result, surface the
            # trace_id back to the caller so an error payload can be
            # cross-referenced against server.log via grep.
            if isinstance(result, dict):
                result.setdefault("trace_id", trace_id)
            return result

        return wrapper

    return decorate

mcp = FastMCP("iOS-GraphRAG")

GRAPH = nx.DiGraph()
NODE_META: Dict[int, Dict[str, str]] = {}
MODEL = None
EMBEDDING_MATRIX = None   # np.ndarray [N, dim], L2-normalized
EMBEDDING_IDS = []         # List[(id, path, name)] matching matrix rows

# Phase 4f.2: track DB mtime so each tool call can cheaply check whether the
# indexer has rewritten the database underneath us. ``_LOADED_MTIME`` is
# refreshed inside ``load_graph``; ``_reload_if_stale`` is called by every
# MCP tool handler before it touches the in-memory graph.
_LOADED_MTIME: float = 0.0


def _check_schema_version_or_exit(db_path: str) -> None:
    """Verify the DB schema matches what this build expects.

    Phase 6a contract: the server never writes schema changes (the
    indexer owns that, to avoid concurrent-writer races). At startup we
    read ``MAX(schema_version.version)`` and compare against the
    highest migration this code knows about. Any mismatch is fatal:
    older DB on newer code -> rerun the indexer; newer DB on older
    code -> upgrade ios-graphrag.
    """
    conn = sqlite3.connect(db_path)
    try:
        try:
            db_v = _migrations.current_version(conn)
        except Exception as exc:
            log.error("Failed to read schema_version from %s: %s", db_path, exc)
            sys.exit(1)
    finally:
        conn.close()

    code_v = _migrations.max_known_version()
    if db_v < code_v:
        log.error(
            "Database at v%d, server expects v%d. "
            "Run `ios-graphrag-index --repo <repo> --db %s` to apply migrations.",
            db_v,
            code_v,
            db_path,
        )
        sys.exit(1)
    if db_v > code_v:
        log.error(
            "Database at v%d, server only knows up to v%d. "
            "Upgrade ios-graphrag.",
            db_v,
            code_v,
        )
        sys.exit(1)
    log.info("Database schema v%d", db_v)


def load_graph(db_path: str) -> None:
    global EMBEDDING_MATRIX, EMBEDDING_IDS, _LOADED_MTIME
    # Phase 4f.2: a reload completely replaces the in-memory graph; clear
    # any prior state first so stale nodes/edges from a previous load
    # don't survive into the new view.
    GRAPH.clear()
    NODE_META.clear()
    EMBEDDING_MATRIX = None
    EMBEDDING_IDS = []

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

    # Phase 4f.2: snapshot the mtime _after_ we've finished reading. If the
    # indexer rewrites the DB during this load (rare but possible on a busy
    # box), the next ``_reload_if_stale`` call will see the newer mtime and
    # reload again.
    try:
        _LOADED_MTIME = os.path.getmtime(db_path)
    except OSError:
        _LOADED_MTIME = 0.0


def _reload_if_stale() -> None:
    """Reload ``GRAPH`` + ``EMBEDDING_MATRIX`` if the DB file has been
    rewritten since we last loaded it.

    Phase 4f.2 cache-drift fix: a long-lived MCP server process can serve
    stale graph data after the indexer rewrites the SQLite DB. Comparing
    mtime is one ``stat(2)`` syscall per tool call — cheaper than the
    alternative of a fully manual restart and far cheaper than waiting on
    a content hash. On the indexer's rebuild path the file's mtime always
    advances (SQLite truncates/writes on commit), so this is safe.
    """
    global _LOADED_MTIME
    db_path = os.getenv("GRAPH_DB_PATH", DB_DEFAULT)
    try:
        current = os.path.getmtime(db_path)
    except OSError:
        # DB went missing; let downstream code surface the real error.
        return
    if current > _LOADED_MTIME:
        log.info(
            "Database mtime changed (%s -> %s); reloading",
            _LOADED_MTIME,
            current,
        )
        load_graph(db_path)
        # ``load_graph`` updates ``_LOADED_MTIME`` itself, but mirror it
        # here so the local ``current`` value is authoritative even on
        # very fast successive writes.
        _LOADED_MTIME = current


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
@_traced_handler(tool_name="swift_dependency_tracer")
def _trace_dependencies(
    file_path: Annotated[str, Field(description="Absolute path to the Swift/ObjC file. Use paths from global_codebase_search results.")],
) -> dict:
    _reload_if_stale()
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
@_traced_handler(tool_name="objc_swift_bridge_finder")
def _find_bridging_header_usage() -> dict:
    _reload_if_stale()
    # Phase 4f.1: push the BRIDGING filter into SQL. Walking GRAPH.edges is
    # O(E) on every call — fine for the test fixture's tens of edges, but
    # the production roadmap pegs this at >1M edges where the in-memory
    # scan dominated. The SQL form uses ``idx_edges_type`` (declared in
    # engine/database/schema.sql) so the filter is index-backed and only
    # fans out to the joined ``nodes`` rows we actually return.
    db_path = os.getenv("GRAPH_DB_PATH", DB_DEFAULT)
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT src.symbol_name AS swift_class,
                   src.file_path   AS swift_file,
                   tgt.symbol_name AS objc_parent,
                   tgt.file_path   AS objc_file
            FROM edges e
            JOIN nodes src ON src.id = e.source_node_id
            JOIN nodes tgt ON tgt.id = e.target_node_id
            WHERE e.edge_type = 'BRIDGING'
            ORDER BY src.file_path, src.line_number
            """
        ).fetchall()
    finally:
        conn.close()
    return {
        "count": len(rows),
        "bridging_classes": [
            {
                "swift_class": row[0],
                "swift_file": row[1],
                "objc_parent": row[2],
                "objc_file": row[3],
            }
            for row in rows
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
@_traced_handler(tool_name="read_symbol_source")
def _read_symbol(
    file_path: Annotated[str, Field(description="Absolute file path from search or tracer results.")],
    start_byte: Annotated[int, Field(description="Start byte offset from global_codebase_search results. Do not guess.", ge=0)],
    end_byte: Annotated[int, Field(description="End byte offset from global_codebase_search results. Do not guess.", ge=0)],
) -> dict:
    _reload_if_stale()
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
@_traced_handler(tool_name="global_codebase_search")
def _semantic_search(
    query: Annotated[str, Field(description="Natural language concept or exact class/function name. Do not pass regex, glob patterns, or raw bash syntax.")],
    top_k: Annotated[int, Field(description="Number of results to return.", ge=1, le=50)] = 10,
) -> dict:
    _reload_if_stale()
    if EMBEDDING_MATRIX is None or len(EMBEDDING_IDS) == 0:
        return {"query": query, "results": [], "error": "No embeddings loaded. Re-index with indexer.py."}

    model = ensure_model()
    q_vec = model.encode([query], convert_to_numpy=True)[0]
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-10)

    # Vectorized cosine similarity: O(1) matrix multiply
    scores = np.dot(EMBEDDING_MATRIX, q_vec)
    # Phase 4d.1: ``argpartition`` is O(N) and gives us the top-k unsorted;
    # a follow-up O(k log k) sort over just the k winners orders them.
    # Total: O(N + k log k) vs the previous ``argsort``'s O(N log N).
    # ``argpartition`` raises if k > N, hence the ``min`` guard.
    k = min(top_k, scores.shape[0])
    unsorted_top = np.argpartition(scores, -k)[-k:]
    top_indices = unsorted_top[np.argsort(scores[unsorted_top])[::-1]]

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

    # Phase 6b: centralized logger setup. Stderr keeps human-readable text
    # for the MCP harness's stderr capture; the rotating file handler at
    # ~/Library/Logs/ios-graphrag/server.log gets JSON lines. setup_logging()
    # returns the path of the file handler so we can log it (handy for
    # support-ticket triage).
    from ._logging import setup_logging
    log_file = setup_logging("server")

    # Order matters: TLS configuration must come AFTER logging is wired so
    # the WARNING emitted by configure_insecure_tls_if_requested() reaches
    # both stderr and the JSON file.
    _tls.configure_insecure_tls_if_requested()
    if args.cert_bundle:
        _tls.configure_cert_bundle(args.cert_bundle)

    try:
        db_path = os.getenv("GRAPH_DB_PATH", DB_DEFAULT)
        log.info(f"Starting iOS-GraphRAG MCP server")
        log.info(f"Python: {sys.executable}")
        log.info(f"DB path: {db_path}")
        log.info(f"DB exists: {Path(db_path).exists()}")
        log.info(f"Log file: {log_file}")

        if not Path(db_path).exists():
            log.error(
                f"FATAL: knowledge-graph.sqlite not found at '{db_path}'.\n"
                f"Set the GRAPH_DB_PATH env var to the absolute path of your .sqlite file.\n"
                f"See engine/CONNECTION_GUIDE.md for setup instructions."
            )
            sys.exit(1)

        # Phase 6a: schema-version gate. The server never auto-applies
        # migrations (that would race against a concurrent indexer
        # process); it only verifies that the DB matches the schema this
        # build was compiled against and refuses to start otherwise.
        _check_schema_version_or_exit(db_path)

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

