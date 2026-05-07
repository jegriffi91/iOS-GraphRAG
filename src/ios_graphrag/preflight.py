"""Phase 3 P3.1 — first-run preflight check for iOS-GraphRAG.

Single-screen pass/fail per check. Run on a fresh corp laptop before any
real indexing work happens; surfaces the failures most likely to block
new teammates (Python version, MPS, free disk, model download, grammar
load, smoke index, semantic search round-trip).

Exit codes:
    0 — all hard checks passed (soft warnings allowed).
    1 — at least one hard check failed.

The footer prints copy-pasteable Copilot CLI / VS Code MCP config
snippets with the user's actual home directory baked in. The schema
of those config files has shifted across Copilot releases, so the
output includes a "verify against your current Copilot CLI version"
note (see roadmap P3.2).
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

# Use plain ASCII so the output renders cleanly inside CI logs and corp
# terminal emulators that fall back from a UTF-8-aware font.
PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"


@dataclass
class CheckResult:
    name: str
    status: str  # "pass" | "fail" | "warn"
    detail: str = ""
    remediation: str = ""

    def render(self) -> str:
        prefix = {"pass": PASS, "fail": FAIL, "warn": WARN}[self.status]
        line = f"{prefix:6s} {self.name}"
        if self.detail:
            line = f"{line}: {self.detail}"
        return line


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
MODEL_LOAD_TIMEOUT_SEC = 5 * 60
MIN_FREE_DISK_GB = 10
MIN_PYTHON = (3, 11)


def check_python_version() -> CheckResult:
    """Hard fail if running on Python <3.11 (matches pyproject `requires-python`)."""
    cur = sys.version_info
    cur_str = f"{cur.major}.{cur.minor}.{cur.micro}"
    if (cur.major, cur.minor) < MIN_PYTHON:
        return CheckResult(
            name="Python version",
            status="fail",
            detail=f"found Python {cur_str}; need >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}",
            remediation=(
                "Install Python 3.11+ from python.org or via homebrew "
                "(brew install python@3.12)."
            ),
        )
    return CheckResult(
        name="Python version",
        status="pass",
        detail=f"{cur_str}",
    )


def check_mps_available() -> CheckResult:
    """Soft warn if MPS is unavailable — embedding falls back to CPU."""
    try:
        import torch  # noqa: PLC0415  (lazy: torch is a heavy import)

        if torch.backends.mps.is_available():
            return CheckResult(
                name="MPS (Apple Silicon GPU)",
                status="pass",
                detail=f"torch {torch.__version__} reports MPS available",
            )
        return CheckResult(
            name="MPS (Apple Silicon GPU)",
            status="warn",
            detail=f"torch {torch.__version__} reports MPS NOT available",
            remediation=(
                "Embedding will use CPU; full reindex will be slower. "
                "Expected on Intel Mac / x86 Linux; otherwise check torch install."
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="MPS (Apple Silicon GPU)",
            status="warn",
            detail=f"could not import torch: {exc}",
            remediation="Embedding will use CPU; reinstall torch if this is unexpected.",
        )


def check_free_disk_space() -> CheckResult:
    """Hard fail if the home directory has < MIN_FREE_DISK_GB free."""
    home = Path.home()
    try:
        usage = shutil.disk_usage(str(home))
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="Free disk space",
            status="fail",
            detail=f"could not stat {home}: {exc}",
            remediation="Verify the home directory exists and is readable.",
        )
    free_gb = usage.free / (1024**3)
    if free_gb < MIN_FREE_DISK_GB:
        return CheckResult(
            name="Free disk space",
            status="fail",
            detail=f"{free_gb:.1f} GB free in {home}; need >= {MIN_FREE_DISK_GB} GB",
            remediation=(
                "Free up space or set IOS_GRAPHRAG_DB_PATH to a different volume."
            ),
        )
    return CheckResult(
        name="Free disk space",
        status="pass",
        detail=f"{free_gb:.1f} GB free in {home}",
    )


def check_embedding_model() -> CheckResult:
    """Try to load the embedding model. Triggers a ~500MB download on cold cache.

    Uses sentence_transformers directly (no subprocess); a load timeout is not
    trivially expressible without threading, so we lean on the user's network
    timeout settings. The 5-minute SLA in the roadmap is an external contract
    documented in the remediation copy below.
    """
    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        # trust_remote_code=True matches indexer.py / server.py.
        SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name=f"Embedding model ({MODEL_NAME})",
            status="fail",
            detail=f"{type(exc).__name__}: {exc}",
            remediation=(
                "Download once with: "
                f"`huggingface-cli download {MODEL_NAME} "
                f"--local-dir ~/.cache/ios-graphrag/models/nomic-embed-text-v1.5` "
                "and set IOS_GRAPHRAG_MODEL_DIR=/path/to/local/copy. "
                "If your network blocks HF, ask your team for the offline tarball."
            ),
        )
    return CheckResult(
        name=f"Embedding model ({MODEL_NAME})",
        status="pass",
        detail="model loaded (cached or downloaded successfully)",
    )


_SWIFT_SMOKE = (
    "import Foundation\n"
    "func add(_ a: Int, _ b: Int) -> Int { return a + b }\n"
    "class Foo {}\n"
)


def check_tree_sitter_swift() -> CheckResult:
    """Parse a 3-line Swift snippet and assert the expected node types appear.

    A grammar regression that drops `function_declaration` or
    `simple_identifier` would silently corrupt the index.
    """
    try:
        import tree_sitter_swift  # noqa: PLC0415
        from tree_sitter import Language, Parser  # noqa: PLC0415

        parser = Parser()
        parser.language = Language(tree_sitter_swift.language())
        tree = parser.parse(_SWIFT_SMOKE.encode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="Tree-sitter Swift parse",
            status="fail",
            detail=f"failed to load grammar / parse: {type(exc).__name__}: {exc}",
            remediation=(
                "tree-sitter-swift grammar mismatch; pin tree-sitter-swift>=0.7,<0.8 "
                "in pyproject.toml and reinstall with `uv sync`."
            ),
        )

    seen: set[str] = set()

    def walk(node) -> None:  # noqa: ANN001  (tree-sitter Node is duck-typed)
        seen.add(node.type)
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    required = {"function_declaration", "simple_identifier"}
    missing = required - seen
    if missing:
        return CheckResult(
            name="Tree-sitter Swift parse",
            status="fail",
            detail=f"missing expected node types: {sorted(missing)}",
            remediation=(
                "tree-sitter-swift grammar mismatch; pin tree-sitter-swift>=0.7,<0.8 "
                "in pyproject.toml and reinstall with `uv sync`."
            ),
        )
    return CheckResult(
        name="Tree-sitter Swift parse",
        status="pass",
        detail="grammar loaded; required node types found",
    )


def _smoke_swift_source() -> str:
    """Three-function Swift file used for the round-trip checks."""
    return textwrap.dedent(
        """\
        import Foundation

        struct Greeter {
            func hello(_ name: String) -> String { return "hello \\(name)" }
            func add(_ a: Int, _ b: Int) -> Int { return a + b }
        }

        class Worker {
            func run() {}
        }
        """
    )


def check_smoke_index(workdir: Path) -> Tuple[CheckResult, Optional[Path]]:
    """Build a tiny one-file repo, run the indexer subprocess, assert rows.

    Subprocess (rather than `index_repository(...)` direct call) so any
    import-time SSL/HF env-var games don't poison the parent process; this
    matches how conftest.py drives the indexer.
    """
    repo = workdir / "smoke_repo" / "Sources"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "Smoke.swift").write_text(_smoke_swift_source(), encoding="utf-8")

    db_path = workdir / "smoke.db"

    project_root = _project_root()
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ios_graphrag.indexer",
            "--repo",
            str(repo.parent),
            "--db",
            str(db_path),
            "--full",
        ],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=False,
        timeout=10 * 60,
    )
    if proc.returncode != 0:
        return (
            CheckResult(
                name="Smoke index (3-symbol Swift)",
                status="fail",
                detail=f"indexer subprocess exited rc={proc.returncode}",
                remediation=(
                    "Inspect indexing_errors.log and the indexer stderr above. "
                    "If sentence-transformers fails, see the embedding-model check."
                ),
            ),
            None,
        )

    if not db_path.exists():
        return (
            CheckResult(
                name="Smoke index (3-symbol Swift)",
                status="fail",
                detail="indexer ran clean but no DB file at output path",
                remediation="File a bug report via `ios-graphrag-doctor --bug-report`.",
            ),
            None,
        )

    import sqlite3  # noqa: PLC0415

    conn = sqlite3.connect(str(db_path))
    try:
        n_nodes = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        n_edges = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    finally:
        conn.close()

    if n_nodes < 3:
        return (
            CheckResult(
                name="Smoke index (3-symbol Swift)",
                status="fail",
                detail=f"expected >=3 nodes rows, got {n_nodes}",
                remediation="Tree-sitter parser may be silently failing; rerun preflight.",
            ),
            db_path,
        )
    # The 3-function file has at least the IMPORTS edge for Foundation.
    if n_edges < 1:
        return (
            CheckResult(
                name="Smoke index (3-symbol Swift)",
                status="fail",
                detail=f"nodes={n_nodes} but edges={n_edges} (expected >=1)",
                remediation="Edge builder regressed; check indexer logs.",
            ),
            db_path,
        )
    return (
        CheckResult(
            name="Smoke index (3-symbol Swift)",
            status="pass",
            detail=f"{n_nodes} nodes, {n_edges} edges in {db_path.name}",
        ),
        db_path,
    )


def check_round_trip_search(db_path: Path) -> CheckResult:
    """Hydrate the smoke DB into the server's globals, call _semantic_search."""
    try:
        # Import lazily so the first preflight steps don't transitively pull
        # in heavy server-side dependencies before they're known to work.
        from ios_graphrag import server  # noqa: PLC0415

        # Reset server globals so a previous test/preflight invocation in the
        # same interpreter doesn't pollute the matrix.
        import networkx as nx  # noqa: PLC0415

        server.GRAPH = nx.DiGraph()
        server.NODE_META = {}
        server.EMBEDDING_MATRIX = None
        server.EMBEDDING_IDS = []
        server.MODEL = None

        prev_db = os.environ.get("GRAPH_DB_PATH")
        os.environ["GRAPH_DB_PATH"] = str(db_path)
        try:
            server.load_graph(str(db_path))
            result = server._semantic_search(query="hello", top_k=3)
        finally:
            if prev_db is None:
                os.environ.pop("GRAPH_DB_PATH", None)
            else:
                os.environ["GRAPH_DB_PATH"] = prev_db
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="Round-trip semantic_search",
            status="fail",
            detail=f"{type(exc).__name__}: {exc}",
            remediation=(
                "Server hydrate path is broken; rerun the smoke index and check "
                "indexing_errors.log + server.log."
            ),
        )

    if not isinstance(result, dict):
        return CheckResult(
            name="Round-trip semantic_search",
            status="fail",
            detail=f"unexpected result type {type(result).__name__}",
            remediation="server._semantic_search return shape changed; update preflight.",
        )
    results = result.get("results")
    if not results:
        # If embeddings didn't land we'll see {"error": "No embeddings loaded..."}
        err = result.get("error", "no results")
        return CheckResult(
            name="Round-trip semantic_search",
            status="fail",
            detail=f"empty results: {err}",
            remediation=(
                "Check that the embedding step ran end-to-end during the smoke "
                "index (look for 'Indexing complete' in stderr)."
            ),
        )
    return CheckResult(
        name="Round-trip semantic_search",
        status="pass",
        detail=f"top hit: {results[0].get('symbol')} (score={results[0].get('score', 0):.3f})",
    )


# ---------------------------------------------------------------------------
# Footer / config snippet
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    """Best-effort project root for subprocess `cwd`.

    This file lives at <root>/src/ios_graphrag/preflight.py.
    """
    return Path(__file__).resolve().parents[2]


def render_config_footer(home: Optional[Path] = None) -> str:
    """Return the copy-pasteable Copilot CLI + VS Code MCP config snippet.

    The schema may have shifted across Copilot releases (P3.2 is empirical
    work the owner does on their machine). The output is annotated with a
    "verify against your current Copilot CLI version" note.
    """
    home = home or Path.home()
    db_path = home / ".cache" / "ios-graphrag" / "graph.sqlite"

    return textwrap.dedent(
        f"""
        ============================================================
         Next step: wire iOS-GraphRAG into your AI client.
        ============================================================

         Recommended DB path:
            {db_path}

         GitHub Copilot CLI — ~/.config/gh-copilot/config.yml
         (Note: verify against your current Copilot CLI version —
         config keys may have changed since this preflight was written.)

            mcp_servers:
              ios-graphrag:
                command: ios-graphrag-server
                env:
                  GRAPH_DB_PATH: {db_path}

         VS Code (settings.json):
         (Note: verify the exact key name — VS Code's Copilot extension
         has shifted between e.g. "github.copilot.chat.mcp.servers" and
         "github.copilot.advanced.mcpServers" across releases.)

            "github.copilot.chat.mcp.servers": {{
                "ios-graphrag": {{
                    "command": "ios-graphrag-server",
                    "env": {{
                        "GRAPH_DB_PATH": "{db_path}"
                    }}
                }}
            }}

         To build the index, run:
            ios-graphrag-index --repo /path/to/your/repo --db {db_path} --full

        """
    ).strip()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_all_checks() -> Tuple[List[CheckResult], int]:
    """Run every check sequentially. Returns (results, exit_code)."""
    results: List[CheckResult] = []

    # Hard checks first; if Python is too old or model load fails, the
    # downstream checks (smoke index, semantic search) won't be useful
    # data — but they'll still run so users get a complete picture.
    sequence: List[Callable[[], CheckResult]] = [
        check_python_version,
        check_mps_available,
        check_free_disk_space,
        check_embedding_model,
        check_tree_sitter_swift,
    ]
    for fn in sequence:
        try:
            results.append(fn())
        except Exception as exc:  # noqa: BLE001  (preflight must never crash)
            results.append(
                CheckResult(
                    name=fn.__name__,
                    status="fail",
                    detail=f"unexpected error: {type(exc).__name__}: {exc}",
                    remediation="File a bug report via `ios-graphrag-doctor --bug-report`.",
                )
            )

    # Smoke index + round-trip share a tmpdir so the round-trip can read
    # the DB the smoke index just wrote.
    smoke_db: Optional[Path] = None
    with tempfile.TemporaryDirectory(prefix="ios-graphrag-preflight-") as tmp:
        try:
            smoke_result, smoke_db = check_smoke_index(Path(tmp))
            results.append(smoke_result)
        except Exception as exc:  # noqa: BLE001
            results.append(
                CheckResult(
                    name="Smoke index (3-symbol Swift)",
                    status="fail",
                    detail=f"unexpected error: {type(exc).__name__}: {exc}",
                    remediation="File a bug report via `ios-graphrag-doctor --bug-report`.",
                )
            )
        if smoke_db is not None and smoke_db.exists():
            try:
                results.append(check_round_trip_search(smoke_db))
            except Exception as exc:  # noqa: BLE001
                results.append(
                    CheckResult(
                        name="Round-trip semantic_search",
                        status="fail",
                        detail=f"unexpected error: {type(exc).__name__}: {exc}",
                        remediation="File a bug report via `ios-graphrag-doctor --bug-report`.",
                    )
                )
        else:
            results.append(
                CheckResult(
                    name="Round-trip semantic_search",
                    status="fail",
                    detail="skipped: smoke index did not produce a DB",
                    remediation="Fix the smoke index check first.",
                )
            )

    failed = [r for r in results if r.status == "fail"]
    return results, (1 if failed else 0)


def render_results(results: List[CheckResult]) -> str:
    lines = ["", "iOS-GraphRAG preflight", "----------------------"]
    for r in results:
        lines.append(r.render())
        if r.status != "pass" and r.remediation:
            lines.append(f"        -> {r.remediation}")
    n_pass = sum(1 for r in results if r.status == "pass")
    n_warn = sum(1 for r in results if r.status == "warn")
    n_fail = sum(1 for r in results if r.status == "fail")
    lines.append("")
    lines.append(f"Summary: {n_pass} passed, {n_warn} warned, {n_fail} failed")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="ios-graphrag-preflight",
        description=(
            "Run a one-time fresh-machine readiness check for iOS-GraphRAG. "
            "Verifies Python version, MPS, free disk, embedding model load, "
            "tree-sitter grammar, and a tiny smoke-index round trip."
        ),
    )
    # Reserve --help only; no other flags today. The roadmap may add a
    # --smoke-only mode later (see Phase 6e CI gating in PRODUCTION_ROADMAP.md).
    parser.parse_args()

    results, rc = run_all_checks()
    print(render_results(results))
    if rc == 0:
        print()
        print(render_config_footer())
    else:
        print()
        print(
            "Preflight detected hard failures. Re-run after applying the "
            "remediation steps above. For deeper diagnostics, run:"
        )
        print("    ios-graphrag-doctor --bug-report")
    return rc


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
