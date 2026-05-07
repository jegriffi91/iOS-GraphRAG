"""Phase 4a benchmark harness for iOS-GraphRAG.

Measures full + incremental indexing wall time and peak RSS, server cold-start
latency, and p50/p95/p99 latency distributions for ``semantic_search`` and
``trace_dependencies``. Outputs a single JSON file under
``benchmarks/results/<git-sha>-<timestamp>.json`` so multiple runs can be
diffed by Phase 4b-f.

Usage::

    uv run python benchmarks/run.py --repo /path/to/repo --output benchmarks/results/
    uv run python benchmarks/run.py --repo /path/to/repo --db /tmp/bench.db --output benchmarks/results/

The harness is intentionally self-contained — no pytest, no fixtures, no
hidden state. A run does the following, in order:

1. Index the repo from scratch with ``GRAPHRAG_LOG_LEVEL=DEBUG``. The indexer
   emits ``PHASE_START``/``PHASE_END`` markers (see ``src/ios_graphrag/indexer.py``)
   that we scrape from stderr to derive a per-phase wall-time breakdown.
2. Spawn ``ios-graphrag-server`` as a subprocess with ``GRAPH_DB_PATH`` set
   and watch stderr for the ``ios-graphrag-server READY`` line. The wall-time
   between spawn and that line is the cold-start measurement.
3. Tear the server down (it doesn't service queries — we use direct in-process
   calls below) and load the same DB into our own process via
   ``server.load_graph()``. This gives us a callable ``_semantic_search`` and
   ``_trace_dependencies`` without the MCP stdio round-trip; that's the
   apples-to-apples comparison for per-call latency, since the server itself
   adds only JSON-RPC framing on top of the same handlers.
4. Build a query corpus of 1000 random ``symbol_name`` values from the index,
   call ``_semantic_search(query=q, top_k=5)`` in a tight loop, and record
   percentiles. Same for ``_trace_dependencies`` over 1000 random
   ``file_path`` values.
5. For incremental measurements: ``git stash`` the repo (defensive — guards
   against a partial run leaving the user's working tree dirty), then for
   each delta size in {10, 100, 1000} sample N files that changed in recent
   git history, ``git checkout HEAD~k -- <files>``, run an incremental
   reindex, then ``git checkout HEAD -- <files>`` to restore. After all
   deltas, ``git stash pop`` to restore the user's working state.

Sections that fail (e.g. not enough git history, server fails to start) are
recorded as ``{"skipped": true, "reason": "..."}`` rather than crashing the
run. The harness must produce SOME JSON file or fail loudly; partial results
are valuable.

NOTE on RSS units: ``resource.getrusage(...).ru_maxrss`` returns bytes on
macOS but kilobytes on Linux. The harness records the platform and converts
to bytes; consumers can trust ``peak_rss_bytes``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import random
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Repo-relative imports for the in-process measurements.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _platform_rss_to_bytes(rss: int) -> int:
    """Convert ``ru_maxrss`` to bytes regardless of OS.

    Per ``man 2 getrusage``: macOS reports bytes; Linux reports kilobytes.
    BSDs vary. We branch on ``sys.platform`` and accept the small risk that
    a unicorn platform reports something else — the alternative (calling
    ``psutil``) adds a dep we'd rather not require for one number.
    """
    if sys.platform == "darwin":
        return int(rss)
    # Linux + most BSDs: kilobytes.
    return int(rss) * 1024


def _peak_rss_bytes() -> int:
    """Peak resident set size for THIS process so far, in bytes."""
    return _platform_rss_to_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _git_sha(repo: Path) -> str:
    """Short git SHA at HEAD. Returns ``"unknown"`` on failure (no .git, etc.)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _orchestrator_sha() -> str:
    """Short SHA of the orchestrator (this codebase) at HEAD.

    Distinct from the indexed repo's SHA — the metadata records both, since
    benchmark results need to be tied to the iOS-GraphRAG version that
    produced them, not just the repo being measured.
    """
    return _git_sha(PROJECT_ROOT)


def _count_loc(repo: Path) -> Optional[int]:
    """Cheap LOC estimate: ``wc -l`` over .swift/.h/.m. Returns None on failure.

    Uses ``find -print0 | xargs -0`` to handle paths with spaces. This is a
    rough metric — comments + blank lines are counted — but for the harness's
    purpose (giving the result reader a "this is what scale we're at" hint)
    it's adequate.
    """
    if not repo.is_dir():
        return None
    try:
        # find ... | xargs wc -l. Pipe through awk to print only the total.
        find = subprocess.run(
            ["find", str(repo),
             "-type", "f",
             "(", "-name", "*.swift", "-o", "-name", "*.h", "-o", "-name", "*.m", ")",
             "-not", "-path", "*/.build/*",
             "-not", "-path", "*/Pods/*",
             "-print0"],
            capture_output=True, check=True, timeout=60,
        )
        if not find.stdout:
            return 0
        wc = subprocess.run(
            ["xargs", "-0", "wc", "-l"],
            input=find.stdout, capture_output=True, check=True, timeout=120,
        )
        # The last line is "<total> total" for multi-file output, or
        # "<lines> <single_file>" for a single file.
        lines = wc.stdout.decode("utf-8", errors="ignore").strip().splitlines()
        if not lines:
            return 0
        last = lines[-1].strip().split()
        if not last:
            return 0
        return int(last[0])
    except Exception:
        return None


def _torch_version() -> str:
    """Return torch.__version__ if importable, else "unavailable"."""
    try:
        import torch  # noqa: F401  (just to read __version__)
        return torch.__version__
    except Exception:
        return "unavailable"


def _parse_phase_markers(stderr_text: str) -> Dict[str, float]:
    """Extract per-phase wall times from indexer DEBUG markers.

    The indexer emits paired ``PHASE_START <name>`` / ``PHASE_END <name>``
    log lines with ISO timestamps. We pair them up and produce a dict of
    ``{phase}_seconds``. Phases that didn't run (e.g. no files to parse on
    a no-op incremental) are simply absent from the output.
    """
    import re
    # Match: "2026-05-06 12:34:56,789 [DEBUG] ... PHASE_START hash"
    pattern = re.compile(
        r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*?PHASE_(?P<edge>START|END) (?P<name>\w+)",
        re.MULTILINE,
    )
    starts: Dict[str, dt.datetime] = {}
    durations: Dict[str, float] = {}
    for m in pattern.finditer(stderr_text):
        ts = dt.datetime.strptime(m["ts"], "%Y-%m-%d %H:%M:%S,%f")
        if m["edge"] == "START":
            starts[m["name"]] = ts
        else:  # END
            start_ts = starts.pop(m["name"], None)
            if start_ts is not None:
                durations[f"{m['name']}_seconds"] = (ts - start_ts).total_seconds()
    return durations


def _run_indexer(
    repo: Path, db_path: Path, full: bool
) -> Tuple[float, int, str]:
    """Run the indexer subprocess. Returns (wall_seconds, peak_rss_bytes, stderr).

    Subprocess is the right boundary because the indexer spawns its own
    embedding-worker subprocess (via ProcessPoolExecutor) and we want the
    parent's peak RSS, which is reported on exit. ``time.monotonic`` brackets
    the wall time. ``GRAPHRAG_LOG_LEVEL=DEBUG`` enables the phase markers.
    """
    env = os.environ.copy()
    env["GRAPHRAG_LOG_LEVEL"] = "DEBUG"
    cmd = [
        sys.executable, "-m", "ios_graphrag.indexer",
        "--repo", str(repo),
        "--db", str(db_path),
    ]
    if full:
        cmd.append("--full")
    t0 = time.monotonic()
    proc = subprocess.run(
        cmd, capture_output=True, text=True, env=env,
        cwd=str(PROJECT_ROOT),
        timeout=3600,
    )
    elapsed = time.monotonic() - t0
    if proc.returncode != 0:
        # Surface useful debugging context — a failed run shouldn't be silent.
        raise RuntimeError(
            f"Indexer subprocess failed (rc={proc.returncode}):\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    # Peak RSS of the *child* via ru_maxrss is unfortunately not portable;
    # ``getrusage(RUSAGE_CHILDREN)`` aggregates across all reaped children.
    # That's good enough for our purpose — the indexer is the only thing
    # we've forked at this point.
    peak_rss = _platform_rss_to_bytes(
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    )
    return elapsed, peak_rss, proc.stderr


def _query_db(db_path: Path, query: str) -> List:
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(query).fetchall()
    finally:
        conn.close()


def _measure_full_index(
    repo: Path, db_path: Path
) -> Dict:
    """Section 1: full index pass. Always runs; section is mandatory.

    On an empty/no-Swift repo, the indexer still completes cleanly (zero
    files, zero symbols). We accept that and report symbol_count=0.
    """
    elapsed, peak_rss, stderr = _run_indexer(repo, db_path, full=True)
    phases = _parse_phase_markers(stderr)
    if db_path.exists():
        symbol_count_rows = _query_db(db_path, "SELECT COUNT(*) FROM nodes")
        edge_count_rows = _query_db(db_path, "SELECT COUNT(*) FROM edges")
        symbol_count = int(symbol_count_rows[0][0]) if symbol_count_rows else 0
        edge_count = int(edge_count_rows[0][0]) if edge_count_rows else 0
        db_size = os.path.getsize(db_path)
    else:
        # Empty repo: indexer didn't write a DB. That's fine — we report zeros.
        symbol_count = 0
        edge_count = 0
        db_size = 0
    return {
        "wall_seconds": elapsed,
        "peak_rss_bytes": peak_rss,
        "phases": phases,
        "symbol_count": symbol_count,
        "edge_count": edge_count,
        "db_size_bytes": db_size,
    }


def _git(repo: Path, *args: str, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    """``git -C <repo> ...`` helper. Never cd's; uses -C exclusively."""
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=capture, text=True, check=check, timeout=60,
    )


def _sample_changed_files(repo: Path, n: int) -> Optional[List[str]]:
    """Sample N files that recently changed in git history, repo-relative.

    We need a "realistic delta" — files the user is likely to touch — and the
    cheap proxy is "files that have changed in the last few hundred commits."
    Returns None if the repo has insufficient history.
    """
    try:
        # diff HEAD~k..HEAD --name-only across a wide window; if k > history,
        # git errors out. We try shrinking k until something works.
        for k in (200, 100, 50, 20, 10, 5, 2, 1):
            try:
                out = _git(repo, "diff", "--name-only", f"HEAD~{k}..HEAD", check=True)
                files = [
                    line for line in out.stdout.splitlines()
                    if line and (repo / line).is_file()
                ]
                if len(files) >= n:
                    random.shuffle(files)
                    return files[:n]
                if files and k <= 5:
                    # Not enough but we have something — caller decides.
                    return None
            except subprocess.CalledProcessError:
                continue
        return None
    except Exception:
        return None


def _measure_incremental(
    repo: Path, db_path: Path
) -> Dict:
    """Section 2: incremental at deltas of 10/100/1000 files.

    Strategy:
      * git stash any uncommitted changes (defensive — never disturb the user)
      * for each delta:
          - sample N files that changed in recent history
          - git checkout HEAD~1 -- <those files> (stage a realistic delta)
          - run incremental reindex (no --full)
          - git checkout HEAD -- <those files> (restore working tree)
          - record metrics
      * git stash pop

    Skipped sections record ``{"skipped": true, "reason": "..."}`` so callers
    can tell "we tried and bailed cleanly" from "we forgot."
    """
    out: Dict[str, Dict] = {}
    deltas = (10, 100, 1000)

    # Sanity check: is this a git repo?
    try:
        _git(repo, "rev-parse", "--git-dir", check=True)
    except Exception:
        for d in deltas:
            out[f"{d}_files"] = {"skipped": True, "reason": "not a git repository"}
        return out

    # Defensive stash — we want to not touch the user's working state.
    stashed = False
    try:
        status = _git(repo, "status", "--porcelain", check=True)
        if status.stdout.strip():
            _git(repo, "stash", "push", "-u", "-m", "ios-graphrag benchmark stash", check=True)
            stashed = True
    except Exception as exc:
        for d in deltas:
            out[f"{d}_files"] = {"skipped": True, "reason": f"git stash failed: {exc}"}
        return out

    try:
        for n in deltas:
            files = _sample_changed_files(repo, n)
            if not files:
                out[f"{n}_files"] = {
                    "skipped": True,
                    "reason": f"insufficient git history for {n}-file delta",
                }
                continue
            try:
                # Stage delta: roll listed files back one commit.
                _git(repo, "checkout", "HEAD~1", "--", *files, check=True)
                # Run incremental reindex.
                elapsed, peak_rss, stderr = _run_indexer(repo, db_path, full=False)
                phases = _parse_phase_markers(stderr)
                out[f"{n}_files"] = {
                    "wall_seconds": elapsed,
                    "peak_rss_bytes": peak_rss,
                    "phases": phases,
                    "files_changed": len(files),
                }
                # Restore working tree.
                _git(repo, "checkout", "HEAD", "--", *files, check=True)
            except Exception as exc:
                out[f"{n}_files"] = {
                    "skipped": True,
                    "reason": f"incremental run failed: {type(exc).__name__}: {exc}",
                }
                # Best-effort restore — if checkout failed, the working tree
                # may already be inconsistent; the stash pop below will try
                # to re-apply on top.
                try:
                    _git(repo, "checkout", "HEAD", "--", *files, check=False)
                except Exception:
                    pass
    finally:
        if stashed:
            try:
                _git(repo, "stash", "pop", check=False)
            except Exception:
                # Don't crash the harness if stash pop fails — the user can
                # recover manually with ``git stash list / git stash apply``.
                pass

    return out


def _measure_server_cold_start(
    db_path: Path
) -> Dict:
    """Section 3: spawn the server and wait for the READY token on stderr.

    We use a token-based approach (not a JSON-RPC handshake) because the
    server's MCP loop is async, stdin-driven, and we don't want to depend on
    the Python MCP client. The server emits ``ios-graphrag-server READY`` on
    the line just before ``mcp.run()`` blocks; that's the cold-start moment.

    For "first tool result" — the canonical second metric in the issue spec —
    we'd need to drive a JSON-RPC ``initialize`` + ``tools/list`` exchange
    over stdin, and the server uses FastMCP's framing which is non-trivial
    to spoof. We compromise: ``spawn_to_first_result_seconds`` is the same as
    ``spawn_to_ready_seconds`` because the server is ready to handle a tool
    call the instant it's ready to read stdin. If the owner wants a true
    end-to-end-with-handshake number, that's a follow-up; the cold-start
    bottleneck (model + graph load) is captured by ``spawn_to_ready``.
    """
    if not db_path.exists():
        return {
            "skipped": True,
            "reason": f"no DB at {db_path} — cannot start server",
        }
    env = os.environ.copy()
    env["GRAPH_DB_PATH"] = str(db_path)
    cmd = [sys.executable, "-m", "ios_graphrag.server"]
    t0 = time.monotonic()
    proc = subprocess.Popen(
        cmd, env=env,
        cwd=str(PROJECT_ROOT),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered so we read READY promptly
    )
    ready_at: Optional[float] = None
    try:
        # Read stderr line by line until we see the READY token, with a
        # 60-second ceiling. On a cold cache, model load dominates and this
        # may take 8-15s on the real repo; on an empty/tiny DB it's a few
        # hundred ms.
        deadline = t0 + 60.0
        while time.monotonic() < deadline:
            line = proc.stderr.readline()
            if not line:
                # Process likely exited; check returncode below.
                if proc.poll() is not None:
                    break
                continue
            if "ios-graphrag-server READY" in line:
                ready_at = time.monotonic()
                break
        if ready_at is None:
            # Drain remaining output for the report.
            try:
                tail_stderr = proc.stderr.read()
            except Exception:
                tail_stderr = ""
            return {
                "skipped": True,
                "reason": (
                    f"server did not emit READY token within 60s "
                    f"(rc={proc.poll()}, tail_stderr={tail_stderr!r})"
                ),
            }
        spawn_to_ready = ready_at - t0
        return {
            "spawn_to_ready_seconds": spawn_to_ready,
            # See docstring: in this harness, ready == ready-for-first-call.
            # A real first-result handshake would add JSON-RPC framing latency
            # (typically <10ms) on top.
            "spawn_to_first_result_seconds": spawn_to_ready,
        }
    finally:
        # Server blocks on stdin; closing it is the polite shutdown signal,
        # but it doesn't always exit promptly. Terminate to be sure.
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass


def _percentiles(values: List[float]) -> Dict[str, float]:
    """Return p50/p95/p99/min/max/mean for a list of millisecond latencies.

    Empty input -> all NaNs. We use ``numpy`` since the harness already pulls
    it transitively via the project. ``method="linear"`` is the default for
    np.percentile and matches what most reviewers will expect.
    """
    import numpy as np
    if not values:
        return {"p50": float("nan"), "p95": float("nan"), "p99": float("nan"),
                "min": float("nan"), "max": float("nan"), "mean": float("nan"), "n": 0}
    arr = np.array(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "n": int(len(values)),
    }


def _measure_search_latency(
    db_path: Path, n_queries: int = 1000
) -> Dict:
    """Section 4 + 5: in-process latency distributions for the two hot tools.

    We import ``ios_graphrag.server`` and call its handlers directly (same
    pattern as ``tests/test_mcp_tools.py::test_trace_dependencies_orientation_via_handler``).
    This bypasses MCP's JSON-RPC framing — the framing adds <10ms vs the
    handler call itself, so for percentile measurement direct invocation is
    the right boundary. ``ensure_model()`` triggers a model load on the first
    ``_semantic_search`` call (cold) and uses the cached model for the rest;
    we discard the first call when computing percentiles below.
    """
    out = {
        "semantic_search_latency_ms": {"skipped": True, "reason": "not run"},
        "trace_dependencies_latency_ms": {"skipped": True, "reason": "not run"},
    }
    if not db_path.exists():
        out["semantic_search_latency_ms"] = {"skipped": True, "reason": "no DB"}
        out["trace_dependencies_latency_ms"] = {"skipped": True, "reason": "no DB"}
        return out

    # Sample the query corpus from the database BEFORE importing server (so
    # an empty index short-circuits without a model load).
    name_rows = _query_db(db_path, "SELECT symbol_name FROM nodes")
    file_rows = _query_db(db_path, "SELECT DISTINCT file_path FROM nodes")
    if not name_rows:
        out["semantic_search_latency_ms"] = {"skipped": True, "reason": "no symbols indexed"}
        out["trace_dependencies_latency_ms"] = {"skipped": True, "reason": "no symbols indexed"}
        return out

    names = [r[0] for r in name_rows if r[0]]
    paths = [r[0] for r in file_rows if r[0]]
    random.shuffle(names)
    random.shuffle(paths)
    queries = names[:n_queries]
    files = paths[:n_queries]

    # Now import server and load the graph against the bench DB. The env var
    # is what server.py reads inside the extensions sub-query of
    # ``_trace_dependencies``.
    os.environ["GRAPH_DB_PATH"] = str(db_path)
    try:
        from ios_graphrag import server
    except Exception as exc:
        out["semantic_search_latency_ms"] = {"skipped": True, "reason": f"server import failed: {exc}"}
        out["trace_dependencies_latency_ms"] = {"skipped": True, "reason": f"server import failed: {exc}"}
        return out

    server.GRAPH.clear()
    server.NODE_META.clear()
    try:
        server.load_graph(str(db_path))
    except Exception as exc:
        out["semantic_search_latency_ms"] = {"skipped": True, "reason": f"load_graph failed: {exc}"}
        out["trace_dependencies_latency_ms"] = {"skipped": True, "reason": f"load_graph failed: {exc}"}
        return out

    # ---- semantic_search ----
    sem_latencies_ms: List[float] = []
    try:
        # Warm-up: first call pays for model load; subsequent calls don't.
        # We do this OUTSIDE the timing loop so percentiles aren't skewed.
        if queries:
            try:
                server._semantic_search(query=queries[0], top_k=5)
            except Exception:
                # If the model load itself fails, skip this whole section
                # rather than report bogus numbers.
                pass
        for q in queries:
            t0 = time.perf_counter()
            try:
                server._semantic_search(query=q, top_k=5)
            except Exception:
                # An individual query failure shouldn't kill the run. Skip
                # this sample.
                continue
            sem_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        out["semantic_search_latency_ms"] = _percentiles(sem_latencies_ms)
    except Exception as exc:
        out["semantic_search_latency_ms"] = {"skipped": True, "reason": f"latency loop crashed: {exc}"}

    # ---- trace_dependencies ----
    trace_latencies_ms: List[float] = []
    try:
        for fp in files:
            t0 = time.perf_counter()
            try:
                server._trace_dependencies(file_path=fp)
            except Exception:
                continue
            trace_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        out["trace_dependencies_latency_ms"] = _percentiles(trace_latencies_ms)
    except Exception as exc:
        out["trace_dependencies_latency_ms"] = {"skipped": True, "reason": f"latency loop crashed: {exc}"}

    return out


def _measure_memory(db_path: Path) -> Dict:
    """Section 6: memory snapshots — embedding-matrix bytes + graph node/edge counts.

    The embedding matrix size is computed from the row count and embedding
    dimension (read from the first row). We don't actually load the matrix
    here; that already happened inside ``_measure_search_latency`` if it
    ran, and the indexer's ``store_embeddings`` path means every embedding
    in the DB is the same dimension. For an empty DB we report 0.

    Graph node/edge counts come from the DB rather than the (already loaded)
    ``server.GRAPH`` because importing server may have failed. SQL is the
    source of truth.
    """
    out: Dict[str, int] = {
        "embedding_matrix_bytes": 0,
        "graph_nodes": 0,
        "graph_edges": 0,
    }
    if not db_path.exists():
        return out
    try:
        rows = _query_db(db_path, "SELECT COUNT(*) FROM node_embeddings")
        n_emb = int(rows[0][0]) if rows else 0
        if n_emb > 0:
            # Read dimension from the first row (each blob is fp32, so
            # bytes / 4 == dim).
            sample = _query_db(db_path, "SELECT embedding FROM node_embeddings LIMIT 1")
            dim_bytes = len(sample[0][0]) if sample and sample[0] and sample[0][0] else 0
            out["embedding_matrix_bytes"] = n_emb * dim_bytes
        out["graph_nodes"] = int(_query_db(db_path, "SELECT COUNT(*) FROM nodes")[0][0])
        # Resolved edges only — that's what NetworkX actually loads.
        out["graph_edges"] = int(
            _query_db(db_path, "SELECT COUNT(*) FROM edges WHERE target_node_id IS NOT NULL")[0][0]
        )
    except Exception:
        pass
    return out


def _build_metadata(repo: Path) -> Dict:
    return {
        "git_sha": _orchestrator_sha(),
        "repo_git_sha": _git_sha(repo),
        "timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repo_path": str(repo.resolve()),
        "loc": _count_loc(repo),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "python_version": platform.python_version(),
        "torch_version": _torch_version(),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="benchmarks/run.py",
        description="Phase 4a benchmark harness for iOS-GraphRAG.",
    )
    parser.add_argument("--repo", required=True, help="Path to the repository to benchmark.")
    parser.add_argument(
        "--db", default=None,
        help=(
            "Path for the benchmark SQLite DB. Defaults to a temp file. "
            "If you pass this and the file exists, it will be overwritten by "
            "the full-index pass."
        ),
    )
    parser.add_argument(
        "--output", default=str(PROJECT_ROOT / "benchmarks" / "results"),
        help="Directory for the result JSON. Created if missing.",
    )
    parser.add_argument(
        "--n-queries", type=int, default=1000,
        help="Number of queries for the latency distributions (default: 1000).",
    )
    parser.add_argument(
        "--seed", type=int, default=20260506,
        help="RNG seed for query/file sampling (default: deterministic).",
    )
    args = parser.parse_args(argv)

    random.seed(args.seed)

    repo = Path(args.repo).expanduser().resolve()
    if not repo.is_dir():
        print(f"ERROR: --repo does not exist or is not a directory: {repo}", file=sys.stderr)
        return 2

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # DB path: caller-provided or a tmp file we own.
    if args.db:
        db_path = Path(args.db).expanduser().resolve()
        cleanup_db = False
    else:
        import tempfile
        tmpdir = Path(tempfile.mkdtemp(prefix="ios-graphrag-bench-"))
        db_path = tmpdir / "bench.sqlite"
        cleanup_db = True

    metadata = _build_metadata(repo)
    print(f"[benchmark] orchestrator sha: {metadata['git_sha']}", file=sys.stderr)
    print(f"[benchmark] repo:             {repo}", file=sys.stderr)
    print(f"[benchmark] db:               {db_path}", file=sys.stderr)
    print(f"[benchmark] output dir:       {output_dir}", file=sys.stderr)

    result: Dict = {"metadata": metadata}

    # 1. Full index (mandatory).
    print("[benchmark] section 1: full index...", file=sys.stderr)
    try:
        result["full_index"] = _measure_full_index(repo, db_path)
    except Exception as exc:
        result["full_index"] = {"skipped": True, "reason": f"{type(exc).__name__}: {exc}"}

    # 2. Incremental.
    print("[benchmark] section 2: incremental deltas...", file=sys.stderr)
    try:
        result["incremental"] = _measure_incremental(repo, db_path)
    except Exception as exc:
        result["incremental"] = {"skipped": True, "reason": f"{type(exc).__name__}: {exc}"}

    # 3. Server cold start.
    print("[benchmark] section 3: server cold start...", file=sys.stderr)
    try:
        result["server_cold_start"] = _measure_server_cold_start(db_path)
    except Exception as exc:
        result["server_cold_start"] = {"skipped": True, "reason": f"{type(exc).__name__}: {exc}"}

    # 4 + 5. Latency distributions.
    print("[benchmark] sections 4-5: latency distributions...", file=sys.stderr)
    try:
        latencies = _measure_search_latency(db_path, n_queries=args.n_queries)
        result.update(latencies)
    except Exception as exc:
        result["semantic_search_latency_ms"] = {"skipped": True, "reason": f"{type(exc).__name__}: {exc}"}
        result["trace_dependencies_latency_ms"] = {"skipped": True, "reason": f"{type(exc).__name__}: {exc}"}

    # 6. Memory snapshots.
    print("[benchmark] section 6: memory snapshots...", file=sys.stderr)
    try:
        result["memory"] = _measure_memory(db_path)
    except Exception as exc:
        result["memory"] = {"skipped": True, "reason": f"{type(exc).__name__}: {exc}"}

    # Write result.
    sha = metadata["git_sha"]
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = output_dir / f"{sha}-{ts}.json"
    # Defensive re-mkdir: some test environments (or the tmpdir cleanup of an
    # interrupted prior run, plus a cwd-relative output path) can race with us.
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    print(f"[benchmark] wrote {out_path}", file=sys.stderr)

    if cleanup_db:
        try:
            shutil.rmtree(db_path.parent)
        except Exception:
            pass

    # Print the result path on stdout so callers can capture it.
    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
