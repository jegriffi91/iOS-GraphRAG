"""Phase 4.5a parse-error audit for iOS-GraphRAG.

Walks a repository, parses every ``.swift`` / ``.h`` / ``.m`` file with the
SAME tree-sitter setup the indexer uses (``src/ios_graphrag/indexer.py``
:func:`parse_file`), and records per-file ``has_error`` plus counts of
``ERROR`` / ``MISSING`` nodes. Aggregates into summary stats + a Strategy
A/B/C decision recommendation per the roadmap's Phase 4.5a decision matrix:

- ``< 5%``  files with any parse error  -> Strategy A (tree-sitter primary,
  hardened fallback).
- ``5-20%``                              -> Strategy A + B (tree-sitter +
  SourceKitten validator in CI).
- ``> 20%``                              -> Strategy C (consider
  SourceKit-LSP as primary).

Usage::

    uv run python tools/parse_audit.py --repo /path/to/repo --output benchmarks/results/
    uv run python tools/parse_audit.py --repo /path/to/repo --extensions .swift,.h,.m

Output: ``benchmarks/results/parse-audit-<orchestrator-sha>-<YYYYMMDD-HHMMSS>.json``
plus a human-readable console summary.

The tool deliberately does NOT use the indexer's ``should_index`` symbol
filter (which excludes ``*Tests*``, ``Mock*``, ``*.generated.swift``, etc.)
because the audit is measuring what the *parser* does on the actual file
population — filtering tests/mocks would understate exposure to
SwiftUI/macro patterns that often live in test or generated files. We DO
mirror the indexer's ``BLOCK_DIRS`` (``.build``, ``Pods``, ``Carthage``,
``DerivedData``, ``vendor``) plus the common-case skip list from the
roadmap brief (``.git``, ``node_modules``).
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata as _metadata
import json
import multiprocessing
import os
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from tree_sitter import Language, Parser
import tree_sitter_objc
import tree_sitter_swift


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Mirror indexer.BLOCK_DIRS + add common-case ignores. The audit walks the
# directory tree itself rather than reusing ``collect_indexable_files``
# because the indexer's discovery applies should_index() name filters that
# would distort the audit (see module docstring).
SKIP_DIRS = {
    # From src/ios_graphrag/indexer.py BLOCK_DIRS
    ".build",
    "Pods",
    "Carthage",
    "DerivedData",
    "vendor",
    # Common-case ignores per Phase 4.5a brief
    ".git",
    "node_modules",
}

DEFAULT_EXTENSIONS = (".swift", ".h", ".m")

# Per-file parse timeout. Tree-sitter is ordinarily very fast (<100ms even
# on multi-MB Swift files), so 30s is a generous ceiling that catches
# pathological grammar regressions without giving up on legitimate large
# files. Files exceeding this are recorded with ``timeout: true`` and do
# NOT contribute to the error-rate denominator's "errored" count, but
# DO count toward total_files (see roadmap: timeouts are themselves a
# parser-strategy signal).
PARSE_TIMEOUT_SECONDS = 30


@dataclass
class FileResult:
    """Per-file audit record. ``timeout=True`` means the parse exceeded
    ``PARSE_TIMEOUT_SECONDS`` and the other counts are unfilled."""

    path: str
    extension: str
    size_bytes: int
    has_error: bool = False
    error_node_count: int = 0
    missing_node_count: int = 0
    total_nodes: int = 0
    timeout: bool = False
    skipped_reason: Optional[str] = None  # set when we couldn't parse (e.g. unreadable)


# ---------------------------------------------------------------------------
# Parser setup — mirrors src/ios_graphrag/indexer.py::parse_file
# ---------------------------------------------------------------------------
def _build_parser_for(path: str) -> Parser:
    """Construct a tree-sitter Parser for the given file's language.

    Mirrors ``parse_file`` in ``src/ios_graphrag/indexer.py`` so the audit
    measures the SAME parser the indexer would use. We don't import that
    function because it does symbol extraction + DB I/O — the audit only
    needs the parse tree.
    """
    parser = Parser()
    if path.endswith(".swift"):
        parser.language = Language(tree_sitter_swift.language())
    else:
        parser.language = Language(tree_sitter_objc.language())
    return parser


def _read_bytes(path: Path) -> Tuple[Optional[bytes], Optional[str]]:
    """Read file as bytes. Returns ``(data, None)`` on success or
    ``(None, reason)`` on failure. Handles UTF-8 BOM and falls back to
    latin-1 for decode-validation purposes — tree-sitter parses bytes
    directly, so we just need to confirm the bytes are readable."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        return None, f"read_error: {exc}"
    # Strip a UTF-8 BOM if present so the parser sees the canonical bytes.
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    # A quick decode-probe: tree-sitter doesn't need text, but if neither
    # utf-8 nor latin-1 work the file is likely binary garbage we should
    # skip rather than feed to the parser.
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            data.decode("latin-1")
        except UnicodeDecodeError:
            return None, "decode_error"
    return data, None


def _walk_tree_for_errors(root_node) -> Tuple[int, int, int]:
    """Recursive walk: returns (error_node_count, missing_node_count, total_nodes).

    - ``error_node_count`` counts nodes whose ``type == 'ERROR'`` (the
      tree-sitter classification for unrecoverable parses) OR whose
      ``is_error`` flag is set (newer API — tree-sitter sometimes flags
      a typed node as errored without renaming it to ``ERROR``).
    - ``missing_node_count`` counts nodes with ``is_missing == True`` —
      these are tokens the parser inserted to recover. Both are
      independent error indicators per the tree-sitter docs.
    """
    error_count = 0
    missing_count = 0
    total = 0
    stack = [root_node]
    while stack:
        node = stack.pop()
        total += 1
        if node.type == "ERROR" or getattr(node, "is_error", False):
            error_count += 1
        if getattr(node, "is_missing", False):
            missing_count += 1
        # Iterate via .children (same access pattern indexer uses).
        for child in node.children:
            stack.append(child)
    return error_count, missing_count, total


def _audit_one_file(path_str: str) -> FileResult:
    """Parse a single file and return its audit record. Run in a worker
    process so a single pathological file can't take down the whole audit.

    Note: tree-sitter holds the GIL while parsing, so a process pool gives
    real parallelism; with a thread pool we'd serialize on the GIL.
    """
    path = Path(path_str)
    ext = path.suffix.lower()
    try:
        size_bytes = path.stat().st_size
    except OSError:
        size_bytes = 0
    result = FileResult(path=path_str, extension=ext, size_bytes=size_bytes)

    data, read_err = _read_bytes(path)
    if data is None:
        result.skipped_reason = read_err
        return result

    try:
        parser = _build_parser_for(path_str)
        tree = parser.parse(data)
    except Exception as exc:
        result.skipped_reason = f"parse_exception: {type(exc).__name__}: {exc}"
        return result

    result.has_error = bool(tree.root_node.has_error)
    err, missing, total = _walk_tree_for_errors(tree.root_node)
    result.error_node_count = err
    result.missing_node_count = missing
    result.total_nodes = total
    return result


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def discover_files(repo: Path, extensions: Sequence[str]) -> List[Path]:
    """Walk ``repo`` collecting files matching ``extensions``.

    Skips any directory whose basename is in :data:`SKIP_DIRS`. We don't
    parse ``.gitignore`` — the brief explicitly says "be reasonable;
    you don't need a full gitignore parser, just skip the common-cases
    listed". Adding a gitignore parser later would be a small extension
    if Phase 4.5a finds it necessary.
    """
    repo = repo.resolve()
    ext_set = {e if e.startswith(".") else "." + e for e in extensions}
    found: List[Path] = []
    for root, dirs, files in os.walk(repo, followlinks=False):
        # Mutate dirs in-place to prune the walk (os.walk semantics).
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            p = Path(root) / name
            if p.suffix.lower() in ext_set:
                found.append(p)
    found.sort()  # deterministic ordering for reproducibility
    return found


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _bin_error_count(n: int) -> str:
    if n == 0:
        return "0"
    if n <= 5:
        return "1-5"
    if n <= 20:
        return "6-20"
    if n <= 100:
        return "21-100"
    return "100+"


def _decision_for(percent_with_any_error: float) -> str:
    """Roadmap Phase 4.5a decision matrix.

    Boundary handling: ``< 5%`` is Strategy A; ``[5%, 20%)`` is A+B;
    ``>= 20%`` is C. The roadmap text uses ``5-20%`` which is ambiguous
    at the upper bound — we pick the conservative interpretation that
    20.0% triggers Strategy C (the 5-20% band is "<20%").
    """
    if percent_with_any_error < 5.0:
        return "Strategy A (tree-sitter primary, hardened fallback)"
    if percent_with_any_error < 20.0:
        return "Strategy A + B (tree-sitter primary, SourceKitten validator in CI)"
    return "Strategy C (consider SourceKit-LSP as primary)"


def aggregate(results: List[FileResult], top_n: int = 20) -> Dict:
    total_files = len(results)
    files_with_has_error = sum(1 for r in results if r.has_error and not r.timeout)
    # "Any error node" = root.has_error OR any ERROR node OR any MISSING node.
    # has_error is the canonical boolean per the tree-sitter docs, but a
    # tree can have MISSING-recovery nodes without root.has_error tripping
    # in older grammars; counting both is the safer measure.
    files_with_any_error_node = sum(
        1
        for r in results
        if not r.timeout and (r.has_error or r.error_node_count > 0 or r.missing_node_count > 0)
    )
    timeouts = sum(1 for r in results if r.timeout)
    skipped = sum(1 for r in results if r.skipped_reason)

    pct_has = (100.0 * files_with_has_error / total_files) if total_files else 0.0
    pct_any = (100.0 * files_with_any_error_node / total_files) if total_files else 0.0

    histogram: Dict[str, int] = {"0": 0, "1-5": 0, "6-20": 0, "21-100": 0, "100+": 0}
    for r in results:
        if r.timeout or r.skipped_reason:
            continue
        histogram[_bin_error_count(r.error_node_count)] += 1

    # Per-extension breakdown.
    by_ext: Dict[str, Dict] = {}
    for r in results:
        bucket = by_ext.setdefault(r.extension, {"total": 0, "files_with_error": 0})
        bucket["total"] += 1
        if not r.timeout and (r.has_error or r.error_node_count > 0 or r.missing_node_count > 0):
            bucket["files_with_error"] += 1
    for ext, bucket in by_ext.items():
        bucket["percent"] = (
            round(100.0 * bucket["files_with_error"] / bucket["total"], 2)
            if bucket["total"]
            else 0.0
        )

    # Top-N error-prone files: sort by error_node_count desc, then size desc.
    errored = [
        r for r in results
        if not r.timeout and (r.error_node_count > 0 or r.has_error or r.missing_node_count > 0)
    ]
    errored.sort(key=lambda r: (r.error_node_count, r.size_bytes), reverse=True)
    top_files = [
        {
            "path": r.path,
            "error_node_count": r.error_node_count,
            "missing_node_count": r.missing_node_count,
            "total_nodes": r.total_nodes,
            "has_error": r.has_error,
            "size_bytes": r.size_bytes,
        }
        for r in errored[:top_n]
    ]

    return {
        "total_files": total_files,
        "files_with_has_error": files_with_has_error,
        "files_with_has_error_percent": round(pct_has, 2),
        "files_with_any_error_node": files_with_any_error_node,
        "files_with_any_error_node_percent": round(pct_any, 2),
        "histogram_error_node_counts": histogram,
        "by_extension": by_ext,
        "timeouts": timeouts,
        "skipped": skipped,
        "decision_recommendation": _decision_for(pct_any),
        "top_files": top_files,
    }


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------
def _orchestrator_sha() -> str:
    """Short SHA of THIS codebase (the iOS-GraphRAG repo) at HEAD.

    The audit JSON pins this so a result can be tied to the exact
    indexer/grammar version that produced it. Returns ``"unknown"`` if
    git isn't available (e.g. running from a tarball).
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _pkg_version(name: str) -> str:
    try:
        return _metadata.version(name)
    except _metadata.PackageNotFoundError:
        return "unknown"


def _platform_info() -> Dict[str, str]:
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "release": platform.release(),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_audit(
    repo: Path,
    extensions: Sequence[str],
    output_dir: Path,
    top_n: int = 20,
    workers: Optional[int] = None,
) -> Path:
    """Run the audit. Returns the path to the JSON file written.

    ``workers`` defaults to ``os.cpu_count()``; the caller can pin it for
    deterministic test runs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    files = discover_files(repo, extensions)

    results: List[FileResult] = []
    if files:
        # 'spawn' for parity with the indexer's embedding worker. Tree-sitter
        # doesn't strictly need spawn, but spawn-pools have a clean import
        # state and avoid forking a parent that may hold large allocations
        # (tests), so we use it consistently.
        ctx = multiprocessing.get_context("spawn")
        n_workers = workers or os.cpu_count() or 1
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
            futures = {executor.submit(_audit_one_file, str(p)): p for p in files}
            for future in as_completed(futures):
                p = futures[future]
                try:
                    result = future.result(timeout=PARSE_TIMEOUT_SECONDS)
                    results.append(result)
                except FutureTimeoutError:
                    # Build a skeleton FileResult so the file still appears
                    # in totals — exclusion would understate audit coverage.
                    try:
                        size_bytes = p.stat().st_size
                    except OSError:
                        size_bytes = 0
                    results.append(
                        FileResult(
                            path=str(p),
                            extension=p.suffix.lower(),
                            size_bytes=size_bytes,
                            timeout=True,
                        )
                    )
                except Exception as exc:
                    # Worker raised; record as skipped with the exception
                    # message. We don't re-raise — a single-file crash
                    # shouldn't kill the whole audit.
                    try:
                        size_bytes = p.stat().st_size
                    except OSError:
                        size_bytes = 0
                    results.append(
                        FileResult(
                            path=str(p),
                            extension=p.suffix.lower(),
                            size_bytes=size_bytes,
                            skipped_reason=f"worker_error: {type(exc).__name__}: {exc}",
                        )
                    )

    summary = aggregate(results, top_n=top_n)

    metadata = {
        "git_sha": _orchestrator_sha(),
        "timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repo_path": str(repo.resolve()),
        "tree_sitter_version": _pkg_version("tree-sitter"),
        "tree_sitter_swift_version": _pkg_version("tree-sitter-swift"),
        "tree_sitter_objc_version": _pkg_version("tree-sitter-objc"),
        "extensions_audited": list(extensions),
        "platform": _platform_info(),
        "python_version": platform.python_version(),
        "skip_dirs": sorted(SKIP_DIRS),
    }

    payload = {
        "metadata": metadata,
        "summary": {k: v for k, v in summary.items() if k != "top_files"},
        "top_files": summary["top_files"],
        # Per-file results are written in full so downstream tools can do
        # their own slicing without re-running the audit. For a 1M-LOC
        # repo this list is bounded by file count (~tens of thousands of
        # rows), still well under JSON size concerns.
        "files": [asdict(r) for r in results],
    }

    sha = metadata["git_sha"]
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = output_dir / f"parse-audit-{sha}-{ts}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return out_path


def _print_console_summary(repo: Path, summary: Dict, json_path: Path) -> None:
    """Human-readable summary to stderr (so callers piping JSON aren't
    polluted, and so logging doesn't fight stdout)."""
    print("=== Parse-error audit ===", file=sys.stderr)
    print(f"  Repo: {repo}", file=sys.stderr)
    print(f"  Files audited: {summary['total_files']}", file=sys.stderr)
    print(
        f"  Files with any parse error: {summary['files_with_any_error_node']} "
        f"({summary['files_with_any_error_node_percent']}%)",
        file=sys.stderr,
    )
    print(f"  Decision: {summary['decision_recommendation']}", file=sys.stderr)
    print("", file=sys.stderr)
    print("  By extension:", file=sys.stderr)
    for ext, bucket in sorted(summary["by_extension"].items()):
        print(
            f"    {ext:<7} {bucket['percent']}% error rate "
            f"({bucket['files_with_error']}/{bucket['total']})",
            file=sys.stderr,
        )
    print("", file=sys.stderr)
    if summary["top_files"]:
        n = min(5, len(summary["top_files"]))
        print(f"  Top {n} error-prone files:", file=sys.stderr)
        for entry in summary["top_files"][:5]:
            # Show the file path relative to repo if possible (cleaner output).
            try:
                rel = str(Path(entry["path"]).resolve().relative_to(repo.resolve()))
            except ValueError:
                rel = entry["path"]
            print(
                f"    {rel:<40s} {entry['error_node_count']:>4d} ERROR nodes",
                file=sys.stderr,
            )
        print("", file=sys.stderr)
    print(f"  Full JSON: {json_path}", file=sys.stderr)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="parse_audit",
        description="Phase 4.5a parse-error audit for iOS-GraphRAG.",
    )
    p.add_argument("--repo", required=True, help="Repository root to audit.")
    p.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "benchmarks" / "results"),
        help="Directory to write the audit JSON (default: benchmarks/results/).",
    )
    p.add_argument(
        "--extensions",
        default=",".join(DEFAULT_EXTENSIONS),
        help=f"Comma-separated file extensions to audit (default: {','.join(DEFAULT_EXTENSIONS)}).",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="How many error-prone files to include in top_files (default: 20).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker process count (default: os.cpu_count()).",
    )
    args = p.parse_args(argv)

    repo = Path(args.repo).expanduser().resolve()
    if not repo.is_dir():
        print(f"error: --repo {repo} is not a directory", file=sys.stderr)
        return 2
    extensions = tuple(e.strip() for e in args.extensions.split(",") if e.strip())
    if not extensions:
        print("error: --extensions produced an empty list", file=sys.stderr)
        return 2

    output_dir = Path(args.output).expanduser().resolve()
    json_path = run_audit(
        repo=repo,
        extensions=extensions,
        output_dir=output_dir,
        top_n=args.top_n,
        workers=args.workers,
    )
    # Re-load just the summary section to print without re-aggregating.
    with open(json_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    summary = dict(payload["summary"])
    summary["top_files"] = payload["top_files"]
    summary["by_extension"] = payload["summary"]["by_extension"]
    _print_console_summary(repo, summary, json_path)
    # Exit 0 regardless of error rate — the audit reporting "20% files
    # have errors" is not a failure of the audit, it's the audit working
    # correctly (per the brief).
    return 0


if __name__ == "__main__":
    sys.exit(main())
