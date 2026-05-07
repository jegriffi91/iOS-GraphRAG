"""Phase 3 P3.4 — `ios-graphrag-doctor` diagnostic CLI.

Default mode runs a short list of structured diagnoses suitable for a
support exchange. ``--bug-report`` adds Python/macOS/dependency-version
detail and a 50-line tail of the indexer / server logs, with home-dir
paths redacted to `~`. The output is intentionally copy-pasteable.

This script never imports torch/sentence-transformers eagerly — checks
that need them are wrapped in try/except so a broken env still yields
a useful report instead of an ImportError stack trace.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Output framework
# ---------------------------------------------------------------------------

# Status tokens. OK = green on a healthy system; everything else is something
# the support engineer wants to look at.
OK = "OK"
MISSING = "MISSING"
LEGACY = "LEGACY"
STALE = "STALE"
WARN = "WARN"
ERROR = "ERROR"


@dataclass
class Diagnosis:
    status: str
    check: str
    details: str = ""
    remediation: str = ""

    def render(self) -> str:
        # Fixed-width status block keeps the output aligned for grep/diff.
        return f"[{self.status:<8s}] {self.check}: {self.details}"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

DB_DEFAULT_NAME = "knowledge-graph.sqlite"
DB_USER_DEFAULT = Path.home() / ".cache" / "ios-graphrag" / "graph.sqlite"
HF_HUB_BASE = Path.home() / ".cache" / "huggingface" / "hub"
MODEL_HUB_DIR = HF_HUB_BASE / "models--nomic-ai--nomic-embed-text-v1.5"
# ``LOG_DIR`` is overridable via ``GRAPHRAG_LOG_DIR`` so tests don't pollute
# the user's Library directory. Resolved per call via :func:`_active_log_dir`.
DEFAULT_LOG_DIR = Path.home() / "Library" / "Logs" / "ios-graphrag"


def _active_log_dir() -> Path:
    override = os.environ.get("GRAPHRAG_LOG_DIR")
    return Path(override) if override else DEFAULT_LOG_DIR


# Backwards-compatible alias retained for tests / external readers that
# still reference the constant. Resolved at import time; tests that need
# a different value should call _active_log_dir() instead.
LOG_DIR = DEFAULT_LOG_DIR


def _resolve_db_path() -> Tuple[Path, str]:
    """Return (path, source) for the DB we should diagnose against.

    Source is either ``"GRAPH_DB_PATH"`` or ``"default"``.
    """
    env = os.environ.get("GRAPH_DB_PATH")
    if env:
        return Path(env).expanduser(), "GRAPH_DB_PATH"
    return DB_USER_DEFAULT, "default"


def _project_root() -> Path:
    """Best-effort project root, used to look for in-repo log files."""
    return Path(__file__).resolve().parents[2]


def _redact_home(text: str) -> str:
    """Replace `/Users/<name>/` with `~/` so bug reports don't leak the user.

    On macOS we additionally honour ``HOME``; falling back to a regex on
    `/Users/[^/]+/` catches most cases including paths that came from
    other users in shared logs (which we redact too — better safer)."""
    home = str(Path.home())
    if home and home in text:
        text = text.replace(home, "~")
    text = re.sub(r"/Users/[^/\s]+/", "~/", text)
    return text


# ---------------------------------------------------------------------------
# Individual diagnoses
# ---------------------------------------------------------------------------


def diagnose_db_exists() -> Diagnosis:
    """File-level existence/readability check on the DB the server would load."""
    db_path, source = _resolve_db_path()
    if not db_path.exists():
        return Diagnosis(
            status=MISSING,
            check=f"DB exists at {db_path} (source: {source})",
            details="file not found",
            remediation=(
                f"Run `ios-graphrag-index --repo /path/to/your/repo --db {db_path} --full`."
            ),
        )
    try:
        size_mb = db_path.stat().st_size / (1024 * 1024)
    except OSError as exc:
        return Diagnosis(
            status=ERROR,
            check=f"DB exists at {db_path} (source: {source})",
            details=f"stat failed: {exc}",
            remediation="Check filesystem permissions on the parent directory.",
        )
    if not os.access(db_path, os.R_OK):
        return Diagnosis(
            status=ERROR,
            check=f"DB exists at {db_path} (source: {source})",
            details=f"not readable ({size_mb:.1f} MB)",
            remediation=f"Run: chmod u+r {db_path}",
        )
    return Diagnosis(
        status=OK,
        check=f"DB exists at {db_path} (source: {source})",
        details=f"{size_mb:.1f} MB",
    )


def diagnose_schema_version() -> Diagnosis:
    """Look for the schema_version table; mark LEGACY if absent (Phase 6a not yet shipped)."""
    db_path, _ = _resolve_db_path()
    if not db_path.exists():
        return Diagnosis(
            status=MISSING,
            check="Schema version",
            details=f"no DB at {db_path}",
            remediation="Build the index first (see DB-exists diagnosis).",
        )
    try:
        conn = sqlite3.connect(str(db_path))
    except sqlite3.Error as exc:
        return Diagnosis(
            status=ERROR,
            check="Schema version",
            details=f"sqlite open failed: {exc}",
            remediation="DB may be corrupt; rebuild with --full.",
        )
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
        ).fetchone()
        if row is None:
            return Diagnosis(
                status=LEGACY,
                check="Schema version",
                details="schema_version table not present",
                remediation=(
                    "Schema versioning lands in Phase 6a; rebuild the index "
                    "after that ships. Today's DBs are forward-compatible."
                ),
            )
        ver_row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        ver = ver_row[0] if ver_row else None
        if ver is None:
            return Diagnosis(
                status=WARN,
                check="Schema version",
                details="schema_version table empty",
                remediation="Reapply migrations or rebuild the index.",
            )
        return Diagnosis(
            status=OK,
            check="Schema version",
            details=f"{ver}",
        )
    finally:
        conn.close()


def diagnose_model_cache() -> Diagnosis:
    """Inspect the HF hub cache layout for the embedding model."""
    if not MODEL_HUB_DIR.exists():
        return Diagnosis(
            status=MISSING,
            check="Embedding model cache",
            details=f"{MODEL_HUB_DIR} not found",
            remediation=(
                "Run `ios-graphrag-preflight` to download the model "
                "(or use huggingface-cli for an offline workflow)."
            ),
        )
    snapshots = MODEL_HUB_DIR / "snapshots"
    if not snapshots.exists():
        return Diagnosis(
            status=STALE,
            check="Embedding model cache",
            details=f"{snapshots} missing — partial download",
            remediation=f"Run: rm -rf {MODEL_HUB_DIR} && ios-graphrag-preflight",
        )
    snapshot_dirs = [d for d in snapshots.iterdir() if d.is_dir()]
    if not snapshot_dirs:
        return Diagnosis(
            status=STALE,
            check="Embedding model cache",
            details="no snapshot directories present",
            remediation=f"Run: rm -rf {MODEL_HUB_DIR} && ios-graphrag-preflight",
        )
    # Sentence-transformers expects either model.safetensors or pytorch_model.bin.
    snapshot = snapshot_dirs[0]
    has_weights = any(
        (snapshot / candidate).exists() for candidate in ("model.safetensors", "pytorch_model.bin")
    )
    if not has_weights:
        # The lock files in HF cache are written during in-flight downloads.
        # Their presence with no weight file usually means the download was
        # interrupted.
        lock_files = list(MODEL_HUB_DIR.rglob("*.lock"))
        return Diagnosis(
            status=STALE,
            check="Embedding model cache",
            details=(
                "snapshot has no model.safetensors or pytorch_model.bin "
                f"({len(lock_files)} lock file(s) present)"
            ),
            remediation=f"Run: rm -rf {MODEL_HUB_DIR} && ios-graphrag-preflight",
        )
    return Diagnosis(
        status=OK,
        check="Embedding model cache",
        details=f"snapshot at {snapshot.name}",
    )


def diagnose_graph_db_path_env() -> Diagnosis:
    """Validate the GRAPH_DB_PATH env var if it's set; otherwise note the default."""
    env = os.environ.get("GRAPH_DB_PATH")
    if not env:
        return Diagnosis(
            status=OK,
            check="GRAPH_DB_PATH env var",
            details=f"unset; falling back to {DB_USER_DEFAULT}",
        )
    expanded = Path(env).expanduser()
    if not expanded.exists():
        return Diagnosis(
            status=MISSING,
            check="GRAPH_DB_PATH env var",
            details=f"set to {env} but file does not exist",
            remediation=(
                "Either build the index at that path or unset GRAPH_DB_PATH to use the default."
            ),
        )
    if not os.access(expanded, os.R_OK):
        return Diagnosis(
            status=ERROR,
            check="GRAPH_DB_PATH env var",
            details=f"set to {expanded}, file present but not readable",
            remediation=f"Run: chmod u+r {expanded}",
        )
    return Diagnosis(
        status=OK,
        check="GRAPH_DB_PATH env var",
        details=f"-> {expanded}",
    )


def diagnose_tree_sitter_grammars() -> Diagnosis:
    """Both swift + objc grammars must expose .language()."""
    failures: List[str] = []
    try:
        import tree_sitter_swift  # noqa: PLC0415

        tree_sitter_swift.language()
    except Exception as exc:  # noqa: BLE001
        failures.append(f"tree_sitter_swift: {type(exc).__name__}: {exc}")
    try:
        import tree_sitter_objc  # noqa: PLC0415

        tree_sitter_objc.language()
    except Exception as exc:  # noqa: BLE001
        failures.append(f"tree_sitter_objc: {type(exc).__name__}: {exc}")

    if failures:
        return Diagnosis(
            status=ERROR,
            check="Tree-sitter grammars",
            details="; ".join(failures),
            remediation=(
                "Reinstall with `uv sync` or check pyproject.toml pins for "
                "tree-sitter-swift/tree-sitter-objc."
            ),
        )
    return Diagnosis(
        status=OK,
        check="Tree-sitter grammars",
        details="swift + objc loaded",
    )


def diagnose_torch_mps() -> Diagnosis:
    """torch import + MPS availability. Soft-fail on no-MPS (Intel/x86 Linux)."""
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return Diagnosis(
            status=ERROR,
            check="Torch + MPS",
            details=f"could not import torch: {type(exc).__name__}: {exc}",
            remediation="Reinstall torch with `uv sync`.",
        )
    has_mps = False
    try:
        has_mps = bool(torch.backends.mps.is_available())
    except Exception:  # noqa: BLE001
        has_mps = False
    if has_mps:
        return Diagnosis(
            status=OK,
            check="Torch + MPS",
            details=f"torch {torch.__version__}, MPS available",
        )
    return Diagnosis(
        status=WARN,
        check="Torch + MPS",
        details=f"torch {torch.__version__}, MPS NOT available",
        remediation="Embedding will fall back to CPU; expected on Intel/x86.",
    )


def diagnose_log_directory() -> Diagnosis:
    """Future log home (Phase 6b will formalize). Today: just check it's creatable."""
    if LOG_DIR.exists():
        if not os.access(LOG_DIR, os.W_OK):
            return Diagnosis(
                status=ERROR,
                check="Log directory",
                details=f"{LOG_DIR} exists but is not writable",
                remediation=f"Run: chmod u+w {LOG_DIR}",
            )
        return Diagnosis(
            status=OK,
            check="Log directory",
            details=f"{LOG_DIR}",
        )
    parent = LOG_DIR.parent
    if not parent.exists() or not os.access(parent, os.W_OK):
        return Diagnosis(
            status=WARN,
            check="Log directory",
            details=f"{LOG_DIR} not present and parent {parent} not writable",
            remediation=(
                "Phase 6b will formalize this path. Today the indexer / server "
                "use the cwd-relative log files, which still work."
            ),
        )
    return Diagnosis(
        status=OK,
        check="Log directory",
        details=f"{LOG_DIR} not present, but parent is writable (will be created)",
    )


# ---------------------------------------------------------------------------
# Log tailing
# ---------------------------------------------------------------------------

# Phase 6b: the canonical log directory is ``~/Library/Logs/ios-graphrag/``
# (or whatever ``GRAPHRAG_LOG_DIR`` overrides it to in tests). The two log
# basenames live in that directory; the legacy ``indexing_errors.log`` next
# to cwd is still tried as a fallback so doctor remains useful on a host
# that hasn't re-indexed since the Phase 6b cutover.
LOG_FILE_NAMES = ("indexer.log", "server.log")
LEGACY_LOG_FILE_NAMES = ("indexing_errors.log", "server.log")


def _candidate_log_paths() -> List[Path]:
    """Where the indexer / server might have written logs.

    Phase 6b path order: the canonical location first
    (``~/Library/Logs/ios-graphrag/`` or ``GRAPHRAG_LOG_DIR``), then
    legacy cwd / project-root fallbacks for hosts that haven't re-indexed
    since the cutover. First hit wins per filename.
    """
    candidates: List[Path] = []
    log_dir = _active_log_dir()
    for name in LOG_FILE_NAMES:
        p = log_dir / name
        if p not in candidates:
            candidates.append(p)
    cwd = Path.cwd()
    project = _project_root()
    for base in (cwd, project):
        for name in LEGACY_LOG_FILE_NAMES:
            p = base / name
            if p not in candidates:
                candidates.append(p)
    return candidates


_TAIL_ERROR_LEVELS = {"WARNING", "ERROR", "CRITICAL"}


def _format_json_log_line(line: str) -> Optional[str]:
    """Format one structured log line for ``--tail-errors`` output.

    Returns ``None`` if the line should be skipped (i.e. its level is not
    WARNING+). Returns the raw line (stripped) if the line isn't JSON --
    so the caller can fall back to displaying the original text. Returns
    a formatted string when the line parses as JSON and meets the level
    filter.
    """
    try:
        record = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        # Non-JSON line (legacy format / crash dump). Filter by substring.
        upper = line.upper()
        if any(level in upper for level in _TAIL_ERROR_LEVELS):
            return line.rstrip()
        return None

    if not isinstance(record, dict):
        return line.rstrip()

    level = str(record.get("level", "")).upper()
    if level not in _TAIL_ERROR_LEVELS:
        return None

    ts = record.get("ts", "")
    logger_name = record.get("logger", "")
    message = record.get("message", "")
    parts = [str(ts), f"{level:<8s}", str(logger_name) + ":", str(message)]
    trace_id = record.get("trace_id")
    if trace_id:
        parts.append(f"[trace_id={trace_id}]")
    return "  ".join(parts)


def _tail_error_lines(path: Path, n: int) -> List[str]:
    """Return the last ``n`` WARNING+/ERROR lines from a log file.

    Reads the entire file (logs are bounded by RotatingFileHandler at
    50 MB), filters by level via :func:`_format_json_log_line`, and
    returns the trailing window. Each returned string is already
    formatted for printing.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [f"<could not read {path}: {exc}>"]
    formatted: List[str] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        rendered = _format_json_log_line(raw)
        if rendered is not None:
            formatted.append(rendered)
    return formatted[-n:]


def render_tail_errors(n: int = 20) -> str:
    """Build the ``--tail-errors`` payload for ``ios-graphrag-doctor``.

    Reads each canonical log file (``indexer.log`` + ``server.log``) from
    the active log directory, prints the last ``n`` WARNING+/ERROR
    records per file. Missing files print a friendly stub so support
    requests on a fresh install still produce a useful diagnostic.
    """
    log_dir = _active_log_dir()
    out: List[str] = []
    for name in LOG_FILE_NAMES:
        out.append(f"=== Last {n} errors from {name} ===")
        path = log_dir / name
        if not path.exists():
            out.append(f"(no log file at {_redact_home(str(path))})")
            continue
        recent = _tail_error_lines(path, n)
        if not recent:
            out.append("(no WARNING/ERROR/CRITICAL records found)")
        else:
            out.extend(_redact_home(line) for line in recent)
    return "\n".join(out)


def _tail_lines(path: Path, n: int) -> List[str]:
    """Return the last ``n`` newline-separated lines of ``path``.

    Cheap implementation: read everything. The log files are bounded by
    the indexer's per-run output (KB to single-digit MB at worst); a
    streaming read is overkill.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError) as exc:
        return [f"<could not read {path}: {exc}>"]
    lines = text.splitlines()
    return lines[-n:]


def collect_recent_errors(n: int) -> List[Tuple[Path, List[str]]]:
    """Return [(log_path, last_n_error_lines)] for each known log file we found.

    "Error lines" = lines whose level is WARNING or above (filters
    INFO / DEBUG noise in default-mode output).
    """
    seen: set[Path] = set()
    out: List[Tuple[Path, List[str]]] = []
    level_re = re.compile(r"\b(WARNING|ERROR|CRITICAL)\b")
    for path in _candidate_log_paths():
        if path in seen:
            continue
        seen.add(path)
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            out.append((path, [f"<could not read: {exc}>"]))
            continue
        candidates = [ln for ln in text.splitlines() if level_re.search(ln)]
        out.append((path, candidates[-n:]))
    return out


# ---------------------------------------------------------------------------
# Bug-report extras
# ---------------------------------------------------------------------------


_BUG_REPORT_DEPS = (
    "mcp",
    "tree-sitter",
    "tree-sitter-swift",
    "tree-sitter-objc",
    "networkx",
    "numpy",
    "sentence-transformers",
    "tqdm",
    "einops",
    "torch",
)


def _git_sha_short() -> str:
    """Short SHA of the worktree at this script's project root, or 'unknown'."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_project_root()),
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if sha.returncode == 0:
            return sha.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _dep_versions() -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for dep in _BUG_REPORT_DEPS:
        try:
            out.append((dep, importlib.metadata.version(dep)))
        except importlib.metadata.PackageNotFoundError:
            out.append((dep, "(not installed)"))
    return out


def _relevant_env_vars() -> List[Tuple[str, str]]:
    """Env vars whose names start with GRAPHRAG_/HF_ or contain SSL/CA_BUNDLE.

    Values are redacted to replace the user's home with ``~`` so bug
    reports don't leak the username.
    """
    interesting: List[Tuple[str, str]] = []
    for key, raw in os.environ.items():
        if (
            key.startswith("GRAPHRAG_")
            or key.startswith("HF_")
            or "SSL" in key
            or "CA_BUNDLE" in key
        ):
            value = _redact_home(raw) if raw else raw
            interesting.append((key, value))
    interesting.sort()
    return interesting


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def run_default_diagnoses() -> List[Diagnosis]:
    return [
        diagnose_db_exists(),
        diagnose_schema_version(),
        diagnose_model_cache(),
        diagnose_graph_db_path_env(),
        diagnose_tree_sitter_grammars(),
        diagnose_torch_mps(),
        diagnose_log_directory(),
    ]


def render_default_report(diagnoses: Iterable[Diagnosis]) -> str:
    sha = _git_sha_short()
    lines: List[str] = []
    lines.append(f"=== ios-graphrag-doctor (build: {sha}) ===")
    for d in diagnoses:
        lines.append(d.render())
        if d.status != OK and d.remediation:
            lines.append(f"           -> {d.remediation}")
    return "\n".join(lines)


def render_recent_errors(n: int = 5) -> str:
    lines = ["", f"=== Last {n} errors ==="]
    found_any = False
    for path, recent in collect_recent_errors(n):
        found_any = True
        lines.append(f"-- {_redact_home(str(path))}")
        if not recent:
            lines.append("   (no WARNING/ERROR/CRITICAL lines found)")
        else:
            lines.extend(f"   {_redact_home(line)}" for line in recent)
    if not found_any:
        lines.append("(no log files found at any candidate path)")
    return "\n".join(lines)


def render_bug_report_extras() -> str:
    py = sys.version.split()[0]
    plat_release, _, _ = platform.mac_ver()
    plat_release = plat_release or platform.platform()

    deps = _dep_versions()
    env = _relevant_env_vars()

    lines: List[str] = []
    lines.append("")
    lines.append("=== Environment ===")
    lines.append(f"Python:    {py}")
    lines.append(f"Executable:{_redact_home(sys.executable)}")
    lines.append(f"Platform:  {platform.platform()}")
    lines.append(f"macOS:     {plat_release}")
    lines.append("")
    lines.append("=== Dependency versions ===")
    for name, ver in deps:
        lines.append(f"{name:<25s} {ver}")
    lines.append("")
    lines.append("=== Relevant env vars ===")
    if not env:
        lines.append("(none set)")
    else:
        for key, val in env:
            lines.append(f"{key} = {val}")
    lines.append("")
    lines.append("=== Last 50 lines: indexing_errors.log + server.log ===")
    for path in _candidate_log_paths():
        if not path.exists():
            continue
        lines.append(f"-- {_redact_home(str(path))}")
        for ln in _tail_lines(path, 50):
            lines.append(f"   {_redact_home(ln)}")
    return "\n".join(lines)


def render_integrity_report(result: dict) -> str:
    """Format the dict from :func:`_integrity.run_integrity_checks` for humans.

    Output format (Phase 6c)::

        === Integrity verification ===
        Row counts:
          nodes:        12345
          edges:        67890
          file_hashes:    234

        [OK]      No orphan edges (12000 resolved, 0 dangling).
        [WARNING] 3 file paths in nodes are no longer on disk:
          /path/to/Old.swift
          ...
        [INFO]    18 symbol names resolve to multiple nodes ...

    Returned as a single string so the caller can decide whether to
    print, return, or pipe it into another renderer.
    """
    counts = result.get("row_counts", {}) or {}
    orphans = int(result.get("orphan_edges", 0) or 0)
    resolved = int(result.get("resolved_edges", 0) or 0)
    missing = list(result.get("missing_files") or [])
    collisions = int(result.get("symbol_collisions", 0) or 0)

    lines: List[str] = ["", "=== Integrity verification ==="]
    lines.append("Row counts:")
    lines.append(f"  nodes:       {counts.get('nodes', 0):>8d}")
    lines.append(f"  edges:       {counts.get('edges', 0):>8d}")
    lines.append(f"  file_hashes: {counts.get('file_hashes', 0):>8d}")
    lines.append("")

    if orphans == 0:
        lines.append(f"[OK]       No orphan edges ({resolved} resolved, 0 dangling).")
    else:
        lines.append(
            f"[WARNING]  {orphans} orphan edge(s) detected "
            f"({resolved} resolved). target_node_id points at a "
            "non-existent node."
        )

    if not missing:
        lines.append("[OK]       All node file_path entries exist on disk.")
    else:
        lines.append(f"[WARNING]  {len(missing)} file path(s) in nodes are no longer on disk:")
        for path in missing[:10]:
            lines.append(f"  {_redact_home(path)}")
        if len(missing) > 10:
            lines.append(f"  ... ({len(missing) - 10} more)")

    lines.append(
        f"[INFO]     {collisions} symbol name(s) resolve to multiple nodes "
        "(collisions logged in indexing_errors.log)."
    )
    return "\n".join(lines)


def run_verify(argv_db: Optional[str] = None) -> int:
    """Run the Phase 6c integrity checks against the configured DB.

    Returns ``1`` if any WARNING-class issue is found (orphan edges or
    missing file paths), ``0`` otherwise. Output goes to stdout so the
    caller can pipe / capture it. ``argv_db`` is unused today but kept
    in the signature so a future ``--db`` flag can land without
    changing the call site.
    """
    db_path, source = _resolve_db_path()
    if not db_path.exists():
        print(
            f"[ERROR]    DB not found at {db_path} (source: {source}). "
            "Build the index first with `ios-graphrag-index`."
        )
        return 1

    # Local import: keeps the doctor's default-mode start-up cheap and
    # avoids a hard dep cycle if _integrity ever needs anything from
    # doctor (it doesn't today, but defensively isolated).
    from . import _integrity  # noqa: PLC0415

    try:
        conn = sqlite3.connect(str(db_path))
    except sqlite3.Error as exc:
        print(f"[ERROR]    sqlite open failed: {exc}")
        return 1
    try:
        result = _integrity.run_integrity_checks(conn)
    finally:
        conn.close()

    print(render_integrity_report(result))

    has_warning = int(result.get("orphan_edges", 0) or 0) > 0 or bool(result.get("missing_files"))
    return 1 if has_warning else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ios-graphrag-doctor",
        description=(
            "Diagnose iOS-GraphRAG installation health. Default mode prints "
            "one structured line per check; --bug-report appends Python/macOS/"
            "dependency-version detail and a redacted log tail suitable for an "
            "issue ticket; --verify runs the Phase 6c integrity checks "
            "(row counts, orphan edges, missing-file detection) and exits 1 "
            "on any WARNING-class issue; --tail-errors N prints the last N "
            "WARNING+/ERROR records from indexer.log and server.log."
        ),
    )
    parser.add_argument(
        "--bug-report",
        action="store_true",
        help="Emit a longer, copy-pasteable report (env, deps, log tail) for issue tickets.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Run Phase 6c integrity checks (row counts, orphan edges, missing "
            "file paths, symbol collisions) and exit 1 on any WARNING-class "
            "issue. Skips the default health diagnoses."
        ),
    )
    parser.add_argument(
        "--tail-errors",
        type=int,
        nargs="?",
        const=20,
        default=None,
        metavar="N",
        help=(
            "Print the last N WARNING+/ERROR records from indexer.log and "
            "server.log under ~/Library/Logs/ios-graphrag/ (or "
            "$GRAPHRAG_LOG_DIR). Default N=20. Skips the default health "
            "diagnoses."
        ),
    )
    args = parser.parse_args(argv)

    if args.verify:
        return run_verify()

    if args.tail_errors is not None:
        print(render_tail_errors(n=args.tail_errors))
        return 0

    diagnoses = run_default_diagnoses()
    print(render_default_report(diagnoses))
    print(render_recent_errors(n=5))
    if args.bug_report:
        print(render_bug_report_extras())

    has_error = any(d.status == ERROR for d in diagnoses)
    return 1 if has_error else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
