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
import os
import platform
import re
import sqlite3
import subprocess
import sys
import textwrap
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
LOG_DIR = Path.home() / "Library" / "Logs" / "ios-graphrag"


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
        ver_row = conn.execute(
            "SELECT MAX(version) FROM schema_version"
        ).fetchone()
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
        (snapshot / candidate).exists()
        for candidate in ("model.safetensors", "pytorch_model.bin")
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
                "Either build the index at that path or unset GRAPH_DB_PATH "
                "to use the default."
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

LOG_FILE_NAMES = ("indexing_errors.log", "server.log")


def _candidate_log_paths() -> List[Path]:
    """Where the indexer / server might have written logs.

    The current code still uses cwd-relative log files; in CI / repo dev
    they show up at the project root. We try the project root and the
    cwd both — first hit wins per filename.
    """
    candidates: List[Path] = []
    cwd = Path.cwd()
    project = _project_root()
    for base in (cwd, project):
        for name in LOG_FILE_NAMES:
            p = base / name
            if p not in candidates:
                candidates.append(p)
    return candidates


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


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ios-graphrag-doctor",
        description=(
            "Diagnose iOS-GraphRAG installation health. Default mode prints "
            "one structured line per check; --bug-report appends Python/macOS/"
            "dependency-version detail and a redacted log tail suitable for an "
            "issue ticket."
        ),
    )
    parser.add_argument(
        "--bug-report",
        action="store_true",
        help="Emit a longer, copy-pasteable report (env, deps, log tail) for issue tickets.",
    )
    args = parser.parse_args(argv)

    diagnoses = run_default_diagnoses()
    print(render_default_report(diagnoses))
    print(render_recent_errors(n=5))
    if args.bug_report:
        print(render_bug_report_extras())

    has_error = any(d.status == ERROR for d in diagnoses)
    return 1 if has_error else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
