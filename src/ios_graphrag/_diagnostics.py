"""Phase 6d crash diagnostics: recent-call ring buffer + crashdump hook.

This module packages two pieces that together turn a server crash into an
actionable bug report:

1. A bounded, thread-safe deque of the last N MCP tool calls. The
   ``_traced_handler`` decorator in :mod:`server` records each call
   (success or error) into this buffer so a post-crash dump has the most
   recent activity available.
2. An ``sys.excepthook`` that, on any *uncaught* exception, writes a
   human-readable crashdump under ``get_log_dir()/crashdump-<UTC>-<pid>.txt``
   containing the traceback, process info, package versions, redacted env
   vars, and the buffer's contents. The hook chains into the previously
   installed excepthook so Python's default behavior (printing the
   traceback to stderr) still runs.

Crashdumps are intentionally separate from the regular
:mod:`_logging`-driven structured logs. They're meant to be
copy-pasteable into a GitHub issue, so the format is plain text rather
than JSON. The structured logs already record per-tool success/error
events with trace IDs (Phase 6b); this module adds the missing
"unhandled-exception" branch.

Handled errors (raised inside a tool handler and caught by the
``_traced_handler`` decorator) do NOT trigger a crashdump — those are
already observable via ``ios-graphrag-doctor --tail-errors``. The
crashdump is reserved for excepthook-level failures (startup errors,
thread-level crashes) that the decorator can't see.
"""
from __future__ import annotations

import collections
import importlib.metadata
import logging
import os
import platform
import re
import sys
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, Deque, Dict, List, Optional, Tuple, Type

# ---------------------------------------------------------------------------
# Ring buffer of recent tool calls
# ---------------------------------------------------------------------------

# Default max entries; can be overridden via GRAPHRAG_RECENT_CALLS_LIMIT for
# shops that want a longer post-crash audit trail. Resolved at import time
# rather than per-call to keep _record_call cheap.
DEFAULT_RECENT_CALLS_LIMIT = 50


def _resolve_limit() -> int:
    """Read the ring-buffer cap from the environment, with a safe default.

    Bad values (non-int, <=0) silently fall back to the default — the ring
    buffer is best-effort instrumentation; failing to start the server
    because someone fat-fingered an env var would be worse than ignoring it.
    """
    raw = os.environ.get("GRAPHRAG_RECENT_CALLS_LIMIT")
    if raw is None:
        return DEFAULT_RECENT_CALLS_LIMIT
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_RECENT_CALLS_LIMIT
    if value <= 0:
        return DEFAULT_RECENT_CALLS_LIMIT
    return value


_recent_calls: Deque[Dict[str, Any]] = collections.deque(maxlen=_resolve_limit())
_recent_calls_lock = threading.Lock()


def _reset_recent_calls(limit: Optional[int] = None) -> None:
    """Test-only helper: replace the buffer (e.g., to honour an env-var change).

    Production callers don't need this — :func:`_record_call` already
    handles the steady state. Tests that flip ``GRAPHRAG_RECENT_CALLS_LIMIT``
    mid-run use this to rebuild the deque under the new cap.
    """
    global _recent_calls
    new_limit = limit if limit is not None else _resolve_limit()
    with _recent_calls_lock:
        _recent_calls = collections.deque(maxlen=new_limit)


def _record_call(
    trace_id: str,
    tool: str,
    tool_args: Dict[str, Any],
    status: str,
    duration_ms: float,
    error: Optional[str] = None,
) -> None:
    """Append a (timestamped) tool-call record to the bounded ring buffer.

    Called from :func:`server._traced_handler` on both success and
    failure paths. The buffer is bounded by ``maxlen`` (no manual
    eviction needed) and protected by a single lock. Latency budget:
    one lock acquisition + one ``deque.append`` -- both O(1).

    ``status`` is the literal "ok" or "error". ``error`` carries the
    exception's ``str()`` representation for the error case; omitted
    for successes.
    """
    entry: Dict[str, Any] = {
        "trace_id": trace_id,
        "tool": tool,
        "tool_args": tool_args,
        "status": status,
        "duration_ms": duration_ms,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    if error is not None:
        entry["error"] = error
    with _recent_calls_lock:
        _recent_calls.append(entry)


def _snapshot_recent_calls() -> List[Dict[str, Any]]:
    """Return a list copy of the current ring-buffer contents.

    A copy (not a reference) is critical: callers iterate over this for
    reporting, and the buffer mutates concurrently from the handler
    threads. The lock protects the snapshot operation; the returned
    list is then safe to walk without further locking.
    """
    with _recent_calls_lock:
        return list(_recent_calls)


# ---------------------------------------------------------------------------
# Crashdump hook
# ---------------------------------------------------------------------------

# Same package set the doctor's --bug-report tracks. Duplicated rather than
# imported because doctor is CLI-only and this module is import-stable; we
# don't want a hard dependency from server.py onto doctor's argument parser.
_CRASHDUMP_DEPS: Tuple[str, ...] = (
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


def _redact_home(text: str) -> str:
    """Replace ``/Users/<name>/`` with ``~/`` so dumps don't leak the user.

    Intentional duplicate of :func:`doctor._redact_home` -- crashdumps
    must work even if doctor's import chain is broken (the very thing
    we're crashing in). The ``HOME`` env var takes priority to handle
    macOS's mixed case where the runtime user differs from the path
    macro; the regex catches paths from other users (e.g., shared logs).
    """
    home = os.environ.get("HOME") or str(Path.home())
    if home and home in text:
        text = text.replace(home, "~")
    text = re.sub(r"/Users/[^/\s]+/", "~/", text)
    return text


def _redact_value(value: Any) -> Any:
    """Best-effort redaction for arbitrary values written to the dump.

    Strings get :func:`_redact_home` applied. Containers (list/tuple/dict)
    are walked recursively so a tool call's args dict — which may
    contain absolute file paths — is sanitized before it lands in a
    paste-able dump.
    """
    if isinstance(value, str):
        return _redact_home(value)
    if isinstance(value, list):
        return [_redact_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(v) for v in value)
    if isinstance(value, dict):
        return {k: _redact_value(v) for k, v in value.items()}
    return value


def _interesting_env_vars() -> List[Tuple[str, str]]:
    """Same filter the doctor uses: ``GRAPHRAG_*``, ``HF_*``, ``*SSL*``,
    ``*CA_BUNDLE*``. Plus ``GRAPH_DB_PATH`` because it's the most common
    knob a support engineer wants to see, and wouldn't otherwise match
    the prefix filter.
    """
    out: List[Tuple[str, str]] = []
    for key, raw in os.environ.items():
        if (
            key == "GRAPH_DB_PATH"
            or key.startswith("GRAPHRAG_")
            or key.startswith("HF_")
            or "SSL" in key
            or "CA_BUNDLE" in key
        ):
            value = _redact_home(raw) if raw else raw
            out.append((key, value))
    out.sort()
    return out


def _dep_versions() -> List[Tuple[str, str]]:
    """Look up the installed version for each tracked dependency.

    Wrapped so a missing package surfaces as ``"(not installed)"`` rather
    than crashing the dump itself. Dumps are best-effort.
    """
    out: List[Tuple[str, str]] = []
    for dep in _CRASHDUMP_DEPS:
        try:
            out.append((dep, importlib.metadata.version(dep)))
        except importlib.metadata.PackageNotFoundError:
            out.append((dep, "(not installed)"))
        except Exception as exc:  # noqa: BLE001 - dump must not raise
            out.append((dep, f"(lookup failed: {type(exc).__name__})"))
    return out


def _format_crashdump(
    exc_type: Type[BaseException],
    exc_value: BaseException,
    exc_tb: Optional[TracebackType],
) -> str:
    """Render the full crashdump as a single human-readable string.

    Sections:
      - Header (timestamp + pid)
      - Traceback
      - Process info (Python, executable, platform)
      - Dependency versions
      - Redacted env vars
      - Recent calls (or "no recent calls" stub)

    All home paths are redacted via :func:`_redact_home`.
    """
    now = datetime.now(timezone.utc).isoformat()
    pid = os.getpid()

    py_version = sys.version.split()[0]
    plat = platform.platform()
    mac_release, _, _ = platform.mac_ver()

    tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))

    lines: List[str] = []
    lines.append("=== iOS-GraphRAG crashdump ===")
    lines.append(f"timestamp: {now}")
    lines.append(f"pid:       {pid}")
    lines.append("")
    lines.append("=== Traceback ===")
    lines.append(_redact_home(tb_text).rstrip())
    lines.append("")
    lines.append("=== Process info ===")
    lines.append(f"Python:    {py_version}")
    lines.append(f"Executable:{_redact_home(sys.executable)}")
    lines.append(f"Platform:  {plat}")
    if mac_release:
        lines.append(f"macOS:     {mac_release}")
    lines.append("")
    lines.append("=== Dependency versions ===")
    for name, ver in _dep_versions():
        lines.append(f"{name:<25s} {ver}")
    lines.append("")
    lines.append("=== Environment ===")
    env = _interesting_env_vars()
    if not env:
        lines.append("(none set)")
    else:
        for key, value in env:
            lines.append(f"{key} = {value}")
    lines.append("")
    lines.append("=== Recent calls ===")
    snapshot = _snapshot_recent_calls()
    if not snapshot:
        lines.append("(no recent calls)")
    else:
        for entry in snapshot:
            redacted = _redact_value(entry)
            ts = redacted.get("ts", "?")
            tool = redacted.get("tool", "?")
            status = redacted.get("status", "?")
            duration = redacted.get("duration_ms", "?")
            trace_id = redacted.get("trace_id", "?")
            lines.append(
                f"{ts}  tool={tool}  status={status}  "
                f"duration_ms={duration}  trace_id={trace_id}"
            )
            tool_args = redacted.get("tool_args", {})
            if tool_args:
                lines.append(f"  args: {tool_args}")
            err = redacted.get("error")
            if err:
                lines.append(f"  error: {err}")
    return "\n".join(lines) + "\n"


def _write_crashdump(payload: str) -> Optional[Path]:
    """Write ``payload`` to ``<get_log_dir()>/crashdump-<UTC>-<pid>.txt``.

    Best-effort: any OSError (disk full, permission denied, missing dir)
    is swallowed -- the original exception still has to propagate, so
    a dump-write failure must NOT raise. Returns the dump path on
    success and ``None`` on any failure.

    Uses ``_logging.get_log_dir()`` so ``GRAPHRAG_LOG_DIR`` overrides
    work the same way they do for the rotating log files. Creates the
    directory if absent (the user may have wiped it).
    """
    # Local import: keeps module-load cheap and avoids a circular import
    # if _logging ever needs anything from this module (it doesn't today).
    from ._logging import get_log_dir  # noqa: PLC0415

    try:
        log_dir = get_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        # Filename uses a sortable UTC timestamp + pid so multiple crashes
        # don't collide. Colons are fine on macOS but stripped for
        # cross-platform paste-friendliness.
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = log_dir / f"crashdump-{ts}-{os.getpid()}.txt"
        path.write_text(payload, encoding="utf-8")
        return path
    except (OSError, Exception):  # noqa: BLE001 - dump is best-effort
        return None


_PREVIOUS_EXCEPTHOOK: Optional[Any] = None
_HOOK_INSTALLED = False


def _crashdump_excepthook(
    exc_type: Type[BaseException],
    exc_value: BaseException,
    exc_tb: Optional[TracebackType],
) -> None:
    """``sys.excepthook`` replacement that writes a dump then chains.

    Steps:
      1. Build the dump payload from the live exc_info.
      2. Write it under ``get_log_dir()`` (best-effort).
      3. Log a structured "crashdump written" record so the dump path
         shows up in server.log alongside the rest of the lifecycle.
      4. Call the previously-installed excepthook (or the stdlib default
         if none) so Python's normal stderr traceback still appears.

    Even if every step in 1-3 raises, we MUST reach step 4 -- otherwise
    a logging failure swallows the original crash and the user sees
    nothing on stderr. Hence the broad except blocks.
    """
    payload: Optional[str] = None
    try:
        payload = _format_crashdump(exc_type, exc_value, exc_tb)
    except Exception:  # noqa: BLE001 - format failure must not block default hook
        payload = None

    dump_path: Optional[Path] = None
    if payload is not None:
        dump_path = _write_crashdump(payload)

    # Log the dump location through the structured logger so support engineers
    # can grep server.log for "crashdump written". Wrapped in try/except
    # because the logging stack itself might be the thing that crashed.
    try:
        log = logging.getLogger("ios_graphrag.server")
        if dump_path is not None:
            log.error(
                "crashdump written",
                extra={"path": str(dump_path)},
            )
        else:
            log.error("crashdump write skipped or failed")
    except Exception:  # noqa: BLE001
        pass

    # Chain into the previously-installed excepthook so Python's default
    # behavior (print traceback to stderr, set exit code) still runs.
    # If chaining fails for any reason, fall through to sys.__excepthook__
    # as a last resort -- the user must always see the traceback.
    try:
        if _PREVIOUS_EXCEPTHOOK is not None:
            _PREVIOUS_EXCEPTHOOK(exc_type, exc_value, exc_tb)
        else:
            sys.__excepthook__(exc_type, exc_value, exc_tb)
    except Exception:  # noqa: BLE001
        try:
            sys.__excepthook__(exc_type, exc_value, exc_tb)
        except Exception:  # noqa: BLE001
            pass


def install_crashdump_hook() -> None:
    """Install :func:`_crashdump_excepthook` as ``sys.excepthook``.

    Idempotent: a second call is a no-op. The previous excepthook is
    captured in ``_PREVIOUS_EXCEPTHOOK`` so the chain stays intact even
    if other libraries (test runners, IPython, etc.) layered hooks
    underneath us.

    Called from ``server.main()`` AFTER ``setup_logging("server")`` so
    the post-write log emission lands in the right handler chain.
    """
    global _PREVIOUS_EXCEPTHOOK, _HOOK_INSTALLED
    if _HOOK_INSTALLED:
        return
    _PREVIOUS_EXCEPTHOOK = sys.excepthook
    sys.excepthook = _crashdump_excepthook
    _HOOK_INSTALLED = True


def _uninstall_crashdump_hook_for_tests() -> None:
    """Test-only: restore the prior excepthook and clear install state.

    Production code never calls this. Tests that exercise install-twice
    or hook-not-installed paths use it to roll back the global state
    between cases.
    """
    global _PREVIOUS_EXCEPTHOOK, _HOOK_INSTALLED
    if _HOOK_INSTALLED and _PREVIOUS_EXCEPTHOOK is not None:
        sys.excepthook = _PREVIOUS_EXCEPTHOOK
    _PREVIOUS_EXCEPTHOOK = None
    _HOOK_INSTALLED = False
