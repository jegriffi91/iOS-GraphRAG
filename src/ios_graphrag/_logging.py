"""Centralized logging setup for iOS-GraphRAG (Phase 6b).

Both ios-graphrag-index and ios-graphrag-server call setup_logging() from
their main(). Layout:

  ~/Library/Logs/ios-graphrag/
    indexer.log   (rotating, 50 MB x 5)
    server.log    (rotating, 50 MB x 5)

Stream format: human-readable text on stderr (for interactive use + tests).
File format: JSON, one record per line (NDJSON), with timestamp/level/logger/
message and any trace_id/extra fields attached via the ``extra=`` kwarg.

GRAPHRAG_LOG_LEVEL controls verbosity (default INFO).
GRAPHRAG_LOG_DIR overrides the default ~/Library/Logs/ios-graphrag path
(used in tests + CI to avoid polluting the user's Library directory).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

DEFAULT_LOG_DIR = Path.home() / "Library" / "Logs" / "ios-graphrag"
ROTATION_BYTES = 50 * 1024 * 1024
ROTATION_BACKUPS = 5


# logging.LogRecord exposes a fixed set of attributes we never want to surface
# as JSON fields — they're stdlib bookkeeping, not user payload. Anything not
# in this set (and not already in our base payload) is treated as caller-
# attached extra (e.g. trace_id, duration_ms, tool, args).
_RESERVED_RECORD_ATTRS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


class JsonLineFormatter(logging.Formatter):
    """Emits one JSON object per log record (NDJSON).

    Base fields: ``ts``, ``level``, ``logger``, ``message``.
    Exception info, if present, is rendered into ``exc_info``.
    Any caller-attached extras (``log.info(..., extra={"trace_id": x})``)
    are pulled from ``record.__dict__`` and merged into the payload.
    """

    def _format_ts(self, record: logging.LogRecord) -> str:
        """Render the record's creation time as ISO-8601 with milliseconds.

        ``logging.Formatter.formatTime`` calls ``time.strftime`` against a
        :class:`time.struct_time`, which doesn't expand ``%f``. We assemble
        the ISO-8601 string ourselves and append the record's ``msecs`` so
        log lines stay sortable to millisecond precision.
        """
        ct = time.localtime(record.created)
        return time.strftime("%Y-%m-%dT%H:%M:%S", ct) + f".{int(record.msecs):03d}"

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self._format_ts(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        # Pull any extra fields the caller attached via the ``extra=`` kwarg.
        # Skip private (underscore-prefixed) attrs and the stdlib's own
        # bookkeeping fields so the JSON output is just our message + extras.
        for key, value in record.__dict__.items():
            if key in payload or key.startswith("_"):
                continue
            if key in _RESERVED_RECORD_ATTRS:
                continue
            payload[key] = value

        return json.dumps(payload, default=str)


def get_log_dir() -> Path:
    """Return the active log directory.

    Honours ``GRAPHRAG_LOG_DIR`` (test/CI override) and falls back to the
    documented macOS Logs path. Note the env var is read on every call so
    a test that sets it via monkeypatch immediately takes effect.
    """
    override = os.environ.get("GRAPHRAG_LOG_DIR")
    return Path(override) if override else DEFAULT_LOG_DIR


def setup_logging(component: str, *, level: Optional[str] = None) -> Path:
    """Configure root logger for the named component ('indexer' or 'server').

    - ``component`` becomes the basename of the log file: ``indexer.log`` /
      ``server.log``.
    - Stream handler emits human-readable text to stderr (kept for the
      ``test_indexer_stdout.py`` invariant + interactive use).
    - File handler is a ``RotatingFileHandler`` (50 MB x 5) with the
      JSON-line formatter.
    - Idempotent: calling twice does not double-attach handlers; any
      previously-installed root handlers are removed first so the
      indexer's worker subprocesses (which re-import this module) see a
      clean handler chain rather than stacking duplicates.

    Returns the path to the file handler's log file.
    """
    log_level = level or os.environ.get("GRAPHRAG_LOG_LEVEL", "INFO").upper()
    log_dir = get_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{component}.log"

    text_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    json_fmt = JsonLineFormatter()

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(text_fmt)

    file_handler = RotatingFileHandler(
        log_file, maxBytes=ROTATION_BYTES, backupCount=ROTATION_BACKUPS
    )
    file_handler.setFormatter(json_fmt)

    root = logging.getLogger()
    # Detach any prior handlers so re-init is idempotent (worker subprocesses
    # re-import this module; tests may call setup_logging() in different
    # phases of their lifecycle).
    for handler in list(root.handlers):
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    # Resolve string level (e.g. "INFO") to numeric. ``getLevelName`` returns
    # the string back if the level is unknown, so we guard with a default.
    numeric_level = logging.getLevelName(log_level)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    root.setLevel(numeric_level)
    root.addHandler(stderr_handler)
    root.addHandler(file_handler)
    return log_file
