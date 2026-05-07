"""File-system watcher for iOS-GraphRAG (Phase 4c).

``ios-graphrag-watch`` monitors a Swift/Objective-C repo and triggers an
incremental reindex on each meaningful change, keeping the embedding
model loaded between runs (Phase 4b).

Debounce window: 1 second by default (``--debounce-seconds``). Multiple
FS events arriving within the window collapse into a single reindex.
The window is implemented as a cancel-and-restart timer (sliding
window): every event resets the wait, so a developer typing a burst of
saves only triggers ONE reindex after they stop.

Files matching the indexer's existing ``.swift`` / ``.h`` / ``.m`` set
are eligible. The watcher itself is opt-in; production deployments
install via launchd (see ``dist/launchd/com.user.ios-graphrag.watcher.plist``).

Why a long-lived process: Phase 4b moved the embedding model to an
in-process module-level cache (``indexer._MODEL_CACHE``). The first
``index_repository`` call in this process loads the model (~5s); every
subsequent call reuses it. The watcher exists to keep that cache warm
across many small deltas — exactly the workflow that hurt most before
Phase 4b/4c.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from pathlib import Path
from typing import Optional, Set

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from ios_graphrag import _logging, _tls, indexer

log = logging.getLogger(__name__)

DEFAULT_DEBOUNCE_SECONDS = 1.0
DEFAULT_EXTENSIONS = (".swift", ".h", ".m")


class DebouncedReindexer:
    """Coalesces FS events and triggers an incremental reindex after a quiet period.

    Thread-safe: events arrive on the watchdog observer thread; the flush
    runs on a single ``threading.Timer`` thread that calls
    ``indexer.index_repository``. The model cache lives on the indexer
    module, so successive flushes reuse the warm model.

    Sliding-window debounce: every ``on_change`` cancels the pending
    timer and starts a new one. A burst of N events that arrive within
    ``debounce`` seconds of each other produces exactly one flush, fired
    ``debounce`` seconds after the LAST event.
    """

    def __init__(self, repo: Path, db: Path, debounce: float = DEFAULT_DEBOUNCE_SECONDS):
        self.repo = repo
        self.db = db
        self.debounce = debounce
        self._pending: Set[str] = set()
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None
        self._stop = threading.Event()

    def on_change(self, path: str) -> None:
        with self._lock:
            self._pending.add(path)
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.debounce, self._flush)
            self._timer.daemon = True
            self._timer.start()

    def _flush(self) -> None:
        if self._stop.is_set():
            return
        with self._lock:
            paths = self._pending.copy()
            self._pending.clear()
            self._timer = None
        if not paths:
            return
        log.info("debounce flush: reindexing after %d changed file(s)", len(paths))
        try:
            # Incremental reindex; index_repository handles change detection
            # via the file_hashes table. The signature is positional:
            # (repo_root, db_path, full_reindex=False).
            indexer.index_repository(str(self.repo), str(self.db), full_reindex=False)
            log.info("incremental reindex complete")
        except Exception as e:
            log.exception("incremental reindex failed: %s", e)
            # Don't kill the watcher on a single failed reindex; next event
            # will trigger another attempt.

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None


class _Handler(FileSystemEventHandler):
    """Forwards relevant FS events to the debouncer.

    Filters by file extension before notifying the debouncer so we
    don't wake up the reindex pipeline for ``.DS_Store`` writes,
    Xcode build artifacts, or any other non-source churn.
    """

    def __init__(self, debouncer: DebouncedReindexer, extensions: tuple[str, ...]):
        self._debouncer = debouncer
        self._extensions = extensions

    def _is_eligible(self, path: str) -> bool:
        return path.endswith(self._extensions)

    def on_modified(self, event):
        if not event.is_directory and self._is_eligible(event.src_path):
            self._debouncer.on_change(event.src_path)

    def on_created(self, event):
        if not event.is_directory and self._is_eligible(event.src_path):
            self._debouncer.on_change(event.src_path)

    def on_moved(self, event):
        # Renames/moves arrive as a single moved event; the destination
        # path is the post-rename file we want to (re)index.
        if not event.is_directory and self._is_eligible(event.dest_path):
            self._debouncer.on_change(event.dest_path)

    def on_deleted(self, event):
        if not event.is_directory and self._is_eligible(event.src_path):
            self._debouncer.on_change(event.src_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="ios-graphrag-watch",
        description="Watch an iOS repo and incrementally reindex on changes.",
    )
    parser.add_argument(
        "--repo", required=True, type=Path, help="Repo root path to monitor"
    )
    parser.add_argument(
        "--db", required=True, type=Path, help="SQLite DB path"
    )
    parser.add_argument(
        "--debounce-seconds",
        type=float,
        default=DEFAULT_DEBOUNCE_SECONDS,
        help="FS event debounce window (default: %(default)s)",
    )
    parser.add_argument(
        "--cert-bundle",
        type=Path,
        default=None,
        help="Optional corporate CA bundle path",
    )
    args = parser.parse_args()

    _logging.setup_logging("watcher")
    _tls.configure_insecure_tls_if_requested()
    if args.cert_bundle:
        _tls.configure_cert_bundle(str(args.cert_bundle))

    repo = args.repo.expanduser().resolve()
    db = args.db.expanduser().resolve()

    if not repo.is_dir():
        log.error("repo path is not a directory: %s", repo)
        return 1

    debouncer = DebouncedReindexer(repo=repo, db=db, debounce=args.debounce_seconds)
    handler = _Handler(debouncer, DEFAULT_EXTENSIONS)
    observer = Observer()
    observer.schedule(handler, str(repo), recursive=True)
    observer.start()
    log.info(
        "ios-graphrag-watch started: repo=%s db=%s debounce=%.1fs",
        repo,
        db,
        args.debounce_seconds,
    )

    # Schedule an initial sync so the index reflects current disk state
    # even if no events arrive immediately. Falls into the same debounce
    # window, so it costs ~debounce_seconds before the first reindex.
    debouncer.on_change(str(repo))

    stop_event = threading.Event()

    def _on_signal(signum, frame):
        log.info("received signal %d; shutting down", signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    try:
        while not stop_event.wait(0.5):
            pass
    finally:
        log.info("stopping observer")
        observer.stop()
        observer.join()
        debouncer.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
