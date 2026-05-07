"""
Pytest configuration and fixtures for iOS-GraphRAG tests.

Provides:
- Indexed test database fixture
- Database connection fixture
- Path constants for test fixtures
"""

import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Generator

import pytest

# Path constants
PROJECT_ROOT = Path(__file__).parent.parent
ENGINE_DIR = PROJECT_ROOT / "engine"
TEST_FIXTURES_DIR = PROJECT_ROOT / "test_fixtures" / "CalculatorApp"
SCHEMA_PATH = ENGINE_DIR / "database" / "schema.sql"
TEST_DB_PATH = PROJECT_ROOT / "test-knowledge-graph.sqlite"


@pytest.fixture(scope="session")
def indexed_db_path(tmp_path_factory) -> Generator[Path, None, None]:
    """
    Session-scoped fixture that indexes the test fixture project
    and returns the path to the test database.

    The database is created fresh at the start of the test session
    and cleaned up at the end.

    Phase 6b: ``GRAPHRAG_LOG_DIR`` is pinned to a session-scoped tmpdir so
    the indexer subprocess invoked here does not rotate logs into the
    user's ``~/Library/Logs/ios-graphrag/`` directory on every test run.
    """
    # Remove any existing test database
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()

    # Set environment variable for the indexer
    env = os.environ.copy()
    env["GRAPH_DB_PATH"] = str(TEST_DB_PATH)
    log_dir = tmp_path_factory.mktemp("ios-graphrag-logs")
    env["GRAPHRAG_LOG_DIR"] = str(log_dir)

    # Run the indexer on the test fixtures
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ios_graphrag.indexer",
            "--repo",
            str(TEST_FIXTURES_DIR),
            "--db",
            str(TEST_DB_PATH),
            "--full",
        ],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        pytest.fail(f"Indexer failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")

    yield TEST_DB_PATH

    # Cleanup after all tests
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()


@pytest.fixture
def db_connection(indexed_db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    """
    Provides a database connection to the indexed test database.
    """
    conn = sqlite3.connect(indexed_db_path)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


@pytest.fixture
def test_fixtures_path() -> Path:
    """Returns the path to the test fixtures directory."""
    return TEST_FIXTURES_DIR
