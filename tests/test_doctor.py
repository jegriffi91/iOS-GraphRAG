"""Tests for the Phase 3 P3.4 doctor CLI.

Covers:
* ``ios-graphrag-doctor`` runs end-to-end on a healthy system (i.e. one
  where the test fixture has been indexed at least once via the
  shared ``indexed_db_path`` fixture from conftest.py).
* ``ios-graphrag-doctor --bug-report`` produces the expanded sections
  (env vars, dep versions, log tail).
* The home-dir redaction helper masks ``/Users/<name>/`` -> ``~``.
* Each top-level diagnose function returns a Diagnosis with a known
  status.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from ios_graphrag import doctor


_PROJECT_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Subprocess end-to-end tests
# ---------------------------------------------------------------------------


def test_doctor_runs_on_healthy_system(indexed_db_path: Path):
    """`ios-graphrag-doctor` exits cleanly when the test DB exists.

    Uses the session-scoped ``indexed_db_path`` fixture so the doctor
    has a real SQLite file to inspect.
    """
    env = os.environ.copy()
    env["GRAPH_DB_PATH"] = str(indexed_db_path)

    result = subprocess.run(
        ["uv", "run", "ios-graphrag-doctor"],
        cwd=str(_PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    # Doctor exits 0 unless a hard ERROR diagnosis surfaces. On this machine
    # MPS may be unavailable (WARN) but that does not cause a non-zero exit.
    assert result.returncode == 0, (
        f"doctor exited rc={result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    # Header line and the expected diagnostic checks must appear.
    assert "ios-graphrag-doctor" in result.stdout
    assert "DB exists at" in result.stdout
    assert "Schema version" in result.stdout
    assert "Embedding model cache" in result.stdout
    assert "Tree-sitter grammars" in result.stdout
    assert "Last 5 errors" in result.stdout


def test_doctor_bug_report_includes_env_section(indexed_db_path: Path):
    """`ios-graphrag-doctor --bug-report` adds env / deps / log-tail sections."""
    env = os.environ.copy()
    env["GRAPH_DB_PATH"] = str(indexed_db_path)
    # A GRAPHRAG_-prefixed env var must surface in the env section.
    env["GRAPHRAG_BOGUS_VAR"] = "hello"

    result = subprocess.run(
        ["uv", "run", "ios-graphrag-doctor", "--bug-report"],
        cwd=str(_PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0
    assert "=== Environment ===" in result.stdout
    assert "=== Dependency versions ===" in result.stdout
    assert "=== Relevant env vars ===" in result.stdout
    assert "GRAPHRAG_BOGUS_VAR" in result.stdout
    # Verify dependency version table includes torch.
    assert "torch" in result.stdout


# ---------------------------------------------------------------------------
# Unit tests of individual helpers
# ---------------------------------------------------------------------------


def test_redact_home_replaces_user_path():
    sample = "/Users/jdoe/dev/repo/file.py and /Users/asmith/secret"
    out = doctor._redact_home(sample)
    assert "/Users/jdoe/" not in out
    assert "/Users/asmith/" not in out
    assert "~/dev/repo/file.py" in out


def test_diagnose_db_exists_handles_missing(monkeypatch, tmp_path):
    """When GRAPH_DB_PATH points at a non-existent file, status is MISSING."""
    bogus = tmp_path / "nope.sqlite"
    monkeypatch.setenv("GRAPH_DB_PATH", str(bogus))
    diag = doctor.diagnose_db_exists()
    assert diag.status == doctor.MISSING
    assert "file not found" in diag.details


def test_diagnose_graph_db_path_env_unset(monkeypatch):
    """Unset env var is OK with a note about the default fallback."""
    monkeypatch.delenv("GRAPH_DB_PATH", raising=False)
    diag = doctor.diagnose_graph_db_path_env()
    assert diag.status == doctor.OK
    assert "unset" in diag.details


def test_diagnose_tree_sitter_grammars_loads():
    """Both grammars should be importable in this environment."""
    diag = doctor.diagnose_tree_sitter_grammars()
    assert diag.status == doctor.OK
    assert "swift + objc" in diag.details
