"""Tests for ``benchmarks/run.py`` (Phase 4a benchmark harness).

These tests run the harness as a subprocess against tiny inputs and verify
the JSON output has the expected shape. They take ~30-60 seconds combined
because each test triggers a full indexer run (which loads the 500MB
sentence-transformers model on first call, though subsequent calls reuse
the on-disk cache).

Marked indirectly via the ``@pytest.mark.slow`` decorator if available.
The slow marker isn't currently configured in ``pyproject.toml`` so the
tests just run as part of the default suite — they're slow but not
substantially slower than the existing ``test_console_scripts.py`` tests
that also exercise full indexer + server lifecycles.

Each test gets its own tmp_path output directory so concurrent runs and
re-runs don't trip over each other's results.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TEST_FIXTURES_DIR = PROJECT_ROOT / "test_fixtures" / "CalculatorApp"
HARNESS_PATH = PROJECT_ROOT / "benchmarks" / "run.py"


def _run_harness(repo: Path, output_dir: Path, db_path: Path) -> Path:
    """Run the harness and return the path to the produced JSON.

    Uses ``--n-queries=10`` to keep test wall time reasonable; the harness's
    default of 1000 makes each test take ~30s longer for no test value.
    """
    result = subprocess.run(
        [
            sys.executable,
            str(HARNESS_PATH),
            "--repo",
            str(repo),
            "--db",
            str(db_path),
            "--output",
            str(output_dir),
            "--n-queries",
            "10",
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=600,
    )
    assert result.returncode == 0, (
        f"harness failed (rc={result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    # Harness prints the path to the JSON on stdout.
    json_path_str = result.stdout.strip().splitlines()[-1]
    json_path = Path(json_path_str)
    assert json_path.is_file(), f"JSON not found at {json_path}"
    assert json_path.parent == output_dir.resolve()
    return json_path


def test_harness_smoke_against_calculator_app(tmp_path):
    """The harness runs end-to-end against the test fixture and produces JSON.

    Asserts the JSON has metadata + full_index sections, and that the
    indexer actually found symbols (>0). Numbers don't have to be
    representative — this is a smoke test, not a regression on values.
    """
    output_dir = tmp_path / "results"
    db_path = tmp_path / "bench.sqlite"
    json_path = _run_harness(TEST_FIXTURES_DIR, output_dir, db_path)
    data = json.loads(json_path.read_text())

    # Metadata shape.
    assert "metadata" in data
    md = data["metadata"]
    assert "git_sha" in md
    assert "timestamp" in md
    assert "platform" in md
    assert "system" in md["platform"]
    assert "python_version" in md
    # repo_path is absolute; the test fixture has a known basename.
    assert md["repo_path"].endswith("CalculatorApp")

    # Full-index shape.
    assert "full_index" in data
    fi = data["full_index"]
    # If full-index was skipped the harness records {"skipped": True, ...}.
    assert "skipped" not in fi, f"full_index unexpectedly skipped: {fi.get('reason')}"
    assert fi["symbol_count"] > 0, (
        f"expected symbols indexed in CalculatorApp, got {fi['symbol_count']}"
    )
    assert fi["edge_count"] >= 0
    assert fi["db_size_bytes"] > 0
    assert fi["wall_seconds"] > 0
    assert fi["peak_rss_bytes"] > 0
    assert "phases" in fi
    # We expect at least the full-pass phases on a non-empty repo.
    # (incremental + empty-repo paths skip some.)
    assert "hash_seconds" in fi["phases"]
    assert "parse_seconds" in fi["phases"]
    assert "embed_seconds" in fi["phases"]


def test_harness_handles_empty_repo(tmp_path):
    """A directory with no Swift files produces a complete JSON with 0 symbols.

    The full-index pass is still expected to run cleanly — the indexer
    just walks an empty file list. A useful regression check: at one point
    the harness would crash on the SELECT COUNT against a not-yet-created
    DB. This pins that path.
    """
    empty_repo = tmp_path / "empty_repo"
    empty_repo.mkdir()
    # A non-Swift sibling file shouldn't trigger indexing.
    (empty_repo / "README.md").write_text("# empty\n")

    output_dir = tmp_path / "results"
    db_path = tmp_path / "bench.sqlite"
    json_path = _run_harness(empty_repo, output_dir, db_path)
    data = json.loads(json_path.read_text())

    assert "metadata" in data
    assert "full_index" in data
    fi = data["full_index"]
    assert "skipped" not in fi
    assert fi["symbol_count"] == 0, f"expected 0 symbols for empty repo, got {fi['symbol_count']}"
    assert fi["edge_count"] == 0


def test_harness_skips_incremental_when_no_git_history(tmp_path):
    """Fresh git repo (1 commit) -> incremental sections marked skipped, not crashed.

    The harness uses ``git diff HEAD~k..HEAD`` to sample a realistic delta;
    with only 1 commit, every k>=1 fails. We expect the harness to record
    each delta size as ``{"skipped": True, "reason": ...}`` and continue.
    """
    repo = tmp_path / "fresh_repo"
    repo.mkdir()

    # Initialize a fresh git repo with 1 commit. Use isolated user/email so we
    # don't depend on the developer's git config.
    env = os.environ.copy()
    env["GIT_AUTHOR_NAME"] = "Bench Test"
    env["GIT_AUTHOR_EMAIL"] = "bench@test.local"
    env["GIT_COMMITTER_NAME"] = "Bench Test"
    env["GIT_COMMITTER_EMAIL"] = "bench@test.local"
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True, env=env)
    # An empty-ish Swift file gives the indexer something to do without
    # generating noise. The point is: there must be SOMETHING for the
    # harness to index, so we can confirm full_index runs while incremental
    # cleanly skips.
    (repo / "Foo.swift").write_text("class Foo {}\n")
    subprocess.run(["git", "-C", str(repo), "add", "Foo.swift"], check=True, env=env)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", "initial"],
        check=True,
        env=env,
    )

    output_dir = tmp_path / "results"
    db_path = tmp_path / "bench.sqlite"
    json_path = _run_harness(repo, output_dir, db_path)
    data = json.loads(json_path.read_text())

    assert "incremental" in data
    inc = data["incremental"]
    for delta in ("10_files", "100_files", "1000_files"):
        assert delta in inc, f"missing {delta} in incremental section"
        assert inc[delta].get("skipped") is True, (
            f"expected {delta} to be skipped on 1-commit repo, got {inc[delta]}"
        )
        assert "reason" in inc[delta]
