"""Tests for ``tools/parse_audit.py`` (Phase 4.5a parse-error audit).

Each test exercises a discrete behavior the owner relies on when running
the audit on the real iOS monorepo:

- the smoke path (clean Swift fixture -> Strategy A);
- the error-detection path (one broken file -> has_error + ERROR count);
- the empty-input path (no source files -> no crash, total_files == 0);
- the directory-skip path (``.build/`` contents are not audited).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make the in-repo `tools/` package importable so we can call run_audit()
# directly. Adding to sys.path is fine here — this is test-only and matches
# how benchmarks/run.py would be tested if it were tested in-process.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from tools import parse_audit  # noqa: E402

CALCULATOR_FIXTURE = _PROJECT_ROOT / "test_fixtures" / "CalculatorApp"


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def test_audit_smoke_against_calculator_app(tmp_path: Path) -> None:
    """End-to-end smoke against the bundled clean fixture.

    The fixture is intentionally clean Swift; we expect zero parse errors
    and the Strategy A recommendation. We pin workers=1 so the test is
    deterministic in CI (otherwise multiprocessing with spawn can be
    flaky on tiny inputs)."""
    json_path = parse_audit.run_audit(
        repo=CALCULATOR_FIXTURE,
        extensions=parse_audit.DEFAULT_EXTENSIONS,
        output_dir=tmp_path,
        workers=1,
    )
    payload = _read_json(json_path)

    # Required top-level + metadata keys.
    assert set(payload.keys()) == {"metadata", "summary", "top_files", "files"}
    md = payload["metadata"]
    for key in (
        "git_sha",
        "timestamp",
        "repo_path",
        "tree_sitter_version",
        "tree_sitter_swift_version",
        "tree_sitter_objc_version",
        "extensions_audited",
        "platform",
        "python_version",
    ):
        assert key in md, f"missing metadata key: {key}"
    # Grammar versions must be recorded for reproducibility (the whole
    # point of the audit is comparable runs).
    assert md["tree_sitter_swift_version"] != "unknown"
    assert md["tree_sitter_objc_version"] != "unknown"

    summary = payload["summary"]
    # The fixture has 8 source files (7 .swift sources + Package.swift).
    # Locking the count guards against accidental fixture changes.
    assert summary["total_files"] == 8, (
        f"fixture file count drifted: got {summary['total_files']}"
    )
    assert summary["files_with_has_error"] == 0
    assert summary["files_with_any_error_node"] == 0
    assert summary["files_with_any_error_node_percent"] == 0.0
    assert summary["decision_recommendation"].startswith("Strategy A")
    # Top files list is empty when there are no errors.
    assert payload["top_files"] == []
    # All bins zero except "0".
    hist = summary["histogram_error_node_counts"]
    assert hist["0"] == 8
    assert hist["1-5"] == hist["6-20"] == hist["21-100"] == hist["100+"] == 0


def test_audit_handles_intentionally_broken_swift(tmp_path: Path) -> None:
    """A known-broken Swift snippet must trip ``has_error`` and at least
    one ``ERROR`` node. This is the audit's sentinel test — if it doesn't
    catch this, it can't catch real grammar regressions."""
    repo = tmp_path / "repo"
    repo.mkdir()
    # Multiple syntax errors so we get a stable ERROR-node count > 0
    # even if the grammar's recovery improves.
    (repo / "broken.swift").write_text(
        "func foo( { broken syntax\nclass A: }\n",
        encoding="utf-8",
    )

    output = tmp_path / "out"
    json_path = parse_audit.run_audit(
        repo=repo,
        extensions=parse_audit.DEFAULT_EXTENSIONS,
        output_dir=output,
        workers=1,
    )
    payload = _read_json(json_path)
    summary = payload["summary"]
    assert summary["total_files"] == 1
    assert summary["files_with_has_error"] == 1
    assert summary["files_with_any_error_node"] == 1
    file_record = payload["files"][0]
    assert file_record["has_error"] is True
    assert file_record["error_node_count"] > 0, file_record
    # 100% error rate -> Strategy C per the decision matrix.
    assert "Strategy C" in summary["decision_recommendation"]


def test_audit_handles_empty_dir(tmp_path: Path) -> None:
    """An empty repo must not crash, and must produce a valid JSON
    document with total_files == 0."""
    repo = tmp_path / "empty"
    repo.mkdir()
    output = tmp_path / "out"
    json_path = parse_audit.run_audit(
        repo=repo,
        extensions=parse_audit.DEFAULT_EXTENSIONS,
        output_dir=output,
        workers=1,
    )
    payload = _read_json(json_path)
    assert payload["summary"]["total_files"] == 0
    assert payload["summary"]["files_with_any_error_node"] == 0
    # The decision string is still produced even on empty input. The audit
    # exits 0 — the operator should look at total_files before trusting
    # the recommendation, which the console summary makes obvious.
    assert "Strategy" in payload["summary"]["decision_recommendation"]
    assert payload["files"] == []


def test_audit_skips_excluded_dirs(tmp_path: Path) -> None:
    """Files under ``.build/`` (and other ``SKIP_DIRS`` entries) must be
    excluded from the audit even if they have indexable extensions.

    The indexer's ``BLOCK_DIRS`` excludes these from indexing, so
    measuring them in the audit would produce error rates that don't
    reflect what the indexer would actually hit."""
    repo = tmp_path / "repo"
    (repo / ".build").mkdir(parents=True)
    (repo / "Pods").mkdir()
    (repo / ".build" / "junk.swift").write_text("not even valid", encoding="utf-8")
    (repo / "Pods" / "vendored.swift").write_text("also broken {", encoding="utf-8")

    output = tmp_path / "out"
    json_path = parse_audit.run_audit(
        repo=repo,
        extensions=parse_audit.DEFAULT_EXTENSIONS,
        output_dir=output,
        workers=1,
    )
    payload = _read_json(json_path)
    assert payload["summary"]["total_files"] == 0, (
        f"audit should skip .build/ and Pods/ but got {payload['summary']}"
    )
    assert payload["files"] == []
