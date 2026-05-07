"""Tests for ``benchmarks/check_regression.py``.

The script is the gate the nightly CI workflow uses to fail on a
fixture-bound regression. The four tests below cover the four signals
the gate watches:

1. Within-tolerance current values produce exit 0.
2. A wall-time slowdown beyond the 25% tolerance fails with a
   ``wall_seconds regressed`` message.
3. A semantic_search p99 slowdown beyond the 35% tolerance fails with
   a ``semantic_search p99 regressed`` message.
4. ANY increase in parse-error rate fails (no tolerance).

Synthesizing the JSONs in tmp_path and invoking the script via
``subprocess.run([sys.executable, ...])`` is the right boundary —
the script is a CLI entrypoint and we want to test the CLI shape,
not the internal helpers.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = PROJECT_ROOT / "benchmarks" / "check_regression.py"


def _make_benchmark_payload(wall_seconds: float, p99_ms: float) -> Dict[str, Any]:
    """Minimal benchmark JSON shape — only the fields the gate reads.

    ``check_benchmark`` looks at exactly two paths:
    ``full_index.wall_seconds`` and ``semantic_search_latency_ms.p99``.
    Everything else is irrelevant for the regression check, so we
    don't bother synthesizing it.
    """
    return {
        "full_index": {"wall_seconds": wall_seconds},
        "semantic_search_latency_ms": {"p99": p99_ms},
    }


def _make_audit_payload(percent: float) -> Dict[str, Any]:
    """Minimal parse-audit JSON shape — only ``summary.files_with_any_error_node_percent``."""
    return {"summary": {"files_with_any_error_node_percent": percent}}


def _write_pair(
    tmp_path: Path,
    benchmark: Dict[str, Any],
    audit: Dict[str, Any],
    suffix: str,
) -> Tuple[Path, Path]:
    bench_path = tmp_path / f"benchmark-{suffix}.json"
    audit_path = tmp_path / f"audit-{suffix}.json"
    bench_path.write_text(json.dumps(benchmark))
    audit_path.write_text(json.dumps(audit))
    return bench_path, audit_path


def _invoke(
    bench_baseline: Path,
    bench_current: Path,
    audit_baseline: Path,
    audit_current: Path,
) -> subprocess.CompletedProcess:
    """Run the script and return the completed process. We don't
    ``check=True`` because some tests assert on a non-zero exit."""
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--benchmark-baseline",
            str(bench_baseline),
            "--benchmark-current",
            str(bench_current),
            "--audit-baseline",
            str(audit_baseline),
            "--audit-current",
            str(audit_current),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_regression_check_passes_when_within_tolerance(tmp_path: Path) -> None:
    """A 10% slowdown is well under both the 25% wall-seconds and 35% p99
    tolerances, and parse-error rate is unchanged. Exit 0; stdout reports
    PASSED."""
    baseline_bench = _make_benchmark_payload(wall_seconds=10.0, p99_ms=20.0)
    baseline_audit = _make_audit_payload(percent=0.0)
    current_bench = _make_benchmark_payload(wall_seconds=11.0, p99_ms=22.0)  # +10% / +10%
    current_audit = _make_audit_payload(percent=0.0)

    bbase, abase = _write_pair(tmp_path, baseline_bench, baseline_audit, "baseline")
    bcur, acur = _write_pair(tmp_path, current_bench, current_audit, "current")

    proc = _invoke(bbase, bcur, abase, acur)
    assert proc.returncode == 0, f"unexpected non-zero exit:\nstdout={proc.stdout}\nstderr={proc.stderr}"
    assert "PASSED" in proc.stdout


def test_regression_check_fails_on_wall_regression(tmp_path: Path) -> None:
    """A 50% wall-time slowdown exceeds the 25% tolerance. Exit 1;
    stderr names ``wall_seconds regressed``."""
    baseline_bench = _make_benchmark_payload(wall_seconds=10.0, p99_ms=20.0)
    baseline_audit = _make_audit_payload(percent=0.0)
    current_bench = _make_benchmark_payload(wall_seconds=15.0, p99_ms=20.0)  # +50%
    current_audit = _make_audit_payload(percent=0.0)

    bbase, abase = _write_pair(tmp_path, baseline_bench, baseline_audit, "baseline")
    bcur, acur = _write_pair(tmp_path, current_bench, current_audit, "current")

    proc = _invoke(bbase, bcur, abase, acur)
    assert proc.returncode == 1, f"expected exit 1, got {proc.returncode}"
    assert "wall_seconds regressed" in proc.stderr
    assert "FAILED" in proc.stderr


def test_regression_check_fails_on_p99_regression(tmp_path: Path) -> None:
    """A 50% p99 slowdown exceeds the 35% tolerance. Exit 1; stderr
    names ``semantic_search p99 regressed``."""
    baseline_bench = _make_benchmark_payload(wall_seconds=10.0, p99_ms=20.0)
    baseline_audit = _make_audit_payload(percent=0.0)
    current_bench = _make_benchmark_payload(wall_seconds=10.0, p99_ms=30.0)  # +50%
    current_audit = _make_audit_payload(percent=0.0)

    bbase, abase = _write_pair(tmp_path, baseline_bench, baseline_audit, "baseline")
    bcur, acur = _write_pair(tmp_path, current_bench, current_audit, "current")

    proc = _invoke(bbase, bcur, abase, acur)
    assert proc.returncode == 1, f"expected exit 1, got {proc.returncode}"
    assert "semantic_search p99 regressed" in proc.stderr
    assert "FAILED" in proc.stderr


def test_regression_check_fails_on_any_parse_error_increase(tmp_path: Path) -> None:
    """Parse-error rate is the strictest signal — ANY increase fails,
    even a fractional one. Baseline 0%, current 0.5%; exit 1; stderr
    names ``parse-error rate regressed``."""
    baseline_bench = _make_benchmark_payload(wall_seconds=10.0, p99_ms=20.0)
    baseline_audit = _make_audit_payload(percent=0.0)
    current_bench = _make_benchmark_payload(wall_seconds=10.0, p99_ms=20.0)
    current_audit = _make_audit_payload(percent=0.5)  # any increase

    bbase, abase = _write_pair(tmp_path, baseline_bench, baseline_audit, "baseline")
    bcur, acur = _write_pair(tmp_path, current_bench, current_audit, "current")

    proc = _invoke(bbase, bcur, abase, acur)
    assert proc.returncode == 1, f"expected exit 1, got {proc.returncode}"
    assert "parse-error rate regressed" in proc.stderr
    assert "FAILED" in proc.stderr
