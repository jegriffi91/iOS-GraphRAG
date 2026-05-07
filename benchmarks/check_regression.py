"""Compare a freshly-generated benchmark JSON to the committed baseline.

Used by the nightly CI workflow. Exits 0 if within tolerance, 1 if regressed.
The thresholds are loose because (a) fixture-based numbers are noisy and
(b) we want to catch egregious regressions, not micro-optimizations.

Tolerances (from roadmap section 3 Phase 6e):
  - Full reindex wall_seconds: regression > 25% fails  (loosened from spec's
    10% because fixture timing is noisier than 1M-LOC repo timing).
  - semantic_search p99 ms: regression > 35% fails (loosened from 20%).
  - Parse-error percent: ANY increase fails (regressions in parser
    robustness should be caught immediately).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

REGRESSION_TOLERANCES = {
    "wall_seconds": 0.25,
    "semantic_search_p99_ms": 0.35,
}


def check_benchmark(baseline_path: Path, current_path: Path) -> List[str]:
    """Compare a current benchmark JSON to the committed baseline.

    Returns a list of human-readable failure messages. Empty list means
    the current run is within tolerance for every gated metric. Each
    message names the metric, the baseline value, the current value,
    and the percentage delta — enough for a CI log reader to triage
    without re-opening the JSONs.
    """
    failures: List[str] = []
    base = json.loads(baseline_path.read_text())
    cur = json.loads(current_path.read_text())

    base_wall = base["full_index"]["wall_seconds"]
    cur_wall = cur["full_index"]["wall_seconds"]
    if cur_wall > base_wall * (1 + REGRESSION_TOLERANCES["wall_seconds"]):
        failures.append(
            f"full_index.wall_seconds regressed: {base_wall:.2f}s -> {cur_wall:.2f}s "
            f"({(cur_wall/base_wall - 1)*100:.1f}% > {REGRESSION_TOLERANCES['wall_seconds']*100:.0f}%)"
        )

    base_p99 = base["semantic_search_latency_ms"]["p99"]
    cur_p99 = cur["semantic_search_latency_ms"]["p99"]
    if cur_p99 > base_p99 * (1 + REGRESSION_TOLERANCES["semantic_search_p99_ms"]):
        failures.append(
            f"semantic_search p99 regressed: {base_p99:.2f}ms -> {cur_p99:.2f}ms "
            f"({(cur_p99/base_p99 - 1)*100:.1f}% > {REGRESSION_TOLERANCES['semantic_search_p99_ms']*100:.0f}%)"
        )

    return failures


def check_parse_audit(baseline_path: Path, current_path: Path) -> List[str]:
    """Compare a current parse-audit JSON to the baseline.

    Any increase in the percentage of files producing parse errors fails
    immediately — there's no "noise" tolerance here. Tree-sitter is
    deterministic against pinned grammar versions, so a higher error
    rate means something genuinely changed (a fixture file got corrupted,
    a grammar bump regressed, or our walker is over-counting).
    """
    failures: List[str] = []
    base = json.loads(baseline_path.read_text())
    cur = json.loads(current_path.read_text())

    base_pct = base["summary"]["files_with_any_error_node_percent"]
    cur_pct = cur["summary"]["files_with_any_error_node_percent"]
    if cur_pct > base_pct:
        failures.append(
            f"parse-error rate regressed: {base_pct:.1f}% -> {cur_pct:.1f}% (any increase fails)"
        )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check current benchmarks against baseline.")
    parser.add_argument("--benchmark-baseline", type=Path, required=True)
    parser.add_argument("--benchmark-current", type=Path, required=True)
    parser.add_argument("--audit-baseline", type=Path, required=True)
    parser.add_argument("--audit-current", type=Path, required=True)
    args = parser.parse_args()

    failures: List[str] = []
    failures.extend(check_benchmark(args.benchmark_baseline, args.benchmark_current))
    failures.extend(check_parse_audit(args.audit_baseline, args.audit_current))

    if failures:
        print("Regression check FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("Regression check PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
