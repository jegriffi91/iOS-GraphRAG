"""Smoke test: `uv run ruff check .` exits 0.

Phase 6e adds ruff to dev deps and a permissive baseline config (E/F/I)
in pyproject.toml. CI runs ruff explicitly, but a developer running
`uv run pytest` locally without remembering to invoke ruff would
otherwise miss a regression. This test catches that.

The matching `ruff format --check` pass is intentionally NOT mirrored
here — running the formatter on every pytest invocation is too noisy
locally, and CI already enforces it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent


def test_ruff_check_clean():
    result = subprocess.run(
        ["uv", "run", "ruff", "check", "."],
        cwd=str(_PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"ruff check failed (rc={result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
