"""Tests for the Phase 3 P3.1 preflight CLI.

The full preflight runs an indexer subprocess (which spawns a child
process to load the embedding model). Both of those steps need the
nomic-embed-text-v1.5 model in the HF cache to be useful — that is
~500 MB and we don't want CI to download it unconditionally.

Strategy:
* ``test_preflight_passes_on_smoke_install`` runs the full CLI via
  ``uv run ios-graphrag-preflight`` and asserts rc=0. We skip when the
  model isn't already present in the HF cache so a clean CI environment
  doesn't kick off a half-gig download silently.
* ``test_preflight_fails_loud_on_python_too_old`` is a unit test for
  the python-version check. monkeypatch swaps ``sys.version_info`` and
  asserts the returned ``CheckResult`` is a fail.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ios_graphrag import preflight


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_PROJECT_ROOT = Path(__file__).parent.parent
_HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"
_MODEL_DIR = _HF_HUB / "models--nomic-ai--nomic-embed-text-v1.5"


def _model_cached() -> bool:
    """True iff the embedding model has at least one snapshot with a weight file.

    Mirrors ``doctor.diagnose_model_cache``: presence of the directory plus a
    snapshot folder with model.safetensors or pytorch_model.bin is what
    sentence-transformers needs to load offline.
    """
    snapshots = _MODEL_DIR / "snapshots"
    if not snapshots.is_dir():
        return False
    for entry in snapshots.iterdir():
        if not entry.is_dir():
            continue
        for weight in ("model.safetensors", "pytorch_model.bin"):
            if (entry / weight).exists():
                return True
    return False


# ---------------------------------------------------------------------------
# Subprocess smoke test (gated on cached model)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _model_cached(),
    reason=(
        "nomic-embed-text-v1.5 not cached locally; running the full preflight "
        "would trigger a ~500MB download. Run "
        "`huggingface-cli download nomic-ai/nomic-embed-text-v1.5` to enable."
    ),
)
def test_preflight_passes_on_smoke_install():
    """End-to-end: `uv run ios-graphrag-preflight` exits 0 on this dev box."""
    result = subprocess.run(
        ["uv", "run", "ios-graphrag-preflight"],
        cwd=str(_PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=20 * 60,  # generous: subprocess + model load + smoke index
    )
    assert result.returncode == 0, (
        f"preflight returned rc={result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    # Must show summary line and at least one [PASS] entry. Avoid asserting on
    # the WARN row count — MPS availability differs across hosts.
    assert "[PASS]" in result.stdout
    assert "Summary:" in result.stdout
    # Footer should appear on green runs.
    assert "Recommended DB path:" in result.stdout


# ---------------------------------------------------------------------------
# Unit tests of individual checks
# ---------------------------------------------------------------------------


def test_preflight_fails_loud_on_python_too_old(monkeypatch):
    """check_python_version returns a failure for Python <3.11."""
    fake = (3, 9, 0, "final", 0)

    class _Vinfo(tuple):
        # sys.version_info exposes both index access and named attributes.
        # Use a tuple subclass so attribute lookups still work.
        @property
        def major(self):
            return self[0]

        @property
        def minor(self):
            return self[1]

        @property
        def micro(self):
            return self[2]

    monkeypatch.setattr(preflight.sys, "version_info", _Vinfo(fake))

    result = preflight.check_python_version()

    assert result.status == "fail"
    assert "3.9" in result.detail
    assert "Python" in result.remediation


def test_render_results_summarizes_pass_warn_fail():
    """render_results produces a single Summary line accounting for each status."""
    fake_results = [
        preflight.CheckResult(name="A", status="pass"),
        preflight.CheckResult(name="B", status="warn", detail="soft"),
        preflight.CheckResult(name="C", status="fail", detail="hard", remediation="fix"),
    ]
    rendered = preflight.render_results(fake_results)
    assert "1 passed" in rendered
    assert "1 warned" in rendered
    assert "1 failed" in rendered
    # Remediation should appear under the failing entry.
    assert "-> fix" in rendered


def test_render_config_footer_uses_home(tmp_path):
    """The footer bakes the absolute DB path into both Copilot CLI and VS Code snippets."""
    rendered = preflight.render_config_footer(home=tmp_path)
    expected_db = str(tmp_path / ".cache" / "ios-graphrag" / "graph.sqlite")
    assert expected_db in rendered
    assert "ios-graphrag-server" in rendered
    assert "github.copilot.chat.mcp.servers" in rendered
    # The "verify against your current Copilot CLI version" caveat must be
    # present so users don't trust the snippet blindly.
    assert "verify" in rendered.lower()
