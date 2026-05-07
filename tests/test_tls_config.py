"""Tests for ``ios_graphrag._tls`` — Phase 2 TLS gating.

The contract under test:

* Default posture (no env, no flag): TLS verification is ON. Calling
  ``configure_insecure_tls_if_requested`` is a no-op that returns False.
* ``GRAPHRAG_INSECURE_TLS=1``: the function activates the legacy bypass and
  emits a WARNING through the configured logger.
* ``configure_cert_bundle(path)``: sets REQUESTS_CA_BUNDLE and SSL_CERT_FILE
  to the absolute path of an existing file; raises FileNotFoundError otherwise.
* Regression posture: no module outside ``_tls.py`` references
  ``_create_unverified_context``.

All tests use monkeypatch fixtures so env-var and ssl-state mutations don't
leak between tests.
"""

from __future__ import annotations

import logging
import ssl
import subprocess
from pathlib import Path

import pytest

from ios_graphrag import _tls

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_ssl_default_context():
    """Snapshot ``ssl._create_default_https_context`` before each test and
    restore it after, so a test that activates the bypass doesn't leak the
    insecure context into subsequent tests."""
    original = ssl._create_default_https_context
    try:
        yield
    finally:
        ssl._create_default_https_context = original


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_env_no_flag_means_tls_verification_on(monkeypatch):
    """Without GRAPHRAG_INSECURE_TLS the function is a no-op and returns False."""
    monkeypatch.delenv(_tls.INSECURE_ENV_VAR, raising=False)

    applied = _tls.configure_insecure_tls_if_requested()

    assert applied is False
    assert ssl._create_default_https_context is not ssl._create_unverified_context


def test_env_var_activates_bypass_and_logs_warning(monkeypatch, caplog):
    """GRAPHRAG_INSECURE_TLS=1 swaps in the unverified context AND logs a WARNING."""
    monkeypatch.setenv(_tls.INSECURE_ENV_VAR, "1")
    # caplog must be told to capture from this logger; otherwise WARNING goes
    # nowhere (the project doesn't propagate to the root logger by default
    # in test isolation).
    caplog.set_level(logging.WARNING, logger=_tls.__name__)

    applied = _tls.configure_insecure_tls_if_requested()

    assert applied is True
    assert ssl._create_default_https_context is ssl._create_unverified_context
    # Find the warning record (formatting goes through %-args, so check the message text).
    warning_records = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "TLS verification disabled" in r.getMessage()
    ]
    assert warning_records, (
        f"expected at least one WARNING about TLS verification disabled; "
        f"got {[r.getMessage() for r in caplog.records]}"
    )


def test_cert_bundle_sets_env_vars(monkeypatch, tmp_path: Path):
    """configure_cert_bundle points REQUESTS_CA_BUNDLE and SSL_CERT_FILE at the file."""
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    bundle = tmp_path / "corp.pem"
    bundle.write_text("# fake corporate bundle for tests\n")

    _tls.configure_cert_bundle(str(bundle))

    import os

    assert os.environ.get("REQUESTS_CA_BUNDLE") == str(bundle.resolve())
    assert os.environ.get("SSL_CERT_FILE") == str(bundle.resolve())


def test_cert_bundle_missing_file_raises(tmp_path: Path):
    """A non-existent path raises FileNotFoundError before mutating env."""
    nonexistent = tmp_path / "does_not_exist.pem"
    with pytest.raises(FileNotFoundError):
        _tls.configure_cert_bundle(str(nonexistent))


def test_grep_no_bypass_outside_tls_module():
    """Regression posture: the only file under src/ios_graphrag/ that mentions
    ``_create_unverified_context`` is ``_tls.py`` itself.

    If this fails, someone re-introduced a module-level SSL bypass somewhere.
    """
    src_root = Path(__file__).parent.parent / "src" / "ios_graphrag"
    assert src_root.is_dir(), f"expected src layout at {src_root}"

    result = subprocess.run(
        [
            "grep",
            "-rln",
            "_create_unverified_context",
            "--include=*.py",
            str(src_root),
        ],
        capture_output=True,
        text=True,
    )
    # grep exits 1 when there are zero matches (which would also be fine —
    # _tls.py contains the string, so we expect rc=0). Treat rc>1 as error.
    assert result.returncode in (0, 1), f"grep failed (rc={result.returncode}): {result.stderr}"

    files_with_hit = [line for line in result.stdout.splitlines() if line.strip()]
    # Normalize: every hit must be inside _tls.py.
    leaks = [f for f in files_with_hit if not f.endswith("_tls.py")]
    assert not leaks, (
        "_create_unverified_context appears outside _tls.py — module-level "
        f"SSL bypass has been re-introduced in: {leaks}"
    )
