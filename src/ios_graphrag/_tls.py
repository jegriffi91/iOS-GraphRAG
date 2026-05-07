"""Single opt-in surface for TLS configuration.

All SSL/TLS bypass code in this project lives here. Both the indexer and the
server (and the embedding worker the indexer spawns) call into these helpers
from their `main()` instead of mutating the SSL stack at module-import time.

Default posture: TLS verification is ON. Two opt-in escape hatches:

* ``GRAPHRAG_INSECURE_TLS=1``    — disables verification entirely, logs a
  WARNING.  Last-resort knob for environments that block all outbound HTTPS
  with a self-signed proxy and where the corporate CA bundle is unavailable.
* ``--cert-bundle PATH``         — points ``REQUESTS_CA_BUNDLE`` and
  ``SSL_CERT_FILE`` at a corporate CA bundle. Verification stays ON. This is
  the recommended path; see ``engine/CONNECTION_GUIDE.md``.
"""
from __future__ import annotations

import logging
import os
import ssl

log = logging.getLogger(__name__)

INSECURE_ENV_VAR = "GRAPHRAG_INSECURE_TLS"
INSECURE_WARNING = (
    "TLS verification disabled by %s=1; this is insecure and should only "
    "be used with corporate proxy approval. Prefer --cert-bundle PATH or "
    "REQUESTS_CA_BUNDLE/SSL_CERT_FILE pointing at your corporate CA bundle."
)


def configure_insecure_tls_if_requested() -> bool:
    """Apply the legacy unconditional SSL bypass IFF GRAPHRAG_INSECURE_TLS=1.

    Returns True if bypass was applied, False otherwise.

    The bypass mirrors the pre-Phase-2 behavior that was scattered across
    ``server.py``, ``indexer.py``, and ``_generate_embeddings_worker``: we
    install ``ssl._create_unverified_context`` as the default HTTPS context
    AND blank out the env vars that ``requests``/``httpx``/``huggingface_hub``
    consult for cert verification. Keeping all of this behind one env-var
    check means there is exactly one place to audit.

    Env-var propagation note: when the indexer's ``_generate_embeddings_worker``
    runs in a child process (``multiprocessing.spawn``), the env vars set
    here are inherited automatically. The worker calls this function again
    so the ``ssl`` mutation is re-applied in the child interpreter.
    """
    if os.environ.get(INSECURE_ENV_VAR) != "1":
        return False
    log.warning(INSECURE_WARNING, INSECURE_ENV_VAR)
    ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SLF001
    os.environ["CURL_CA_BUNDLE"] = ""
    # Mirror the embedding-worker's HF/HTTPX bypass too — keeps a single
    # opt-in surface so users don't get half-bypassed networking.
    os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
    os.environ["HTTPX_SSL_VERIFY"] = "0"
    os.environ["SSL_CERT_FILE"] = ""
    return True


def configure_cert_bundle(path: str) -> None:
    """Point both REQUESTS_CA_BUNDLE and SSL_CERT_FILE at the given file.

    Use this for corporate CA bundles. TLS verification stays ON.

    Raises:
        FileNotFoundError: when ``path`` does not resolve to an existing file.
    """
    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"--cert-bundle path does not exist: {abs_path}")
    os.environ["REQUESTS_CA_BUNDLE"] = abs_path
    os.environ["SSL_CERT_FILE"] = abs_path
    log.info("Using corporate CA bundle: %s", abs_path)
