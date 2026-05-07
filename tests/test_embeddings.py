"""Tests for the in-process embedding refactor (Phase 4b).

Phase 4b moved ``embed_signatures`` from a per-call
``ProcessPoolExecutor(spawn=True)`` worker to an in-process call backed by
a module-level cached ``SentenceTransformer`` singleton. These tests pin
the new behavior:

* ``embed_signatures`` does NOT spawn a child process.
* The model is loaded at most once per Python process (cached on a
  module global), so repeated invocations reuse the same object.
* The model-resolution chain (``$IOS_GRAPHRAG_MODEL_DIR`` → default
  cache → HF id) still flows through ``_resolve_model_path`` and into
  the ``SentenceTransformer`` constructor.

To keep these tests cheap and offline-safe, we monkeypatch
``indexer.SentenceTransformer`` with a fake that records its construction
arguments and returns deterministic toy embeddings. No real model is
downloaded or loaded; the tests run in milliseconds.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
from unittest import mock

import numpy as np
import pytest

from ios_graphrag import indexer


class _FakeModel:
    """Stand-in for ``sentence_transformers.SentenceTransformer``.

    Records constructor arguments and ``encode`` calls so tests can assert
    on call counts and ordering. ``encode`` returns a deterministic
    ``(N, 8)`` matrix of zeros so callers can still call ``.astype`` /
    ``.shape`` like the real thing.
    """

    construct_count = 0
    last_args: Optional[tuple] = None
    last_kwargs: Optional[dict] = None

    def __init__(self, model_path: str, **kwargs):
        type(self).construct_count += 1
        type(self).last_args = (model_path,)
        type(self).last_kwargs = kwargs
        self.model_path = model_path
        self.encode_count = 0

    def to(self, device: str) -> "_FakeModel":
        # Mirror sentence-transformers' ``.to(device)`` contract: return
        # self so a hypothetical caller doing ``m.to('mps').encode(...)``
        # still works. We don't care about the device for testing.
        return self

    def encode(self, texts: List[str], convert_to_numpy: bool = True, **kwargs) -> np.ndarray:
        self.encode_count += 1
        return np.zeros((len(texts), 8), dtype=np.float32)


@pytest.fixture(autouse=True)
def _reset_model_cache():
    """Clear the module-level model cache before AND after each test.

    Without this, a test that populates ``_MODEL_CACHE`` would leak into
    its neighbor and silently break the "load only once" invariant tests.
    """
    indexer._MODEL_CACHE = None
    indexer._MODEL_CACHE_PATH = None
    _FakeModel.construct_count = 0
    _FakeModel.last_args = None
    _FakeModel.last_kwargs = None
    yield
    indexer._MODEL_CACHE = None
    indexer._MODEL_CACHE_PATH = None


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch):
    """Replace the real SentenceTransformer with ``_FakeModel`` for cheap tests."""
    monkeypatch.setattr(indexer, "SentenceTransformer", _FakeModel)
    # Force ``_resolve_model_path`` to return a stable string regardless of the
    # user's environment. We pin it to the HF identifier so the resolver's
    # legacy in-repo `models/` snapshot fallback is still exercised.
    monkeypatch.delenv("IOS_GRAPHRAG_MODEL_DIR", raising=False)
    monkeypatch.setattr(
        indexer, "DEFAULT_LOCAL_MODEL_DIR", Path("/tmp/definitely-not-a-real-path-p4b")
    )
    return _FakeModel


def test_embed_signatures_runs_in_process(fake_model, monkeypatch: pytest.MonkeyPatch) -> None:
    """``embed_signatures`` must NOT spawn a child process.

    The Phase 4b refactor's whole point is that there is no longer a
    ``ProcessPoolExecutor`` between the indexer and the embedding model.
    Spy on ``multiprocessing.get_context`` and assert it isn't called.
    """
    import multiprocessing as _mp

    spy = mock.MagicMock(wraps=_mp.get_context)
    monkeypatch.setattr(_mp, "get_context", spy)

    out = indexer.embed_signatures(["func a()", "func b()", "class C"])

    assert len(out) == 3
    assert out[0].shape == (8,)
    spy.assert_not_called()


def test_embed_signatures_caches_model(fake_model) -> None:
    """Two back-to-back ``embed_signatures`` calls must reuse the same model.

    The Phase 4b cache lives on ``indexer._MODEL_CACHE``; the
    ``construct_count`` on our fake mirrors how many times the
    ``SentenceTransformer`` constructor ran. After two calls the count
    must be 1 — anything else means the cache is broken.
    """
    indexer.embed_signatures(["func a()"])
    assert _FakeModel.construct_count == 1

    cached_after_first = indexer._MODEL_CACHE
    assert cached_after_first is not None

    indexer.embed_signatures(["func b()", "func c()"])
    assert _FakeModel.construct_count == 1, (
        "Second call should reuse the cached model, not reload it"
    )
    assert indexer._MODEL_CACHE is cached_after_first


def test_embed_signatures_empty_list_short_circuits(fake_model) -> None:
    """Empty input must NOT trigger a model load — the early return at
    the top of ``embed_signatures`` is the cheap path the watcher daemon
    relies on for "no changes" incremental runs.
    """
    out = indexer.embed_signatures([])
    assert out == []
    assert _FakeModel.construct_count == 0
    assert indexer._MODEL_CACHE is None


def test_embed_signatures_uses_resolve_model_path(
    fake_model, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The in-process loader still threads through ``_resolve_model_path``.

    Set ``IOS_GRAPHRAG_MODEL_DIR`` to a fabricated checkpoint and verify
    the fake model is constructed with that path — proving the env-var
    override still wins over the HF identifier.
    """
    fake_checkpoint = tmp_path / "p4b-resolved-checkpoint"
    fake_checkpoint.mkdir()
    (fake_checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
    (fake_checkpoint / "model.safetensors").write_bytes(b"\x00")
    monkeypatch.setenv("IOS_GRAPHRAG_MODEL_DIR", str(fake_checkpoint))

    indexer.embed_signatures(["sig"])

    assert _FakeModel.construct_count == 1
    assert _FakeModel.last_args == (str(fake_checkpoint),)
    # `trust_remote_code=True` must still flow through; nomic-embed-text-v1.5
    # depends on it for the model's custom code path.
    assert _FakeModel.last_kwargs == {"trust_remote_code": True}


def test_embed_signatures_invalidates_cache_on_path_change(
    fake_model, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If the resolved model path changes between calls, the cache must
    invalidate and the model reload.

    This is a defensive contract: the indexer doesn't currently flip env
    vars mid-process, but the watcher daemon (Phase 4c) might if a user
    edits their config. We verify the singleton respects path changes.
    """
    indexer.embed_signatures(["x"])
    assert _FakeModel.construct_count == 1

    fake_checkpoint = tmp_path / "switched-checkpoint"
    fake_checkpoint.mkdir()
    (fake_checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
    (fake_checkpoint / "model.safetensors").write_bytes(b"\x00")
    monkeypatch.setenv("IOS_GRAPHRAG_MODEL_DIR", str(fake_checkpoint))

    indexer.embed_signatures(["y"])
    assert _FakeModel.construct_count == 2, (
        "Path change should invalidate cache and trigger reload"
    )
    assert _FakeModel.last_args == (str(fake_checkpoint),)


def test_no_processpoolexecutor_in_embed_signatures_body() -> None:
    """Static guard: the BODY of ``embed_signatures`` must not reference
    a per-call subprocess.

    Inspect the AST so docstring mentions of ``ProcessPoolExecutor`` (which
    are kept intentionally to explain the refactor) don't trip this test.
    A future regression that re-introduces ``multiprocessing.spawn`` /
    ``ProcessPoolExecutor`` in the executable code path will fail loudly.
    """
    import ast
    import inspect
    import textwrap

    source = textwrap.dedent(inspect.getsource(indexer.embed_signatures))
    tree = ast.parse(source)
    func = tree.body[0]
    assert isinstance(func, ast.FunctionDef)
    # Strip the docstring node, then dump the rest as text for substring scan.
    body = func.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    body_src = "\n".join(ast.unparse(node) for node in body)

    assert "ProcessPoolExecutor" not in body_src, (
        "Phase 4b removed the embedding worker subprocess; do not re-add"
    )
    assert "multiprocessing" not in body_src, (
        "Phase 4b: embedding runs in-process; multiprocessing belongs nowhere here"
    )


def test_indexer_does_not_export_generate_embeddings_worker() -> None:
    """``_generate_embeddings_worker`` was removed in Phase 4b.

    If a future refactor re-adds it, the function will need fresh review
    against the audit findings — this test forces that conversation.
    """
    assert not hasattr(indexer, "_generate_embeddings_worker"), (
        "Phase 4b removed _generate_embeddings_worker; "
        "re-introducing it requires a fresh contamination audit"
    )
