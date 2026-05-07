"""Tests for the Phase 4d.1 ``argpartition`` top-k selection in
``server._semantic_search``.

The previous implementation used ``np.argsort(scores)[-top_k:][::-1]``, which
sorts the entire score vector (O(N log N)). The new implementation uses
``np.argpartition`` to pick the top-k in O(N) and then sorts only those k
indices (O(k log k)). The output ranking and identity must remain
identical, plus we have to handle the ``top_k > N`` edge case that
``argpartition`` would otherwise raise on.

These tests stub out ``EMBEDDING_MATRIX`` / ``EMBEDDING_IDS`` and the
embedding model so they run without any indexed DB or sentence-transformers
download. They exercise the math directly.
"""
from __future__ import annotations

from typing import List, Tuple
from unittest.mock import patch

import numpy as np
import pytest

from ios_graphrag import server


class _StubModel:
    """Minimal stand-in for SentenceTransformer.

    We control what ``encode`` returns so we can deterministically pin
    which row of the embedding matrix is the closest match. The model is
    only used for the query side; the matrix rows are fixed by each test.
    """

    def __init__(self, query_vec: np.ndarray) -> None:
        self._q = query_vec.astype(np.float32)

    def encode(self, queries: List[str], convert_to_numpy: bool = True) -> np.ndarray:
        # ``_semantic_search`` indexes ``[0]`` and renormalizes, so the
        # exact magnitude doesn't matter — only the direction.
        return np.tile(self._q, (len(queries), 1))


def _install_matrix(rows: np.ndarray, ids: List[Tuple[int, str, str]]) -> None:
    """Drop a normalized matrix into the server's globals for the test."""
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    server.EMBEDDING_MATRIX = (rows / (norms + 1e-10)).astype(np.float32)
    server.EMBEDDING_IDS = ids


@pytest.fixture(autouse=True)
def _disable_reload(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_semantic_search`` calls ``_reload_if_stale`` first; in this test
    we drive the module globals directly and don't want a stale-mtime
    check to overwrite them. Stub it to a no-op."""
    monkeypatch.setattr(server, "_reload_if_stale", lambda: None)


def test_argpartition_returns_top_k_in_descending_score_order() -> None:
    """top_k=5 against a 50-vector matrix returns 5 results, descending."""
    rng = np.random.default_rng(seed=1234)
    # 50 random unit vectors in 8-D. We want the highest cosine
    # similarities concentrated where we put them.
    rows = rng.normal(size=(50, 8)).astype(np.float32)
    # Pick a known query direction. Make rows 7, 19, 31, 42, 5 line up
    # increasingly well with it so we know the expected top-5 ranking.
    q = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    rows[5] = np.array([0.50, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    rows[42] = np.array([0.60, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    rows[31] = np.array([0.70, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    rows[19] = np.array([0.80, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    rows[7] = np.array([0.99, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    ids = [(i, f"/f/{i}.swift", f"sym{i}") for i in range(50)]
    _install_matrix(rows, ids)

    with patch.object(server, "ensure_model", return_value=_StubModel(q)):
        out = server._semantic_search(query="anything", top_k=5)

    assert "error" not in out, out
    results = out["results"]
    assert len(results) == 5, f"expected 5 results, got {len(results)}"
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), (
        f"scores not descending: {scores!r}"
    )
    # The top-5 must be the rows we pinned (ordering across them
    # confirmed by the descending check above).
    expected_ids = {7, 19, 31, 42, 5}
    got_ids = {r["node_id"] for r in results}
    assert got_ids == expected_ids, (
        f"top-5 node_ids wrong; expected {expected_ids}, got {got_ids}"
    )


def test_argpartition_handles_top_k_exceeding_corpus_size() -> None:
    """top_k=100 against a 10-vector matrix returns all 10 (no crash).

    The previous ``argsort`` form silently clipped via slice semantics,
    but ``argpartition(arr, -k)`` raises ``ValueError`` if ``k > len(arr)``.
    We guard with ``min(top_k, scores.shape[0])``; this test pins that.
    """
    rng = np.random.default_rng(seed=99)
    rows = rng.normal(size=(10, 4)).astype(np.float32)
    ids = [(i, f"/f/{i}.swift", f"sym{i}") for i in range(10)]
    _install_matrix(rows, ids)

    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    with patch.object(server, "ensure_model", return_value=_StubModel(q)):
        # top_k is constrained to 1..50 by the Pydantic Field, but the
        # logic must still cope when the matrix is smaller than top_k.
        out = server._semantic_search(query="anything", top_k=50)

    assert "error" not in out, out
    results = out["results"]
    assert len(results) == 10, f"expected all 10 rows, got {len(results)}"
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), (
        f"scores not descending under k>N: {scores!r}"
    )
    # Sanity: every node_id appears exactly once.
    assert sorted(r["node_id"] for r in results) == list(range(10))
