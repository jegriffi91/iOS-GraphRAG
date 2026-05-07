"""Tests for the embedding-model path resolver in ``ios_graphrag.indexer``.

The resolver implements the chain documented in the production roadmap (P3.3):

1. ``$IOS_GRAPHRAG_MODEL_DIR`` if set and the directory contains weights.
2. ``~/.cache/ios-graphrag/models/nomic-embed-text-v1.5/`` if it contains
   weights.
3. The Hugging Face identifier ``"nomic-ai/nomic-embed-text-v1.5"`` (which
   triggers HF cache or download).

These tests fabricate a minimal "weights present" layout (config.json plus
model.safetensors) in tmpdirs, monkeypatch the relevant env var or the
default-cache constant, and assert the resolver returns the right path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ios_graphrag import indexer


def _make_fake_checkpoint(directory: Path) -> Path:
    """Materialize a minimal sentence-transformers-looking checkpoint.

    Just needs ``config.json`` plus one of ``model.safetensors`` /
    ``pytorch_model.bin`` so ``_has_model_weights`` accepts it.
    """
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text("{}\n", encoding="utf-8")
    (directory / "model.safetensors").write_bytes(b"\x00")
    return directory


def test_model_path_resolves_to_env_var_when_set_and_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Env-var override wins over the default cache and the HF identifier."""
    fake_checkpoint = _make_fake_checkpoint(tmp_path / "custom-model")
    monkeypatch.setenv("IOS_GRAPHRAG_MODEL_DIR", str(fake_checkpoint))
    # Make sure the env path beats anything else: point the default cache at a
    # tmp dir that does NOT exist, so a bug that prefers the default would
    # fall through to the HF id rather than masquerade as success.
    monkeypatch.setattr(
        indexer,
        "DEFAULT_LOCAL_MODEL_DIR",
        tmp_path / "definitely-not-here",
    )

    resolved = indexer._resolve_model_path()

    assert resolved == str(fake_checkpoint)


def test_model_path_falls_back_to_default_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When env var is unset, a populated default cache wins over HF id."""
    monkeypatch.delenv("IOS_GRAPHRAG_MODEL_DIR", raising=False)
    fake_default = _make_fake_checkpoint(tmp_path / "default-cache" / "nomic")
    monkeypatch.setattr(indexer, "DEFAULT_LOCAL_MODEL_DIR", fake_default)

    resolved = indexer._resolve_model_path()

    assert resolved == str(fake_default)


def test_model_path_returns_hf_id_when_nothing_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No env var + empty local cache -> HF identifier (lets HF cache/download)."""
    monkeypatch.delenv("IOS_GRAPHRAG_MODEL_DIR", raising=False)
    # Point the default cache at a directory that exists but has no weights.
    empty_cache = tmp_path / "empty-cache" / "nomic"
    empty_cache.mkdir(parents=True)
    monkeypatch.setattr(indexer, "DEFAULT_LOCAL_MODEL_DIR", empty_cache)

    resolved = indexer._resolve_model_path()

    assert resolved == indexer.MODEL_NAME == "nomic-ai/nomic-embed-text-v1.5"
