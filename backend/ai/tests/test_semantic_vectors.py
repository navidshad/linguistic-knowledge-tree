"""Embedder seam + node-vector artifact (Phase 6 / RQ2, Step 1).

Uses the deterministic ``HashingEmbedder`` throughout, so nothing is downloaded
and the assertions are stable. Exercises the same code path the real MiniLM
backend uses — only the vector quality differs.
"""
import os

import numpy as np
import pytest

from klg_ai.loader import load_map
from klg_ai.semantic import (
    HashingEmbedder,
    NodeVectors,
    build_node_vectors,
    default_embedder,
    load_node_vectors,
    node_text,
    save_node_vectors,
)
from klg_ai.semantic.node_vectors import _category_labels


# --- embedder --------------------------------------------------------------

def test_hashing_embedder_shape_and_normalization():
    emb = HashingEmbedder(dim=32)
    m = emb.encode(["she is reading", "the books"])
    assert m.shape == (2, 32)
    np.testing.assert_allclose(np.linalg.norm(m, axis=1), 1.0, atol=1e-6)


def test_hashing_embedder_is_deterministic():
    emb = HashingEmbedder()
    np.testing.assert_array_equal(emb.encode(["present perfect"]), emb.encode(["present perfect"]))


def test_hashing_embedder_empty():
    assert HashingEmbedder(dim=8).encode([]).shape == (0, 8)


def test_default_embedder_env_forces_hash(monkeypatch):
    monkeypatch.setenv("KLG_EMBEDDER", "hash")
    assert isinstance(default_embedder(), HashingEmbedder)


# --- node vectors ----------------------------------------------------------

def test_node_text_recipe():
    m = load_map()
    cats = _category_labels(m)
    txt = node_text(m.node("present_simple"), cats)
    assert "Present Simple" in txt and "category:" in txt
    assert m.node("present_simple").description in txt


def test_build_node_vectors_matches_map():
    m = load_map()
    nv = build_node_vectors(HashingEmbedder(dim=48), m)
    assert nv.node_ids == [n.id for n in m.nodes]
    assert set(nv.node_ids) == m.node_ids
    assert nv.matrix.shape == (len(m.nodes), 48)
    assert nv.matrix_kbert.shape == nv.matrix.shape
    np.testing.assert_allclose(np.linalg.norm(nv.matrix, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(nv.matrix_kbert, axis=1), 1.0, atol=1e-5)


def test_kbert_injection_changes_some_vectors():
    # Nodes with prerequisite/dependent neighbours get a different (injected) vector.
    nv = build_node_vectors(HashingEmbedder(dim=48), load_map())
    assert not np.allclose(nv.matrix, nv.matrix_kbert)
    assert nv.select(knowledge_injection=False) is nv.matrix
    assert nv.select(knowledge_injection=True) is nv.matrix_kbert


def test_save_load_roundtrip(tmp_path):
    nv = build_node_vectors(HashingEmbedder(dim=16), load_map())
    path = tmp_path / "nv.npz"
    save_node_vectors(nv, path)
    assert path.exists() and path.with_suffix(".meta.json").exists()
    loaded = load_node_vectors(path)
    assert loaded.node_ids == nv.node_ids
    np.testing.assert_array_equal(loaded.matrix, nv.matrix)
    np.testing.assert_array_equal(loaded.matrix_kbert, nv.matrix_kbert)


def test_load_rejects_stale_artifact(tmp_path):
    nv = build_node_vectors(HashingEmbedder(dim=16), load_map())
    stale = NodeVectors(
        node_ids=nv.node_ids[:-1],
        matrix=nv.matrix[:-1],
        matrix_kbert=nv.matrix_kbert[:-1],
        model_name=nv.model_name,
        dim=nv.dim,
        text_recipe=nv.text_recipe,
    )
    path = tmp_path / "stale.npz"
    save_node_vectors(stale, path)
    with pytest.raises(ValueError, match="stale"):
        load_node_vectors(path)
