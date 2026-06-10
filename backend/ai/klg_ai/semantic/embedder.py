"""Text embedders behind one interface, plus a deterministic dependency-free fake.

The semantic mapper (``klg_ai.semantic.mapper``) turns free text into concept-node
evidence by comparing a text embedding to per-node embeddings. Three concerns are
kept separate here:

* ``Embedder`` — the protocol: ``encode(texts) -> (n, dim)`` L2-normalized array.
* ``SentenceTransformerEmbedder`` — the real backend (sentence-transformers
  MiniLM). The heavy deps (sentence-transformers -> transformers -> torch) are
  imported lazily on first use, mirroring ``propagation.py``'s torch import, so
  merely importing this module costs nothing. Pinned to **CPU** so the vectors are
  bit-stable across machines (MPS/GPU introduce small float drift) — the committed
  node-vector artifact must reproduce.
* ``HashingEmbedder`` — a deterministic, dependency-free fake: hashes character
  n-grams into a fixed-width vector. It has the *same* interface as the real one,
  so tests and CI exercise the exact mapper code path with zero model download. It
  is a real (low-quality) embedder, not a mock of the mapper.

``default_embedder()`` returns the real backend when sentence-transformers is
installed, falls back to the fake otherwise, and honors ``KLG_EMBEDDER=hash`` to
force the fake (used by the test suite + CI).
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Encodes texts into L2-normalized row vectors of a fixed dimension."""

    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an ``(len(texts), dim)`` float32 array of L2-normalized rows."""
        ...


def _l2_normalize(m: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization; all-zero rows are left as zeros (norm -> 1)."""
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (m / norms).astype(np.float32)


class HashingEmbedder:
    """Deterministic, dependency-free embedder for tests/CI (no model download).

    Hashes character 3-grams of each lower-cased text into ``dim`` signed buckets,
    then L2-normalizes. Captures coarse lexical overlap — enough to exercise the
    mapper's matching / threshold / knowledge-injection logic deterministically.
    Vector *quality* is poor by design; zero deps + reproducibility are the point.
    """

    name = "hashing"

    def __init__(self, dim: int = 64, ngram: int = 3) -> None:
        self.dim = dim
        self.ngram = ngram

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        s = f" {text.lower().strip()} "
        n = self.ngram
        grams = [s[i : i + n] for i in range(len(s) - n + 1)] if len(s) >= n else [s]
        for g in grams:
            h = int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return v

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _l2_normalize(np.stack([self._vec(t) for t in texts]))


class SentenceTransformerEmbedder:
    """Real backend: sentence-transformers MiniLM, lazily loaded, pinned to CPU."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.name = model_name
        self._model = None
        self._dim: int | None = None

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy, heavy

            self._model = SentenceTransformer(self.name, device="cpu")
            # `get_embedding_dimension` (new) supersedes `get_sentence_embedding_dimension`.
            get_dim = getattr(
                self._model, "get_embedding_dimension", self._model.get_sentence_embedding_dimension
            )
            self._dim = int(get_dim())
        return self._model

    @property
    def dim(self) -> int:
        self._ensure()
        return int(self._dim)  # type: ignore[arg-type]

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        emb = self._ensure().encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        return emb.astype(np.float32)


def default_embedder() -> Embedder:
    """Real MiniLM backend if available, else the deterministic fake.

    ``KLG_EMBEDDER=hash`` forces the fake (test suite + CI). Otherwise we use the
    real backend when ``sentence_transformers`` is importable (checked without
    importing the heavy module) and fall back to the fake if it is absent, so the
    engine/server never hard-fail on a missing optional dependency.
    """
    if os.environ.get("KLG_EMBEDDER", "").lower() == "hash":
        return HashingEmbedder()
    if importlib.util.find_spec("sentence_transformers") is not None:
        return SentenceTransformerEmbedder()
    return HashingEmbedder()
