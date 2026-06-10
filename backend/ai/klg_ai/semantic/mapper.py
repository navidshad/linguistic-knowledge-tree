"""Map free text to concept nodes by embedding similarity (Phase 6 / RQ2).

The semantic analogue of the rule-based ``adapters.slam_mapping``: instead of
reading morphosyntactic tags, embed the text and match it against per-node vectors
(``NodeVectors``). Two arms, toggled by ``knowledge_injection`` — this single flag
*is* the RQ2 ablation:

* off — **BERT**: dot the text vector against the vanilla node matrix.
* on  — **K-BERT**: dot against the graph-injected node matrix. Additionally, when
  a live ``active_nodes`` set is supplied (the chat use case only), re-rank ties
  toward nodes whose prerequisites the learner already knows. The offline RQ2 eval
  never passes ``active_nodes``, so that secondary bias stays out of the scored
  comparison and the ablation remains single-variable (which node matrix).

Selection = the top-``top_k`` node ids whose cosine ≥ ``threshold`` (possibly
empty, mirroring the rule mapper dropping function words). ``scores`` always
reports the raw cosine, never the re-rank bias.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..loader import SyntaxMap, load_map
from .embedder import Embedder, default_embedder
from .node_vectors import NodeVectors, build_node_vectors, load_node_vectors


@dataclass(frozen=True)
class SemanticMatch:
    """The nodes one text is evidence about, with their cosine confidences."""

    node_ids: tuple[str, ...]
    scores: dict[str, float]  # node_id -> cosine similarity


class SemanticMapper:
    """Embedding text->node mapper; ``knowledge_injection`` toggles the RQ2 arm."""

    def __init__(
        self,
        embedder: Embedder,
        node_vectors: NodeVectors,
        *,
        threshold: float = 0.35,
        top_k: int = 3,
        knowledge_injection: bool = False,
        ki_weight: float = 0.25,
        syntax_map: SyntaxMap | None = None,
    ) -> None:
        self.embedder = embedder
        self.nv = node_vectors
        self.threshold = threshold
        self.top_k = top_k
        self.knowledge_injection = knowledge_injection
        self.ki_weight = ki_weight
        self._matrix = node_vectors.select(knowledge_injection)  # (n_nodes, dim)
        self._ids = node_vectors.node_ids
        self._syntax_map = syntax_map
        self._prereqs: dict[str, set[str]] | None = None

    # -- selection ----------------------------------------------------------

    def _select(self, sims: np.ndarray, rank: np.ndarray | None = None) -> SemanticMatch:
        """Top-k node ids with cosine ≥ threshold, ordered by ``rank`` (default cosine)."""
        rank = sims if rank is None else rank
        eligible = [i for i in range(len(sims)) if sims[i] >= self.threshold]
        eligible.sort(key=lambda i: rank[i], reverse=True)
        picked = eligible[: self.top_k]
        return SemanticMatch(
            node_ids=tuple(self._ids[i] for i in picked),
            scores={self._ids[i]: float(sims[i]) for i in picked},
        )

    def _prereq_map(self) -> dict[str, set[str]]:
        if self._prereqs is None:
            m = self._syntax_map or load_map()
            pm: dict[str, set[str]] = {n.id: set() for n in m.nodes}
            for e in m.edges:
                pm[e.target].add(e.source)
            self._prereqs = pm
        return self._prereqs

    def _readiness_bias(self, active_nodes: set[str]) -> np.ndarray:
        """Per-node bias ∝ fraction of its prerequisites already known (chat only)."""
        pm = self._prereq_map()
        bias = np.zeros(len(self._ids), dtype=np.float32)
        for i, nid in enumerate(self._ids):
            pr = pm.get(nid, set())
            if pr:
                bias[i] = self.ki_weight * (len(pr & active_nodes) / len(pr))
        return bias

    # -- mapping ------------------------------------------------------------

    def map_text(self, text: str, *, active_nodes: set[str] | None = None) -> SemanticMatch:
        """Concept nodes a free text is evidence about."""
        if not text or not text.strip():
            return SemanticMatch((), {})
        v = self.embedder.encode([text])[0]
        sims = self._matrix @ v
        rank = sims
        if self.knowledge_injection and active_nodes:
            rank = sims + self._readiness_bias(set(active_nodes))
        return self._select(sims, rank)

    def map_token_in_context(
        self, token: str, sentence: str, *, active_nodes: set[str] | None = None
    ) -> SemanticMatch:
        """Map a token, embedding its *sentence* with the token span marked.

        A bare token ("is", "the") is semantically empty; the marked sentence
        ("She [ is ] reading") is not. Marks the first occurrence.
        """
        if token and sentence and token in sentence:
            marked = sentence.replace(token, f"[ {token} ]", 1)
        else:
            marked = f"{sentence} [ {token} ]".strip()
        return self.map_text(marked, active_nodes=active_nodes)

    def map_exercise(self, ex, *, active_nodes: set[str] | None = None) -> dict[str, "SemanticMatch"]:
        """Map every token in a SLAM exercise; shape-compatible with the rule mapper.

        Returns ``{instance_id: SemanticMatch}`` for tokens matching ≥1 node (others
        omitted). Each token is embedded as the reconstructed sentence with its own
        span marked (robust to duplicate surface forms); the encode is batched.
        """
        tokens = list(ex.tokens)
        words = [t.token for t in tokens]
        marked = []
        for i in range(len(words)):
            parts = list(words)
            parts[i] = f"[ {parts[i]} ]"
            marked.append(" ".join(parts))
        if not marked:
            return {}
        vecs = self.embedder.encode(marked)  # (n_tokens, dim)
        bias = (
            self._readiness_bias(set(active_nodes))
            if (self.knowledge_injection and active_nodes)
            else None
        )
        out: dict[str, SemanticMatch] = {}
        for i, tok in enumerate(tokens):
            sims = self._matrix @ vecs[i]
            match = self._select(sims, None if bias is None else sims + bias)
            if match.node_ids:
                out[tok.instance_id] = match
        return out


# --- shared defaults for the server + eval ---------------------------------
# One embedder + one node-vector pair per process, picked consistently so the
# text vectors and the node matrix always share a dimension.


@lru_cache(maxsize=1)
def _shared_embedder() -> Embedder:
    return default_embedder()


def _node_vectors_for(embedder: Embedder) -> NodeVectors:
    """Committed artifact when it matches the embedder; else build in-memory.

    The deterministic ``HashingEmbedder`` (dim 64) is incompatible with the
    MiniLM-built committed artifact (dim 384), so it always rebuilds in-memory.
    """
    if getattr(embedder, "name", "") == "hashing":
        return build_node_vectors(embedder)
    try:
        nv = load_node_vectors()
    except FileNotFoundError:
        return build_node_vectors(embedder)
    return nv if nv.dim == embedder.dim else build_node_vectors(embedder)


@lru_cache(maxsize=1)
def _shared_node_vectors() -> NodeVectors:
    return _node_vectors_for(_shared_embedder())


def default_mapper(
    *,
    knowledge_injection: bool = False,
    threshold: float = 0.35,
    top_k: int = 3,
    ki_weight: float = 0.25,
    syntax_map: SyntaxMap | None = None,
) -> SemanticMapper:
    """A mapper on the process-shared embedder + node vectors (server/eval entry point)."""
    return SemanticMapper(
        _shared_embedder(),
        _shared_node_vectors(),
        knowledge_injection=knowledge_injection,
        threshold=threshold,
        top_k=top_k,
        ki_weight=ki_weight,
        syntax_map=syntax_map,
    )


def reset_default_caches() -> None:
    """Clear the shared embedder/vectors caches (tests that flip ``KLG_EMBEDDER``)."""
    _shared_embedder.cache_clear()
    _shared_node_vectors.cache_clear()
