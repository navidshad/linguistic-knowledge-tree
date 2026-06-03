"""Semantic evidence->node mapping (Phase 6 / RQ2): embed free text, match to nodes.

Generalizes the Phase-5 rule-based morphosyntactic mapper (``adapters.slam_mapping``)
to *unlabeled* text via local BERT embeddings, with a K-BERT-style knowledge-injected
variant. Heavy deps (sentence-transformers) are optional + lazy; tests use the
deterministic ``HashingEmbedder`` so nothing is downloaded.
"""
from .embedder import (
    Embedder,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    default_embedder,
)
from .node_vectors import (
    DEFAULT_VECTORS_PATH,
    NodeVectors,
    build_node_vectors,
    load_node_vectors,
    node_text,
    save_node_vectors,
)
from .mapper import SemanticMapper, SemanticMatch, default_mapper, reset_default_caches

__all__ = [
    "Embedder",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "default_embedder",
    "NodeVectors",
    "build_node_vectors",
    "load_node_vectors",
    "save_node_vectors",
    "node_text",
    "DEFAULT_VECTORS_PATH",
    "SemanticMapper",
    "SemanticMatch",
    "default_mapper",
    "reset_default_caches",
]
