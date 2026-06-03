"""Per-node embedding vectors — the artifact the mapper matches text against.

Each concept node is embedded from a short text built from its ``label`` +
``description`` (+ its category label). Two matrices come from the *same* embedder:

* ``matrix``       — vanilla **BERT**: each node embedded from its own text.
* ``matrix_kbert`` — **K-BERT** knowledge injection: each node's vector is blended
  with an embedding of its immediate prerequisite parents' and dependent children's
  labels (1-hop graph context), then re-normalized. This is the single variable the
  RQ2 ablation toggles (``SemanticMapper(knowledge_injection=...)``).

The vectors are precomputed offline and committed (``docs/phase6/node_vectors.npz``)
so the server and CI never download or run a model to *match* — mirroring the
committed ``docs/phase5/results.json``. ``load_node_vectors`` fails fast if the
artifact's node set has drifted from the live map.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..loader import Node, SyntaxMap, load_map
from .embedder import Embedder, _l2_normalize

# repo_root/docs/phase6/node_vectors.npz
# (this file: backend/ai/klg_ai/semantic/node_vectors.py -> parents[4] == repo root)
DEFAULT_VECTORS_PATH = Path(__file__).resolve().parents[4] / "docs" / "phase6" / "node_vectors.npz"


@dataclass(frozen=True)
class NodeVectors:
    """The two per-node matrices (row order == ``node_ids``) + provenance."""

    node_ids: list[str]
    matrix: np.ndarray  # (n_nodes, dim), L2-normalized — vanilla BERT
    matrix_kbert: np.ndarray  # (n_nodes, dim), L2-normalized — graph-injected
    model_name: str
    dim: int
    text_recipe: str

    def index(self) -> dict[str, int]:
        return {nid: i for i, nid in enumerate(self.node_ids)}

    def select(self, knowledge_injection: bool) -> np.ndarray:
        """The matrix the mapper dots against for the chosen ablation arm."""
        return self.matrix_kbert if knowledge_injection else self.matrix


def _category_labels(m: SyntaxMap) -> dict[str, str]:
    return {c["id"]: c.get("label", c["id"]) for c in m.categories}


def node_text(node: Node, cat_labels: dict[str, str], *, include_category: bool = True) -> str:
    """The text embedded for a node: ``'<label>. <description> (category: <cat>)'``."""
    text = f"{node.label}. {node.description}".strip()
    if include_category:
        text = f"{text} (category: {cat_labels.get(node.category, node.category)})"
    return text


def _neighbor_text(node: Node, m: SyntaxMap) -> str:
    """1-hop prerequisite-graph context: parent (prereq) + child (dependent) labels."""
    parents = [m.node(e.source).label for e in m.edges if e.target == node.id]
    children = [m.node(e.target).label for e in m.edges if e.source == node.id]
    parts: list[str] = []
    if parents:
        parts.append("prerequisites: " + "; ".join(parents))
    if children:
        parts.append("leads to: " + "; ".join(children))
    return ". ".join(parts)


def build_node_vectors(
    embedder: Embedder,
    m: SyntaxMap | None = None,
    *,
    ki_weight: float = 0.25,
    include_category: bool = True,
) -> NodeVectors:
    """Embed every node into the vanilla + K-BERT (graph-injected) matrices."""
    m = m or load_map()
    cats = _category_labels(m)
    nodes = m.nodes

    own = embedder.encode([node_text(n, cats, include_category=include_category) for n in nodes])
    neigh_texts = [_neighbor_text(n, m) for n in nodes]
    neigh = embedder.encode([t or "" for t in neigh_texts])

    blended = own.copy()
    for i, t in enumerate(neigh_texts):
        if t:  # only inject where the node actually has graph neighbours
            blended[i] = ki_weight * neigh[i] + (1.0 - ki_weight) * own[i]

    return NodeVectors(
        node_ids=[n.id for n in nodes],
        matrix=own,  # encode() already L2-normalizes its rows
        matrix_kbert=_l2_normalize(blended),
        model_name=getattr(embedder, "name", "unknown"),
        dim=int(own.shape[1]),
        text_recipe=(
            f"'label. description (category: X)'; kbert = blend(own, "
            f"1-hop prereq/child labels), ki_weight={ki_weight}"
        ),
    )


def save_node_vectors(nv: NodeVectors, path: str | Path = DEFAULT_VECTORS_PATH) -> Path:
    """Write the ``.npz`` (self-contained) + a human-readable ``.meta.json`` sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        matrix=nv.matrix.astype(np.float32),
        matrix_kbert=nv.matrix_kbert.astype(np.float32),
        node_ids=np.array(nv.node_ids),  # unicode array — no pickle needed to load
        model_name=np.array(nv.model_name),
        text_recipe=np.array(nv.text_recipe),
    )
    path.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "model_name": nv.model_name,
                "dim": nv.dim,
                "n_nodes": len(nv.node_ids),
                "text_recipe": nv.text_recipe,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def load_node_vectors(path: str | Path = DEFAULT_VECTORS_PATH, *, check_map: bool = True) -> NodeVectors:
    """Load the committed artifact; fail fast if its node set has drifted from the map."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"node-vector artifact not found: {path}. Build it with "
            f"`python -m klg_ai.semantic.build_vectors`."
        )
    data = np.load(path, allow_pickle=False)
    node_ids = [str(x) for x in data["node_ids"].tolist()]
    nv = NodeVectors(
        node_ids=node_ids,
        matrix=data["matrix"],
        matrix_kbert=data["matrix_kbert"],
        model_name=str(data["model_name"]),
        dim=int(data["matrix"].shape[1]),
        text_recipe=str(data["text_recipe"]),
    )
    if check_map:
        live = load_map().node_ids
        if set(node_ids) != live:
            raise ValueError(
                f"node-vector artifact is stale vs the live map (missing "
                f"{sorted(live - set(node_ids))}, extra {sorted(set(node_ids) - live)}). Rebuild it."
            )
    return nv
