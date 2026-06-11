"""GNN propagation: infer mastery beyond a learner's direct evidence.

A prerequisite DAG carries information both ways:

  * **backward (strong)** — if you've mastered a concept that *builds on* X, you
    have very likely mastered X too;
  * **forward (weak)** — if you've mastered X's prerequisites you are *ready*
    for X: a mild signal, not mastery.

We diffuse the direct scores over the graph with a small message-passing network
(PyTorch Geometric), weighting backward edges more than forward ones. Two
guarantees hold by construction:

  1. inferred mastery only ever *adds* to direct evidence — a node you have real
     evidence for is never pulled down;
  2. the inferred-only contribution is capped below the "known" threshold
     (`EngineConfig.inferred_ceiling`) — so the graph can corroborate weak
     evidence and light up "in-between" nodes as partially known, but pure
     inference never fabricates mastery where there is no evidence at all
     (this is what keeps interior gaps from filling themselves in).

v1 uses fixed (untrained) weights: the architecture is here so Phase 5 can train
it on Duolingo/EdNet against KT baselines (thesis RQ3). Torch is imported lazily
so the rest of the engine — and the API — run without it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from .activation import EngineConfig

_LAYER = None  # cached PyG MessagePassing layer (built lazily on first use)


def _get_layer():
    """Build (once) the message-passing layer; imports torch_geometric lazily."""
    global _LAYER
    if _LAYER is None:
        from torch_geometric.nn import MessagePassing

        class _PrereqDiffusion(MessagePassing):
            # One round of normalized neighbour aggregation: a node's new value
            # is a convex combination (edge weights are pre-normalized per
            # target) of its own and its neighbours' values, so values stay in
            # [0, 1]. Parameter-free in v1 — add learnable weights to train in
            # Phase 5.
            def __init__(self) -> None:
                super().__init__(aggr="add", flow="source_to_target")

            def forward(self, x, edge_index, edge_weight):
                return self.propagate(edge_index, x=x, w=edge_weight)

            def message(self, x_j, w):
                return w.view(-1, 1) * x_j

        _LAYER = _PrereqDiffusion()
        _LAYER.eval()
    return _LAYER


def _build_edges(
    g: nx.DiGraph,
    idx: dict[str, int],
    config: "EngineConfig",
    torch,
    *,
    edge_factors: dict[tuple[str, str], tuple[float, float]] | None = None,
):
    """Weighted edge set (self + backward + forward), normalized per target.

    Normalizing each target's incoming weights to sum to 1 makes every
    propagation round a convex combination, keeping values bounded in [0, 1].
    ``edge_factors`` (from ``kgt.tune_edges``) personalizes the graph: per DAG
    edge a ``(back, fwd)`` multiplier on the raw alphas, applied *before* the
    normalization so convexity — and with it both propagation guarantees —
    survives any tuning (self-loops are never scaled, so a target's incoming
    mass stays >= 1 even with every edge removed).
    """
    src: list[int] = []
    dst: list[int] = []
    w: list[float] = []
    for i in idx.values():            # self-loops keep a node's own evidence
        src.append(i); dst.append(i); w.append(1.0)
    for u, v in g.edges:              # u is a prerequisite of v
        iu, iv = idx[u], idx[v]
        m_back, m_fwd = edge_factors.get((u, v), (1.0, 1.0)) if edge_factors else (1.0, 1.0)
        src.append(iv); dst.append(iu); w.append(config.prop_alpha_back * m_back)  # dependent -> prereq (strong)
        src.append(iu); dst.append(iv); w.append(config.prop_alpha_fwd * m_fwd)    # prereq -> dependent (weak)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(w, dtype=torch.float32)
    deg = torch.zeros(len(idx), dtype=torch.float32).index_add_(0, edge_index[1], edge_weight)
    edge_weight = edge_weight / deg[edge_index[1]]
    return edge_index, edge_weight


def propagate(
    g: nx.DiGraph,
    direct: dict[str, float],
    config: "EngineConfig",
    *,
    edge_factors: dict[tuple[str, str], tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Per-node mastery in [0, 1] for every node in ``g``.

    Diffuses the direct scores over the prerequisite graph (strong backward,
    weak forward), then adds the graph-inferred lift to the direct evidence,
    capped at ``config.inferred_ceiling``. See the module docstring for the two
    guarantees this enforces. ``edge_factors`` personalizes the edge weights
    (KGT, thesis RQ5) — see ``_build_edges``.
    """
    import torch

    nodes = list(g.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    x = torch.tensor([[direct.get(n, 0.0)] for n in nodes], dtype=torch.float32)

    edge_index, edge_weight = _build_edges(g, idx, config, torch, edge_factors=edge_factors)
    layer = _get_layer()
    with torch.no_grad():
        h = x
        for _ in range(config.prop_rounds):
            h = layer(h, edge_index, edge_weight)
    inferred = h.squeeze(1).clamp(0.0, 1.0)

    direct_t = x.squeeze(1)
    lift = (inferred - direct_t).clamp(min=0.0).clamp(max=config.inferred_ceiling)
    mastery = (direct_t + lift).clamp(0.0, 1.0)
    return {nodes[i]: mastery[i].item() for i in range(len(nodes))}
