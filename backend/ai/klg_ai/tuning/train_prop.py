"""Global propagation-weight training (RQ3): fit the edge weights from open data.

The core engine diffuses evidence along prerequisite edges with **hand-set,
untrained** weights (``prop_alpha_back=0.5``, ``prop_alpha_fwd=0.15``;
``core.propagation`` is parameter-free). Phase 5 found propagation null on
predictive AUROC — but the weights were never *fit*. This module removes that
hedge: gradient-fit **one global weight set, shared by all learners**, on a
knowledge-tracing dataset, then the eval harness compares trained vs. fixed
(``eval.train_prop_predictor``).

This is **not** the per-learner retrain of ``tuning.retrain`` (that fits a
separate θ per learner — the RQ5 cost question). Here a single θ is fit on every
learner's pooled train interactions. It reuses the same differentiable
propagation forward (self-loops, per-target normalization, ``prop_rounds``
rounds, lift clamped at ``inferred_ceiling``), but batches all learners at once:
the graph is shared, only the per-learner direct-score vector differs, so one
sparse diffusion propagates every learner's column simultaneously.

Two granularities (the RQ3 headline + a robustness ablation):

  * ``"edge"`` — a ``(back, fwd)`` multiplier per prerequisite edge
    (``2·n_edges`` params); the headline trained model.
  * ``"scalar"`` — just two multipliers, broadcast to all back / all fwd edges
    (≡ learning the global ``alpha_back`` / ``alpha_fwd``); a 2-param robustness
    check that the result is not an artifact of one hand-set constant.

Both produce the same ``dict[(u, v) -> (back, fwd)]`` shape that
``core.propagation.propagate(edge_factors=...)`` consumes, so the trained weights
plug straight into the engine. ``θ`` is reparameterized ``m = factor_max·sigmoid(θ)``
(θ=0 ⇒ ×1 ≡ the global graph) with L2 on θ anchoring at the global weights.
Deterministic: zero init, full-batch Adam, fixed epochs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx

from klg_ai.core.activation import EngineConfig
from klg_ai.core.evidence import direct_scores

if TYPE_CHECKING:
    from klg_ai.data.dataset import LearnerData

__all__ = ["GlobalPropTrace", "fit_global_edge_factors"]


@dataclass(frozen=True)
class GlobalPropTrace:
    """Outcome of a global propagation-weight fit.

    ``factors`` is the per-edge ``(back, fwd)`` multiplier dict (only edges that
    deviate from ×1) ready for ``propagate(edge_factors=...)``; ``losses[i]`` is
    the pooled train BCE after epoch ``i``.
    """
    factors: dict[tuple[str, str], tuple[float, float]]
    losses: list[float]
    granularity: str
    n_learners: int
    n_items: int

    def as_weights_json(self, g: nx.DiGraph, config: EngineConfig = EngineConfig()) -> dict:
        """Serializable artifact: per-edge multipliers + the effective alphas.

        ``edges`` maps ``"u->v" -> {back, fwd, alpha_back, alpha_fwd}`` where the
        alphas are the multiplier times the global base (the weight the trained
        engine actually diffuses with).
        """
        edges = {}
        for (u, v) in g.edges:
            mb, mf = self.factors.get((u, v), (1.0, 1.0))
            edges[f"{u}->{v}"] = {
                "back": round(mb, 6), "fwd": round(mf, 6),
                "alpha_back": round(config.prop_alpha_back * mb, 6),
                "alpha_fwd": round(config.prop_alpha_fwd * mf, 6),
            }
        return {
            "granularity": self.granularity,
            "base": {"prop_alpha_back": config.prop_alpha_back,
                     "prop_alpha_fwd": config.prop_alpha_fwd,
                     "prop_rounds": config.prop_rounds,
                     "inferred_ceiling": config.inferred_ceiling},
            "n_learners": self.n_learners, "n_items": self.n_items,
            "n_edges_moved": len(self.factors),
            "edges": edges,
        }


def fit_global_edge_factors(
    g: nx.DiGraph,
    learners: "list[LearnerData]",
    config: EngineConfig = EngineConfig(),
    *,
    granularity: str = "edge",
    epochs: int = 40,
    lr: float = 0.05,
    l2: float = 1e-2,
    seed: int = 0,
) -> GlobalPropTrace:
    """Fit one global ``(back, fwd)`` edge-weight set on all learners' train data.

    Direct scores are held fixed (computed once per learner at their last train
    day); only the shared edge factors move, by full-batch Adam minimizing pooled
    BCE between the engine's mastery readout and train correctness.
    """
    import torch

    if granularity not in ("edge", "scalar"):
        raise ValueError(f"granularity must be 'edge' or 'scalar', got {granularity!r}")

    nodes = list(g.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    n_edges = g.number_of_edges()

    # Per-learner direct scores (fixed) -> rows of X; supervision = pooled train items.
    x_rows: list[list[float]] = []
    item_l: list[int] = []          # learner row per (item,node) ref
    item_i: list[int] = []          # item index per (item,node) ref
    item_node: list[int] = []       # node index per (item,node) ref
    ys: list[float] = []
    n_items = 0
    for li, ld in enumerate(learners):
        events = ld.train_events()
        last_day = max((it.day for it in ld.train), default=ld.eval_ref_day)
        direct = direct_scores(events, now=last_day,
                               half_life_days=config.half_life_days, forgetting=config.forgetting)
        x_rows.append([direct.get(node, 0.0) for node in nodes])
        for it in ld.train:
            ids = [idx[nid] for nid in it.node_ids if nid in idx]
            if not ids:
                continue
            for nid in ids:
                item_l.append(li); item_i.append(n_items); item_node.append(nid)
            ys.append(1.0 if it.correct else 0.0)
            n_items += 1

    if not ys or n_edges == 0 or epochs <= 0:
        return GlobalPropTrace(factors={}, losses=[], granularity=granularity,
                               n_learners=len(learners), n_items=n_items)

    torch.manual_seed(seed)
    L = len(learners)
    X = torch.tensor(x_rows, dtype=torch.float32)              # [L, N]

    # Edge structure mirrors core.propagation._build_edges / tuning.retrain:
    # self-loops first (weight 1, never scaled), then per DAG edge back + fwd.
    edge_src = [j for u, v in g.edges for j in (idx[v], idx[u])]
    edge_dst = [j for u, v in g.edges for j in (idx[u], idx[v])]
    self_src = list(range(n)); self_dst = list(range(n))
    src_t = torch.tensor(self_src + edge_src, dtype=torch.long)
    dst_t = torch.tensor(self_dst + edge_dst, dtype=torch.long)
    self_w = torch.ones(n, dtype=torch.float32)
    base_edge = torch.tensor(
        [a for _ in g.edges for a in (config.prop_alpha_back, config.prop_alpha_fwd)],
        dtype=torch.float32)

    # Supervision index tensors (constant across epochs).
    item_l_t = torch.tensor(item_l, dtype=torch.long)
    item_i_t = torch.tensor(item_i, dtype=torch.long)
    item_node_t = torch.tensor(item_node, dtype=torch.long)
    flat_idx = item_l_t * n + item_node_t                       # into mastery.reshape(-1)
    counts = torch.zeros(n_items).index_add(0, item_i_t, torch.ones(len(item_i)))
    ys_t = torch.tensor(ys, dtype=torch.float32)

    theta = torch.zeros(2 * n_edges if granularity == "edge" else 2, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    fmax = config.kgt_factor_max

    def edge_mult():
        if granularity == "edge":
            return fmax * torch.sigmoid(theta)                  # [2*n_edges]
        m2 = fmax * torch.sigmoid(theta)                        # [2] (back, fwd)
        return m2.repeat(n_edges)                               # broadcast per edge

    def forward():
        w = torch.cat([self_w, base_edge * edge_mult()])        # [n + 2*n_edges]
        deg = torch.zeros(n).index_add(0, dst_t, w)
        w_norm = w / deg[dst_t]                                 # convex per target
        H = X
        for _ in range(config.prop_rounds):
            msg = w_norm.unsqueeze(0) * H[:, src_t]             # [L, E]
            H = torch.zeros(L, n).index_add(1, dst_t, msg)
        lift = (H - X).clamp(min=0.0).clamp(max=config.inferred_ceiling)
        return (X + lift).clamp(0.0, 1.0)                       # [L, N]

    losses: list[float] = []
    for _ in range(epochs):
        opt.zero_grad()
        mastery = forward().reshape(-1)                         # [L*N]
        sums = torch.zeros(n_items).index_add(0, item_i_t, mastery[flat_idx])
        preds = (sums / counts).clamp(1e-4, 1 - 1e-4)
        loss = -(ys_t * preds.log() + (1 - ys_t) * (1 - preds).log()).mean() \
            + l2 * (theta ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

    with torch.no_grad():
        m = (fmax * torch.sigmoid(theta))
        m = m.tolist() if granularity == "edge" else (m.tolist() * n_edges)
    factors: dict[tuple[str, str], tuple[float, float]] = {}
    for k, (u, v) in enumerate(g.edges):
        mb, mf = m[2 * k], m[2 * k + 1]
        if abs(mb - 1.0) > 1e-6 or abs(mf - 1.0) > 1e-6:
            factors[(u, v)] = (mb, mf)
    return GlobalPropTrace(factors=factors, losses=losses, granularity=granularity,
                           n_learners=L, n_items=n_items)
