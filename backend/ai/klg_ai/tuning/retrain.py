"""Differentiable per-learner edge-weight fitting (the reusable core of RQ5).

``fit_edge_factors`` gradient-fits a ``(back, fwd)`` multiplier per prerequisite
edge by minimizing BCE between the engine's mastery readout and a learner's own
train correctness — the same quantity ``klg_ai.tuning.kgt`` computes in one
closed-form pass. Multipliers are reparameterized as
``m(θ) = factor_max · sigmoid(θ)`` (θ=0 ⇒ m=1 ≡ the global graph), with L2 on θ
anchoring the fit at the global weights ("fine-tune from the global model"),
which also regularizes the many-parameters / few-items regime.

The propagation forward pass is re-implemented here in plain differentiable
torch (the cached PyG layer in ``core.propagation`` runs under ``no_grad`` and
can't backprop into edge weights). It mirrors ``_build_edges`` + the lift-clamp
exactly: self-loops, per-target normalization, ``prop_rounds`` rounds, lift
capped at ``inferred_ceiling``.

This module holds only the reusable fit; the eval-harness wrapper
(``PerLearnerRetrainPredictor``) lives in ``klg_ai.eval.retrain_predictor``, and
the live ``POST /api/learner/{id}/retrain`` endpoint calls ``fit_edge_factors``
directly (with ``record_epochs=True``) to animate the fit converging.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from klg_ai.core.activation import EngineConfig
from klg_ai.core.events import Event
from klg_ai.core.evidence import direct_scores, reference_now


@dataclass(frozen=True)
class RetrainTrace:
    """Outcome of one per-learner gradient fit.

    ``losses[i]`` is the train BCE after epoch ``i``; ``epoch_factors`` (only
    with ``record_epochs``) snapshots the edge factors after each epoch so the
    viewer can replay the fit converging.
    """
    factors: dict[tuple[str, str], tuple[float, float]]
    losses: list[float]
    epoch_factors: list[dict[tuple[str, str], tuple[float, float]]] | None = None
    n_items: int = 0


def _factor_dict(g: nx.DiGraph, theta, factor_max: float, torch) -> dict:
    """θ -> per-edge (m_back, m_fwd), skipping edges still at the global ×1."""
    with torch.no_grad():
        m = (factor_max * torch.sigmoid(theta)).tolist()
    out: dict[tuple[str, str], tuple[float, float]] = {}
    for k, (u, v) in enumerate(g.edges):
        mb, mf = m[2 * k], m[2 * k + 1]
        if abs(mb - 1.0) > 1e-6 or abs(mf - 1.0) > 1e-6:
            out[(u, v)] = (mb, mf)
    return out


def fit_edge_factors(
    g: nx.DiGraph,
    events: list[Event],
    config: EngineConfig = EngineConfig(),
    *,
    items: list[tuple[tuple[str, ...], bool]] | None = None,
    now: float | None = None,
    epochs: int = 30,
    lr: float = 0.05,
    l2: float = 1e-2,
    seed: int = 0,
    record_epochs: bool = False,
) -> RetrainTrace:
    """Gradient-fit per-edge multipliers for one learner (the RQ5 retrain arm).

    ``items`` are the supervision pairs ``(node_ids, correct)`` — the learner's
    train interactions by default derived from ``events`` themselves. Direct
    scores are held fixed (computed once at ``now``); only the edge factors
    move. Deterministic: zero init, full-batch Adam, fixed ``epochs``.
    """
    import torch

    if now is None:
        now = reference_now(events)
    if items is None:
        items = [(e.node_ids, e.correct) for e in events]

    nodes = list(g.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    direct = direct_scores(
        events, now=now, half_life_days=config.half_life_days, forgetting=config.forgetting
    )

    torch.manual_seed(seed)
    x = torch.tensor([direct.get(node, 0.0) for node in nodes], dtype=torch.float32)

    # Static edge structure, mirroring propagation._build_edges: self-loops
    # first, then per DAG edge the backward and forward message. theta[2k] is
    # edge k's backward multiplier, theta[2k+1] its forward one.
    src = list(range(n)) + [j for u, v in g.edges for j in (idx[v], idx[u])]
    dst = list(range(n)) + [j for u, v in g.edges for j in (idx[u], idx[v])]
    base_w = [1.0] * n + [a for _ in g.edges for a in (config.prop_alpha_back, config.prop_alpha_fwd)]
    src_t = torch.tensor(src, dtype=torch.long)
    dst_t = torch.tensor(dst, dtype=torch.long)
    base_t = torch.tensor(base_w, dtype=torch.float32)
    n_edges = g.number_of_edges()

    # Supervision: mean mastery over each item's mapped nodes vs. correctness.
    sup = [( [idx[nid] for nid in node_ids if nid in idx], 1.0 if correct else 0.0)
           for node_ids, correct in items]
    sup = [(ids, y) for ids, y in sup if ids]
    if not sup or n_edges == 0 or epochs <= 0:
        return RetrainTrace(factors={}, losses=[], epoch_factors=[] if record_epochs else None,
                            n_items=len(sup))

    theta = torch.zeros(2 * n_edges, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)

    def forward():
        m = config.kgt_factor_max * torch.sigmoid(theta)
        w = base_t.clone()
        w[n:] = base_t[n:] * m
        deg = torch.zeros(n).index_add(0, dst_t, w)
        w_norm = w / deg[dst_t]
        h = x
        for _ in range(config.prop_rounds):
            h = torch.zeros(n).index_add(0, dst_t, w_norm * h[src_t])
        lift = (h - x).clamp(min=0.0).clamp(max=config.inferred_ceiling)
        return (x + lift).clamp(0.0, 1.0)

    losses: list[float] = []
    snapshots: list[dict] | None = [] if record_epochs else None
    for _ in range(epochs):
        opt.zero_grad()
        mastery = forward()
        preds = torch.stack([mastery[ids].mean() for ids, _ in sup]).clamp(1e-4, 1 - 1e-4)
        ys = torch.tensor([y for _, y in sup], dtype=torch.float32)
        loss = -(ys * preds.log() + (1 - ys) * (1 - preds).log()).mean() + l2 * (theta ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        if snapshots is not None:
            snapshots.append(_factor_dict(g, theta, config.kgt_factor_max, torch))

    return RetrainTrace(
        factors=_factor_dict(g, theta, config.kgt_factor_max, torch),
        losses=losses,
        epoch_factors=snapshots,
        n_items=len(sup),
    )
