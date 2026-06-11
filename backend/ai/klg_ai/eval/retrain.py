"""The "retrain per user" arm of RQ5: gradient-fit per-learner edge weights.

RQ5 asks whether personalization needs per-user *retraining* or whether the
lightweight closed-form KGT rule (``klg_ai.kgt``) suffices. This module is the
expensive comparator: per learner, gradient-fit the same quantity KGT computes
in one pass — a ``(back, fwd)`` multiplier per prerequisite edge, bounded in
``(0, kgt_factor_max)`` — by minimizing BCE between the engine's mastery
readout and the learner's own train correctness. Same hypothesis space, same
propagation, same calibration as every other engine arm; the two RQ5 arms
differ *only* in fitting strategy, so the comparison isolates exactly what the
research question asks: predictive fit vs. compute cost.

Multipliers are reparameterized as ``m(θ) = factor_max · sigmoid(θ)`` (with
``factor_max = 2`` the default, ``θ = 0`` ⇒ ``m = 1`` ≡ the global graph), and
L2 on ``θ`` anchors the fit at the global weights — "fine-tune from the global
model", which also regularizes the many-parameters / few-items regime (~220
params per learner vs. typically a few hundred train tokens).

The propagation forward pass is re-implemented here in plain differentiable
torch (the cached PyG layer in ``propagation.py`` runs under ``no_grad`` and
can't backprop into edge weights). It mirrors ``_build_edges`` + the lift-clamp
exactly: self-loops, per-target normalization, ``prop_rounds`` rounds, lift
capped at ``inferred_ceiling``.

``fit_edge_factors`` is reused by the live ``POST /api/learner/{id}/retrain``
endpoint (with ``record_epochs=True``) to animate the fit converging in the
viewer, next to KGT's instant result.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from ..activation import EngineConfig
from ..events import Event
from ..evidence import direct_scores, reference_now
from ..graph import default_graph
from .dataset import LearnerData
from .predict import _mean_mastery, _sigmoid, fit_platt, global_correct_rate

import numpy as np


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


class PerLearnerRetrainPredictor:
    """Engine predictor whose edge weights are gradient-retrained per learner.

    Mirrors ``EnginePredictor`` (same causal snapshots, same global Platt
    calibration); the only difference is the per-learner Adam fit in between.
    ``curve_`` (mean train loss per epoch across learners) is populated by
    ``predict`` for the dashboard's retrain-progress plot.
    """
    name = "engine_retrain"

    def __init__(self, config: EngineConfig = EngineConfig(), *, epochs: int = 30,
                 lr: float = 0.05, l2: float = 1e-2, min_train_mapped: int = 5, seed: int = 0):
        self.config = config
        self.epochs = epochs
        self.lr = lr
        self.l2 = l2
        self.min_train_mapped = min_train_mapped
        self.seed = seed
        self.curve_: list[dict] | None = None

    def predict(self, learners: list[LearnerData]) -> list[float]:
        from ..propagation import propagate

        g = default_graph()
        base = global_correct_rate(learners)

        epoch_losses: list[list[float]] = [[] for _ in range(self.epochs)]
        mastery_at_eval: dict[str, dict[str, float]] = {}
        cal_x: list[float] = []
        cal_y: list[float] = []
        for ld in learners:
            events = ld.train_events()
            items = [(it.node_ids, it.correct) for it in ld.train if it.node_ids]
            last_day = max((it.day for it in ld.train), default=ld.eval_ref_day)

            factors: dict[tuple[str, str], tuple[float, float]] = {}
            if len(items) >= self.min_train_mapped:
                trace = fit_edge_factors(
                    g, events, self.config, items=items, now=last_day,
                    epochs=self.epochs, lr=self.lr, l2=self.l2, seed=self.seed,
                )
                factors = trace.factors
                for i, loss in enumerate(trace.losses):
                    epoch_losses[i].append(loss)

            def mastery_at(now: float) -> dict[str, float]:
                direct = direct_scores(events, now=now,
                                       half_life_days=self.config.half_life_days,
                                       forgetting=self.config.forgetting)
                return propagate(g, direct, self.config, edge_factors=factors or None)

            mastery_at_eval[ld.user] = mastery_at(ld.eval_ref_day)
            m_cal = mastery_at(last_day)
            for it in ld.train:
                cal_x.append(_mean_mastery(it.node_ids, m_cal, base))
                cal_y.append(1.0 if it.correct else 0.0)

        self.curve_ = [
            {"epoch": i + 1, "loss": round(sum(ls) / len(ls), 4)}
            for i, ls in enumerate(epoch_losses) if ls
        ]

        a, b = fit_platt(cal_x, cal_y)
        preds: list[float] = []
        for ld in learners:
            m = mastery_at_eval[ld.user]
            for it in ld.evalset:
                xv = _mean_mastery(it.node_ids, m, base)
                preds.append(1.0 - float(_sigmoid(np.array(a * xv + b))))
        return preds
