"""The trained-propagation arm of RQ3: engine with *globally fit* edge weights.

Wraps ``tuning.train_prop.fit_global_edge_factors`` in the eval-harness predictor
interface. Unlike ``PerLearnerRetrainPredictor`` (a separate θ per learner — the
RQ5 cost question), this fits **one** global weight set on all learners' pooled
train data, then predicts exactly like ``EnginePredictor`` but with those trained
edge factors — so trained vs. fixed is an apples-to-apples comparison (same
causal snapshots, same global Platt calibration; only the edge weights differ).

``curve_`` (pooled train loss per epoch) feeds the dashboard's training-loss plot.
"""
from __future__ import annotations

import numpy as np

from klg_ai.core.activation import EngineConfig
from klg_ai.core.evidence import direct_scores
from klg_ai.core.graph import default_graph
from klg_ai.data.dataset import LearnerData
from klg_ai.eval.predict import _mean_mastery, _sigmoid, fit_platt, global_correct_rate
from klg_ai.tuning.train_prop import fit_global_edge_factors


class TrainedEnginePredictor:
    """Engine predictor whose propagation edge weights are globally trained.

    Mirrors ``EnginePredictor`` (same causal point-in-time mastery + global Platt
    calibration); the only difference is the one-time global fit of the edge
    factors on train, applied to every learner.
    """

    def __init__(self, config: EngineConfig = EngineConfig(), *, granularity: str = "edge",
                 epochs: int = 40, lr: float = 0.05, l2: float = 1e-2, seed: int = 0,
                 name: str | None = None):
        self.config = config
        self.granularity = granularity
        self.epochs = epochs
        self.lr = lr
        self.l2 = l2
        self.seed = seed
        self.name = name or ("engine_trained" if granularity == "edge" else "engine_trained_scalar")
        self.curve_: list[dict] | None = None
        self.factors_: dict[tuple[str, str], tuple[float, float]] | None = None

    def predict(self, learners: list[LearnerData]) -> list[float]:
        from klg_ai.core.propagation import propagate

        g = default_graph()
        base = global_correct_rate(learners)

        # One global fit on all learners' pooled train interactions.
        trace = fit_global_edge_factors(
            g, learners, self.config, granularity=self.granularity,
            epochs=self.epochs, lr=self.lr, l2=self.l2, seed=self.seed)
        self.factors_ = trace.factors or None
        self.curve_ = [{"epoch": i + 1, "loss": round(ls, 4)} for i, ls in enumerate(trace.losses)]

        # Per-learner causal mastery with the trained global weights + calibration pairs.
        mastery_at_eval: dict[str, dict[str, float]] = {}
        cal_x: list[float] = []
        cal_y: list[float] = []
        for ld in learners:
            events = ld.train_events()

            def mastery_at(now: float) -> dict[str, float]:
                direct = direct_scores(events, now=now,
                                       half_life_days=self.config.half_life_days,
                                       forgetting=self.config.forgetting)
                return propagate(g, direct, self.config, edge_factors=self.factors_)

            mastery_at_eval[ld.user] = mastery_at(ld.eval_ref_day)
            last_day = max((it.day for it in ld.train), default=ld.eval_ref_day)
            m_cal = mastery_at(last_day)
            for it in ld.train:
                cal_x.append(_mean_mastery(it.node_ids, m_cal, base))
                cal_y.append(1.0 if it.correct else 0.0)

        a, b = fit_platt(cal_x, cal_y)
        preds: list[float] = []
        for ld in learners:
            m = mastery_at_eval[ld.user]
            for it in ld.evalset:
                xv = _mean_mastery(it.node_ids, m, base)
                preds.append(1.0 - float(_sigmoid(np.array(a * xv + b))))
        return preds
