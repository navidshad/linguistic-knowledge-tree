"""The "retrain per user" arm of RQ5: a harness predictor over the per-learner fit.

Wraps ``klg_ai.tuning.retrain.fit_edge_factors`` in the eval-harness predictor
interface. Per learner, gradient-fit the same quantity KGT computes in one pass
(a ``(back, fwd)`` multiplier per edge), then predict like ``EnginePredictor``
with those personalized weights. Same causal snapshots + global Platt
calibration as every other engine arm; the two RQ5 arms differ *only* in fitting
strategy, so the comparison isolates exactly what the research question asks:
predictive fit vs. compute cost.

``curve_`` (mean train loss per epoch across learners) is populated by
``predict`` for the dashboard's retrain-progress plot.
"""
from __future__ import annotations

import numpy as np

from klg_ai.core.activation import EngineConfig
from klg_ai.core.evidence import direct_scores
from klg_ai.core.graph import default_graph
from klg_ai.data.dataset import LearnerData
from klg_ai.eval.predict import _mean_mastery, _sigmoid, fit_platt, global_correct_rate
from klg_ai.tuning.retrain import fit_edge_factors


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
        from klg_ai.core.propagation import propagate

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
