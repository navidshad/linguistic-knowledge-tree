"""Predictors scored by the harness — the engine plus difficulty/ability baselines.

Every predictor turns per-learner data into a flat list of **P(mistake)** values
aligned with ``dataset.all_eval_instances`` order, so ``metrics.evaluate`` can
score them all the same way.

The engine predictor is the thesis model: per learner it folds the train events
into point-in-time mastery (forgetting + GNN propagation, both toggleable for the
RQ3/RQ4 ablations), reads each eval token's mapped-node mastery *causally* at the
learner's first eval day, and turns mastery into a probability with a global
Platt (1-D logistic) calibration fit on train. AUROC is rank-based, so the
calibration leaves it unchanged — it only makes the log-loss/accuracy numbers
meaningful. Tokens that map to no node fall back to the global base rate.
"""
from __future__ import annotations

import numpy as np

from klg_ai.core.activation import EngineConfig, mastery_from_events
from klg_ai.core.graph import default_graph
from klg_ai.data.dataset import LearnerData


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_platt(xs: list[float], ys: list[float], *, iters: int = 200, lr: float = 0.5,
              max_pairs: int = 200_000, seed: int = 0) -> tuple[float, float]:
    """Fit ``P(correct) = sigmoid(a·x + b)`` by gradient descent (numpy).

    ``x`` is a correctness proxy in [0, 1] (e.g. mean mastery), ``y`` is 1 for a
    correct answer. Down-samples to ``max_pairs`` for speed. Returns ``(a, b)``.
    """
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if len(x) > max_pairs:
        idx = np.random.default_rng(seed).choice(len(x), max_pairs, replace=False)
        x, y = x[idx], y[idx]
    if len(x) == 0:
        return 1.0, 0.0
    a, b = 1.0, 0.0
    n = len(x)
    for _ in range(iters):
        p = _sigmoid(a * x + b)
        err = p - y
        a -= lr * float(np.dot(err, x)) / n
        b -= lr * float(err.mean())
    return a, b


def _mean_mastery(nodes: tuple[str, ...], mastery: dict[str, float], base: float) -> float:
    if not nodes:
        return base
    return sum(mastery.get(n, 0.0) for n in nodes) / len(nodes)


def global_correct_rate(learners: list[LearnerData]) -> float:
    """Overall fraction of correct answers across all learners' train history."""
    correct = total = 0
    for ld in learners:
        for it in ld.train:
            total += 1
            correct += it.correct
    return correct / total if total else 0.85


class EnginePredictor:
    """The activation engine used as a knowledge-tracing predictor."""

    def __init__(self, config: EngineConfig, name: str = "engine"):
        self.config = config
        self.name = name

    def predict(self, learners: list[LearnerData]) -> list[float]:
        g = default_graph()
        base = global_correct_rate(learners)

        # Per-learner causal mastery for prediction, plus calibration pairs.
        mastery_at_eval: dict[str, dict[str, float]] = {}
        cal_x: list[float] = []
        cal_y: list[float] = []
        for ld in learners:
            events = ld.train_events()
            mastery_at_eval[ld.user] = mastery_from_events(
                g, events, self.config, now=ld.eval_ref_day)
            last_day = max((it.day for it in ld.train), default=ld.eval_ref_day)
            m_cal = mastery_from_events(g, events, self.config, now=last_day)
            for it in ld.train:
                cal_x.append(_mean_mastery(it.node_ids, m_cal, base))
                cal_y.append(1.0 if it.correct else 0.0)

        a, b = fit_platt(cal_x, cal_y)

        preds: list[float] = []
        for ld in learners:
            m = mastery_at_eval[ld.user]
            for it in ld.evalset:
                x = _mean_mastery(it.node_ids, m, base)
                p_correct = float(_sigmoid(np.array(a * x + b)))
                preds.append(1.0 - p_correct)
        return preds


class GlobalMeanPredictor:
    """Constant baseline: everyone's mistake probability = the global rate."""
    name = "global_mean"

    def predict(self, learners: list[LearnerData]) -> list[float]:
        p_mistake = 1.0 - global_correct_rate(learners)
        return [p_mistake for ld in learners for _ in ld.evalset]


class PerSkillMeanPredictor:
    """Item-difficulty baseline: each concept's train correct-rate (smoothed).

    No sequence, no graph — just how hard each node is on average. A strong,
    honest yardstick for whether the graph/sequence models add anything.
    """
    name = "per_skill_mean"

    def __init__(self, alpha: float = 5.0):
        self.alpha = alpha

    def predict(self, learners: list[LearnerData]) -> list[float]:
        base = global_correct_rate(learners)
        correct: dict[str, float] = {}
        total: dict[str, float] = {}
        for ld in learners:
            for it in ld.train:
                for n in it.node_ids:
                    total[n] = total.get(n, 0.0) + 1
                    correct[n] = correct.get(n, 0.0) + it.correct

        def rate(n: str) -> float:
            return (correct.get(n, 0.0) + self.alpha * base) / (total.get(n, 0.0) + self.alpha)

        preds: list[float] = []
        for ld in learners:
            for it in ld.evalset:
                if it.node_ids:
                    p_correct = sum(rate(n) for n in it.node_ids) / len(it.node_ids)
                else:
                    p_correct = base
                preds.append(1.0 - p_correct)
        return preds


class PerUserMeanPredictor:
    """Learner-ability baseline: each learner's own train correct-rate (smoothed)."""
    name = "per_user_mean"

    def __init__(self, alpha: float = 5.0):
        self.alpha = alpha

    def predict(self, learners: list[LearnerData]) -> list[float]:
        base = global_correct_rate(learners)
        preds: list[float] = []
        for ld in learners:
            c = sum(it.correct for it in ld.train)
            n = len(ld.train)
            p_correct = (c + self.alpha * base) / (n + self.alpha)
            preds.extend(1.0 - p_correct for _ in ld.evalset)
        return preds
