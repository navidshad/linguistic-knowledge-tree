"""Prediction metrics, matching Duolingo SLAM's official ``eval.py``.

All four shared-task metrics are computed exactly as the competition did (the
AUROC port is the same tie-aware Mann–Whitney statistic), so our numbers are
directly comparable to the SLAM leaderboard. ``_OURS`` reproduces ``eval.py``'s
self-test (acc=.700, avglogloss=.613, auroc=.740, F1=.667) as a guard.

Convention: ``actual`` is the SLAM label (**1 = the learner made a mistake**),
``predicted`` is P(mistake). AUROC is rank-based, so it is invariant to any
monotonic calibration of the score — which is why the engine can rank by
(1 − mastery) and still produce a leaderboard-comparable AUROC.
"""
from __future__ import annotations

import math
from collections.abc import Sequence


def compute_acc(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Accuracy with a 0.5 cutoff (rounds both sides, as in SLAM eval.py)."""
    n = len(actual)
    return sum(1 for i in range(n) if round(actual[i]) == round(predicted[i])) / n


def compute_avg_log_loss(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Mean negative log-likelihood of the true label."""
    n = len(actual)
    loss = 0.0
    for i in range(n):
        p = predicted[i] if actual[i] > 0.5 else 1.0 - predicted[i]
        loss -= math.log(max(p, 1e-12))
    return loss / n


def compute_auroc(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Area under the ROC curve (tie-aware), ported from SLAM eval.py.

    Returns 0.5 when only one class is present (AUROC is undefined there).
    """
    num = len(actual)
    temp = sorted([[predicted[i], actual[i]] for i in range(num)], reverse=True)
    sorted_predicted = [row[0] for row in temp]
    sorted_actual = [row[1] for row in temp]

    sorted_posterior = sorted(zip(sorted_predicted, range(len(sorted_predicted))))
    r = [0.0 for _ in sorted_predicted]
    cur_val = sorted_posterior[0][0]
    last_rank = 0
    for i in range(len(sorted_posterior)):
        if cur_val != sorted_posterior[i][0]:
            cur_val = sorted_posterior[i][0]
            for j in range(last_rank, i):
                r[sorted_posterior[j][1]] = (last_rank + 1 + i) / 2.0
            last_rank = i
        if i == len(sorted_posterior) - 1:
            for j in range(last_rank, i + 1):
                r[sorted_posterior[j][1]] = (last_rank + i + 2) / 2.0

    num_positive = len([0 for x in sorted_actual if x == 1])
    num_negative = num - num_positive
    if num_positive == 0 or num_negative == 0:
        return 0.5
    sum_positive = sum(r[i] for i in range(len(r)) if sorted_actual[i] == 1)
    return (sum_positive - num_positive * (num_positive + 1) / 2.0) / (num_negative * num_positive)


def compute_f1(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """F1 at a 0.5 cutoff (positive class = mistake)."""
    tp = fp = fn = 0
    for i in range(len(actual)):
        a, p = actual[i] >= 0.5, predicted[i] >= 0.5
        if a and p:
            tp += 1
        elif (not a) and p:
            fp += 1
        elif a and (not p):
            fn += 1
    try:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        return 0.0


def evaluate(actual: Sequence[float], predicted: Sequence[float]) -> dict[str, float]:
    """All SLAM metrics plus instance count and positive (mistake) base rate."""
    n = len(actual)
    return {
        "n": n,
        "base_rate": (sum(actual) / n) if n else 0.0,
        "auroc": compute_auroc(actual, predicted),
        "F1": compute_f1(actual, predicted),
        "accuracy": compute_acc(actual, predicted),
        "avglogloss": compute_avg_log_loss(actual, predicted),
    }


def roc_curve(
    actual: Sequence[float], predicted: Sequence[float], *, max_points: int = 80
) -> list[dict[str, float]]:
    """ROC points ``[{fpr, tpr}, ...]`` for plotting, down-sampled to ~max_points.

    Sweeps thresholds from high to low over the sorted scores; positive class is
    the mistake label. Always includes the (0,0) and (1,1) endpoints.
    """
    pairs = sorted(zip(predicted, actual), reverse=True)
    pos = sum(1 for _, a in pairs if a == 1)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return [{"fpr": 0.0, "tpr": 0.0}, {"fpr": 1.0, "tpr": 1.0}]

    pts = [{"fpr": 0.0, "tpr": 0.0}]
    tp = fp = 0
    prev_score = None
    for score, a in pairs:
        if prev_score is not None and score != prev_score:
            pts.append({"fpr": fp / neg, "tpr": tp / pos})
        if a == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score
    pts.append({"fpr": 1.0, "tpr": 1.0})

    if len(pts) <= max_points:
        return pts
    step = len(pts) / max_points
    keep = [pts[int(i * step)] for i in range(max_points)]
    keep[-1] = pts[-1]
    return keep


# Self-test against the SLAM eval.py oracle (imported lightly so a typo here fails fast).
_ORACLE_ACTUAL = [1, 0, 0, 1, 1, 0, 0, 1, 0, 1]
_ORACLE_PRED = [0.8, 0.2, 0.6, 0.3, 0.1, 0.2, 0.3, 0.9, 0.2, 0.7]


def _self_check() -> None:
    m = evaluate(_ORACLE_ACTUAL, _ORACLE_PRED)
    assert round(m["accuracy"], 3) == 0.700, m
    assert round(m["avglogloss"], 3) == 0.613, m
    assert round(m["auroc"], 3) == 0.740, m
    assert round(m["F1"], 3) == 0.667, m
