"""Forgetting: how evidence loses weight as time passes.

Spaced-repetition intuition: a correct review from yesterday is much stronger
evidence of *current* mastery than one from months ago. Each event is weighted
by an exponential decay on its age, parameterised by a half-life (one half-life
of age halves the weight).

Deliberately a tiny, pure module so the "forgetting on/off" ablation
(thesis RQ4) is a one-line switch in the engine config.
"""
from __future__ import annotations


def recency_weight(age_days: float, half_life_days: float, *, enabled: bool = True) -> float:
    """Decay factor in (0, 1] for evidence that is ``age_days`` old.

    0 days -> 1.0; one half-life -> 0.5; two half-lives -> 0.25. With
    ``enabled=False`` (the ablation), a non-positive half-life, or non-positive
    age, returns 1.0 (no forgetting).
    """
    if not enabled or half_life_days <= 0 or age_days <= 0:
        return 1.0
    return 0.5 ** (age_days / half_life_days)
