"""Events -> per-node *direct* mastery scores.

"Direct" mastery is what a learner's own logged interactions say about each
node, before any graph propagation (`propagation.py`) infers beyond the
evidence.

For each node we fold its tagged events into a score in [0, 1]:

    mass  = Σ  weight · recency                       # how much / how fresh
    acc   = Σ  weight · recency · 1[correct]  / mass   # weighted accuracy
    score = acc · (1 − exp(−mass / mass0))             # accuracy, mass-gated

The mass gate means a single passive *exposure* barely moves the score while
repeated graded *reviews* push it high, and a node with no events scores 0. Old
evidence decays via `recency` (see `forgetting.py`), so mastery fades without
practice — which is what makes the map temporally dynamic (thesis RQ4).
"""
from __future__ import annotations

import math
from collections.abc import Iterable

from .events import Event
from .forgetting import recency_weight


def reference_now(events: Iterable[Event]) -> float:
    """Default decay reference: the latest event time (0.0 if none have one)."""
    times = [e.ts for e in events if e.ts is not None]
    return max(times) if times else 0.0


def direct_scores(
    events: Iterable[Event],
    *,
    now: float | None = None,
    half_life_days: float = 30.0,
    forgetting: bool = True,
    mass0: float = 0.7,
) -> dict[str, float]:
    """Aggregate events into a per-node direct mastery score in [0, 1].

    ``now`` is the reference time for decay; defaults to the latest event time
    (so the freshest evidence sees no decay). Evaluation is **point-in-time /
    causal**: evidence logged *after* ``now`` (an event whose ``ts`` is greater
    than ``now``) is ignored, so evaluating an earlier ``now`` reconstructs the
    learner's knowledge as it was then — this is what lets the timeline scrubber
    show mastery *grow* as evidence accumulates (events with ``ts is None`` are
    treated as "current" and always count). Nodes with no events are simply
    absent from the result (i.e. score 0). ``mass0`` sets how much evidence is
    needed to be confident: smaller -> a single review counts for more.
    """
    events = list(events)
    if now is None:
        now = reference_now(events)

    pos: dict[str, float] = {}   # Σ weighted-recency over *correct* events
    mass: dict[str, float] = {}  # Σ weighted-recency over *all* events
    for e in events:
        if e.ts is not None and e.ts > now:
            continue  # causal: this evidence hasn't happened yet at `now`
        age = 0.0 if e.ts is None else now - e.ts
        w = e.weight * recency_weight(age, half_life_days, enabled=forgetting)
        for node in e.node_ids:
            mass[node] = mass.get(node, 0.0) + w
            if e.correct:
                pos[node] = pos.get(node, 0.0) + w

    scores: dict[str, float] = {}
    for node, m in mass.items():
        if m <= 0:
            continue
        acc = pos.get(node, 0.0) / m
        scores[node] = acc * (1.0 - math.exp(-m / mass0))
    return scores
