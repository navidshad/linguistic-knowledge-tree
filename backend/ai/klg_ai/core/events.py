"""Learner interaction events — the evidence that activates the map.

An `Event` is one graded interaction tagged to one or more concept nodes. The
activation engine folds a learner's event stream into per-node mastery scores,
weighted by:

  * `source`   — graded recall > productive use > passive exposure,
  * `correct`  — right answers raise mastery, wrong ones lower it,
  * recency    — older evidence counts for less (see `forgetting.py`).

Timestamps (`ts`) are floats in **days on an arbitrary fixed origin** (not
wall-clock time), larger = more recent, so synthetic data and tests stay
deterministic. The engine's reference "now" defaults to the latest event time,
so a learner's most recent evidence sees ~no decay.
"""
from __future__ import annotations

from dataclasses import dataclass

# Evidence sources, strongest first. Graded recall (a Leitner review) is the
# strongest signal that a concept is mastered; productive use in dialog is
# weaker; mere exposure (seeing/saving a phrase) is weakest. Mirrors the
# evidence hierarchy in CLAUDE.md: review > dialog > exposure.
SOURCE_WEIGHTS: dict[str, float] = {
    "review": 1.0,    # graded SRS recall (Leitner)
    "dialog": 0.6,    # productive use in conversation
    "exposure": 0.3,  # passive exposure / phrase save
}
DEFAULT_SOURCE = "review"


@dataclass(frozen=True)
class Event:
    """One graded interaction providing evidence about one or more concepts.

    learner_id : whose interaction this is.
    node_ids   : the concept node ids this interaction is tagged to.
    correct    : whether the learner's response was correct.
    ts         : time in days on a fixed origin (None == "now", no decay).
    source     : review | dialog | exposure  (see ``SOURCE_WEIGHTS``).
    """
    learner_id: str
    node_ids: tuple[str, ...]
    correct: bool
    ts: float | None = None
    source: str = DEFAULT_SOURCE

    @property
    def weight(self) -> float:
        """Source weight; unknown sources fall back to the default."""
        return SOURCE_WEIGHTS.get(self.source, SOURCE_WEIGHTS[DEFAULT_SOURCE])
