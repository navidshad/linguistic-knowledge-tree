"""Synthetic evidence: deterministic learner event streams.

Two generators, both seeded and wall-clock-free so the engine stays
reproducible:

  * `demo_events()` — the Phase-1 demo learner, reproduced through the real
    event pipeline so the viewer's "demo" overlay keeps working.
  * `generate_events()` — a parametric learner who "truly knows" a given set of
    nodes, with noisy graded reviews spread over a recent window. Useful for
    engine tests now and as a stand-in for real KT data until the Duolingo
    adapter lands (Phase 5).
"""
from __future__ import annotations

import random
from collections.abc import Iterable

from ..events import Event

# The demo learner's known set: deliberate interior gaps — knows the first and
# third conditional but not the second; reported questions but not indirect
# questions. (Carried over from the Phase-1 fixture; now expressed as events.)
DEMO_KNOWN: frozenset[str] = frozenset({
    "present_simple", "present_continuous", "past_simple", "past_continuous",
    "present_perfect_simple", "will_future", "going_to_future",
    "countable_uncountable", "plural_nouns", "articles_a_an", "articles_the",
    "quantifiers_basic", "possessive_s",
    "personal_pronouns", "possessive_pronouns", "demonstratives", "reflexive_pronouns",
    "adjective_basic", "comparatives", "superlatives",
    "adverbs_frequency", "adverbs_time_place", "adverbs_manner",
    "can_ability", "could_past_ability", "must_have_to", "should_advice",
    "gerund_basic", "infinitive_of_purpose",
    "zero_conditional", "first_conditional", "third_conditional",
    "coordination", "basic_linkers",
    "yes_no_questions", "wh_questions", "short_answers", "negation_basic",
    "question_words_advanced",
    "prepositions_place", "prepositions_time", "prepositions_movement",
    "relative_pronouns", "relative_clauses_defining",
    "reported_statements", "reported_questions",
    "phrasal_verbs_basic", "collocation",
})


def demo_events() -> list[Event]:
    """One recent, correct graded review per node the demo learner knows.

    With the default engine config every such node clears the "known" threshold
    on direct evidence alone, so the demo learner's activated set is exactly
    `DEMO_KNOWN` — preserving the Phase-1 overlay (and its three interior gaps).
    """
    return [
        Event(learner_id="demo", node_ids=(node,), correct=True, ts=0.0, source="review")
        for node in sorted(DEMO_KNOWN)
    ]


def generate_events(
    known: Iterable[str],
    *,
    learner_id: str = "synthetic",
    seed: int = 0,
    reviews_per_node: int = 3,
    accuracy: float = 0.9,
    span_days: float = 60.0,
) -> list[Event]:
    """A deterministic event stream for a learner who knows ``known``.

    Each known node gets ``reviews_per_node`` graded reviews placed at random
    times within the last ``span_days`` (ts in [0, span_days], larger = more
    recent), correct with probability ``accuracy``. Seeded purely by ``seed`` —
    no wall-clock — so results are stable across runs.
    """
    rng = random.Random(seed)
    events: list[Event] = []
    for node in sorted(set(known)):
        for _ in range(reviews_per_node):
            ts = rng.uniform(0.0, span_days)
            correct = rng.random() < accuracy
            events.append(
                Event(learner_id=learner_id, node_ids=(node,), correct=correct,
                      ts=ts, source="review")
            )
    return events
