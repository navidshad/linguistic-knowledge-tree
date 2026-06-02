"""Per-learner activation: which nodes the learner's data has lit up.

v0 is a fixture (a hardcoded demo learner) so the rest of the stack is testable
end-to-end. Phase 2 replaces `get_activation` with a real event-driven model
(events -> activation, plus GNN propagation and forgetting).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Event:
    """An interaction that provides evidence about one or more concepts.

    Phase 2 will fold these into activation, weighted by `source`
    (review > productive use > exposure). Kept here now to fix the schema.
    """
    learner_id: str
    node_ids: tuple[str, ...]
    correct: bool
    ts: float | None = None
    source: str = "review"  # review | dialog | exposure


# Demo learner with deliberate interior gaps: knows first + third conditional
# (but not second), and reported questions (but not indirect questions).
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


def get_activation(learner_id: str) -> set[str] | None:
    """Return the activated node set for a learner, or None if unknown.

    v0 only knows the 'demo' learner.
    """
    if learner_id == "demo":
        return set(DEMO_KNOWN)
    return None
