"""Per-learner activation: turn a learner's events into the nodes their data has
lit up (and the continuous mastery behind it).

Pipeline::

    events ─▶ direct scores      (evidence.py — source-weighted, with forgetting)
           ─▶ GNN propagation    (propagation.py — graph inference beyond evidence)
           ─▶ threshold          ─▶ activated "known" set

`get_activation(learner_id)` keeps its Phase-1 contract — a `set[str]` of known
nodes, or `None` for an unknown learner — so the status computer, the API, and
the Vue viewer are unchanged; they consume whatever the engine computes.
`compute_mastery` exposes the continuous scores behind the threshold (for a
future confidence overlay / timeline; thesis §3.6, roadmap Phase 4-B).

`EngineConfig` carries the knobs and the `forgetting` / `propagation` on-off
switches the validation ablations toggle (thesis RQ3, RQ4).
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import networkx as nx

# Re-exported so existing imports (`from klg_ai.activation import ...`) keep working.
from .adapters.synthetic import DEMO_KNOWN, demo_events, generate_events
from .evidence import direct_scores
from .events import SOURCE_WEIGHTS, Event
from .graph import default_graph

__all__ = [
    "EngineConfig", "DEFAULT_CONFIG",
    "get_activation", "compute_mastery",
    "mastery_from_events", "activated_from_events",
    "LearnerProfile", "list_learners",
    "Event", "SOURCE_WEIGHTS", "DEMO_KNOWN", "demo_events", "generate_events",
]


@dataclass(frozen=True)
class EngineConfig:
    """Knobs for the activation engine.

    ``inferred_ceiling`` must stay below ``known_threshold``: it bounds how far
    pure graph inference can lift a node, guaranteeing that a node with no direct
    evidence is never marked "known" (so interior gaps don't fill themselves in).
    """
    known_threshold: float = 0.5      # mastery at/above which a node counts as known
    half_life_days: float = 30.0      # forgetting half-life for evidence recency
    forgetting: bool = True           # RQ4 ablation: decay old evidence?
    propagation: bool = True          # RQ3 ablation: run GNN graph inference?
    prop_alpha_back: float = 0.5      # weight of "dependent known => prereq known" (strong)
    prop_alpha_fwd: float = 0.15      # weight of "prereqs known => ready for this" (weak)
    prop_rounds: int = 2              # message-passing iterations
    inferred_ceiling: float = 0.45    # cap on inference-only lift (< known_threshold)

    def __post_init__(self) -> None:
        if self.inferred_ceiling >= self.known_threshold:
            raise ValueError(
                "inferred_ceiling must be < known_threshold so pure inference "
                "cannot fabricate 'known' nodes"
            )


DEFAULT_CONFIG = EngineConfig()


def mastery_from_events(
    g: nx.DiGraph,
    events: Iterable[Event],
    config: EngineConfig = DEFAULT_CONFIG,
    *,
    now: float | None = None,
) -> dict[str, float]:
    """Per-node mastery in [0, 1] for every node in ``g`` from an event stream.

    ``now`` overrides the decay reference time (defaults to the latest event),
    which lets a caller evaluate mastery as of any point in time.
    """
    direct = direct_scores(
        events, now=now, half_life_days=config.half_life_days, forgetting=config.forgetting
    )
    if not config.propagation:
        return {n: direct.get(n, 0.0) for n in g.nodes}
    from .propagation import propagate  # lazy: torch only loaded when propagation runs
    return propagate(g, direct, config)


def activated_from_events(
    g: nx.DiGraph,
    events: Iterable[Event],
    config: EngineConfig = DEFAULT_CONFIG,
    *,
    now: float | None = None,
) -> set[str]:
    """The set of nodes whose mastery clears ``config.known_threshold``."""
    mastery = mastery_from_events(g, events, config, now=now)
    return {n for n, m in mastery.items() if m >= config.known_threshold}


@dataclass(frozen=True)
class LearnerProfile:
    """A selectable learner shown in the viewer (metadata only)."""
    id: str
    label: str
    description: str = ""


# Built-in learners for the selector. "demo" is the curated fixture; the others
# are synthetic personas generated from CEFR bands of the map, so each overlay
# cleanly shows that learner's known core and the frontier just beyond it.
LEARNER_PROFILES: list[LearnerProfile] = [
    LearnerProfile("demo", "Demo — mixed, with gaps",
                   "Curated learner with deliberate interior gaps (knows 1st & 3rd conditional, not 2nd)."),
    LearnerProfile("beginner", "Beginner — A1",
                   "Knows most A1 grammar; everything above is frontier or further."),
    LearnerProfile("intermediate", "Intermediate — A1–B1",
                   "Solid on A1–A2 and about half of B1."),
]


def list_learners() -> list[LearnerProfile]:
    """Metadata for the built-in learners (powers the API's learner selector)."""
    return list(LEARNER_PROFILES)


def _cefr_nodes(*levels: str) -> list[str]:
    """Map node ids in the given CEFR level(s), sorted for determinism."""
    wanted = set(levels)
    return sorted(n for n, d in default_graph().nodes(data=True) if d.get("cefr") in wanted)


# Learner registry: id -> event stream. Synthetic personas now; real adapters
# (Duolingo/EdNet for validation, then Subturtle) plug in here later.
def _events_for(learner_id: str) -> list[Event] | None:
    if learner_id == "demo":
        return demo_events()
    if learner_id == "beginner":
        return generate_events(_cefr_nodes("A1"), learner_id="beginner",
                               seed=11, reviews_per_node=3, accuracy=1.0, span_days=14.0)
    if learner_id == "intermediate":
        b1 = _cefr_nodes("B1")
        known = _cefr_nodes("A1", "A2") + b1[: len(b1) // 2]
        return generate_events(known, learner_id="intermediate",
                               seed=12, reviews_per_node=3, accuracy=1.0, span_days=14.0)
    return None


def compute_mastery(
    learner_id: str,
    *,
    graph: nx.DiGraph | None = None,
    config: EngineConfig = DEFAULT_CONFIG,
    now: float | None = None,
) -> dict[str, float] | None:
    """Continuous per-node mastery for a learner, or None if unknown."""
    events = _events_for(learner_id)
    if events is None:
        return None
    g = default_graph() if graph is None else graph
    return mastery_from_events(g, events, config, now=now)


def get_activation(
    learner_id: str,
    *,
    graph: nx.DiGraph | None = None,
    config: EngineConfig = DEFAULT_CONFIG,
    now: float | None = None,
) -> set[str] | None:
    """Return the activated ("known") node set for a learner, or None if unknown.

    Phase-1 contract preserved: a plain `set[str]` the status computer consumes.
    Now derived from the learner's events rather than a hardcoded fixture.
    """
    events = _events_for(learner_id)
    if events is None:
        return None
    g = default_graph() if graph is None else graph
    return activated_from_events(g, events, config, now=now)
