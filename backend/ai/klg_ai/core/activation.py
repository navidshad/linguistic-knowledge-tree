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

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from klg_ai.tuning.kgt import EdgeAdjustment

# Re-exported so existing imports (`from klg_ai.activation import ...`) keep working.
from klg_ai.data.adapters.synthetic import DEMO_KNOWN, demo_events, generate_events
from klg_ai.core.evidence import direct_scores
from klg_ai.core.events import SOURCE_WEIGHTS, Event
from klg_ai.core.graph import default_graph

__all__ = [
    "EngineConfig", "DEFAULT_CONFIG",
    "get_activation", "compute_mastery", "compute_edge_adjustments", "mastery_timeline",
    "learner_events",
    "mastery_from_events", "activated_from_events", "threshold_activated",
    "TimelineFrame", "event_span",
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
    kgt: bool = False                 # RQ5 ablation: per-learner edge tuning (kgt.py)?
    kgt_mass0: float = 2.0            # evidence gate: > evidence's mass0, demands consistency
    kgt_gain_up: float = 0.5          # γ⁺ — how far agreement reinforces an edge
    kgt_gain_down: float = 1.25       # γ⁻ — how far contradiction weakens it (can reach 0)
    kgt_factor_min: float = 0.0       # multiplier floor (0 = edge removal)
    kgt_factor_max: float = 2.0       # multiplier ceiling
    kgt_min_effect: float = 0.1       # report adjustments deviating at least this from ×1

    def __post_init__(self) -> None:
        if self.inferred_ceiling >= self.known_threshold:
            raise ValueError(
                "inferred_ceiling must be < known_threshold so pure inference "
                "cannot fabricate 'known' nodes"
            )
        if not (0.0 <= self.kgt_factor_min <= 1.0 <= self.kgt_factor_max):
            raise ValueError("kgt factor bounds must satisfy 0 <= min <= 1 <= max")
        if self.kgt_gain_up < 0 or self.kgt_gain_down < 0 or self.kgt_mass0 <= 0:
            raise ValueError("kgt gains must be >= 0 and kgt_mass0 > 0")


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
    which lets a caller evaluate mastery as of any point in time. With
    ``config.kgt`` the propagation runs over the learner's *personalized* edge
    weights (see ``kgt.tune_edges``).
    """
    events = list(events)  # consumed twice when KGT is on
    direct = direct_scores(
        events, now=now, half_life_days=config.half_life_days, forgetting=config.forgetting
    )
    if not config.propagation:
        return {n: direct.get(n, 0.0) for n in g.nodes}
    edge_factors = None
    if config.kgt:
        from klg_ai.tuning.kgt import tune_edges
        edge_factors = tune_edges(g, events, config, now=now).factors
    from klg_ai.core.propagation import propagate  # lazy: torch only loaded when propagation runs
    return propagate(g, direct, config, edge_factors=edge_factors)


def threshold_activated(
    mastery: Mapping[str, float], config: EngineConfig = DEFAULT_CONFIG
) -> set[str]:
    """The activated ("known") set: nodes whose mastery clears the threshold.

    The single place mastery becomes the discrete known-set, so a caller that
    already has the continuous scores doesn't re-run propagation just to derive
    it (e.g. the API computes mastery once, then both reports it and thresholds
    it; the timeline reuses each frame's mastery).
    """
    return {n for n, m in mastery.items() if m >= config.known_threshold}


def activated_from_events(
    g: nx.DiGraph,
    events: Iterable[Event],
    config: EngineConfig = DEFAULT_CONFIG,
    *,
    now: float | None = None,
) -> set[str]:
    """The set of nodes whose mastery clears ``config.known_threshold``."""
    return threshold_activated(mastery_from_events(g, events, config, now=now), config)


@dataclass(frozen=True)
class LearnerProfile:
    """A selectable learner shown in the viewer (metadata only).

    ``editable`` distinguishes the built-in synthetic learners (read-only,
    ``False``) from file-backed user profiles (``True``) — see ``profiles.py``.
    """
    id: str
    label: str
    description: str = ""
    editable: bool = False


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
    LearnerProfile("struggling", "Struggling — contradictory feedback",
                   "Uses some B1 structures correctly while consistently failing their "
                   "prerequisites — watch KGT weaken or cut those inference edges."),
]


def list_learners() -> list[LearnerProfile]:
    """Built-in learners (read-only) + file-backed user profiles (editable).

    Built-ins are listed first and unchanged; user profiles come from the
    persistent store (``profiles.py``).
    """
    from klg_ai.data.profiles import list_profiles
    stored = [
        LearnerProfile(p.id, p.label, p.description, editable=True)
        for p in list_profiles()
    ]
    return list(LEARNER_PROFILES) + stored


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
    if learner_id == "struggling":
        # Knows the A1 core plus two B1/B2 dependents, while consistently failing
        # those dependents' direct prerequisites — the contradiction KGT detects
        # (produces defining relative clauses but fails relative pronouns; handles
        # the third conditional but fails the second).
        known = _cefr_nodes("A1") + ["relative_clauses_defining", "third_conditional"]
        failed = ["relative_pronouns", "second_conditional"]
        return generate_events(known, learner_id="struggling",
                               seed=13, reviews_per_node=4, accuracy=1.0, span_days=14.0,
                               failed=failed)
    from klg_ai.data.profiles import load_events  # lazy: file store only touched on fall-through
    return load_events(learner_id)


def learner_events(learner_id: str) -> list[Event] | None:
    """The built-in learner's event stream, or None if unknown.

    Public accessor over the registry so callers outside the engine (e.g. the
    API's live-retrain endpoint) can feed a learner's events to other engine
    functions without reaching into ``_events_for``.
    """
    return _events_for(learner_id)


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


def compute_edge_adjustments(
    learner_id: str,
    *,
    graph: nx.DiGraph | None = None,
    config: EngineConfig | None = None,
    now: float | None = None,
) -> list["EdgeAdjustment"] | None:
    """A learner's personalized edge adjustments, or None if unknown.

    ``config`` defaults to the engine defaults *with KGT enabled* — the caller
    asking for adjustments wants them computed regardless of the global toggle.
    """
    from klg_ai.tuning.kgt import tune_edges
    events = _events_for(learner_id)
    if events is None:
        return None
    g = default_graph() if graph is None else graph
    cfg = EngineConfig(kgt=True) if config is None else config
    return tune_edges(g, events, cfg, now=now).adjustments


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


@dataclass(frozen=True)
class TimelineFrame:
    """A learner's knowledge state at one reference time, for the viewer scrubber.

    ``mastery`` covers every node; ``activated`` is the thresholded known-set at
    that instant (so colors can shift as nodes cross the threshold over time).
    """
    t: float
    mastery: dict[str, float]
    activated: set[str]


def event_span(events: Iterable[Event]) -> tuple[float, float]:
    """``(first, last)`` timestamp among timed events; ``(0.0, 0.0)`` if none."""
    times = [e.ts for e in events if e.ts is not None]
    return (min(times), max(times)) if times else (0.0, 0.0)


def mastery_timeline(
    learner_id: str,
    *,
    frames: int = 24,
    graph: nx.DiGraph | None = None,
    config: EngineConfig = DEFAULT_CONFIG,
    horizon_half_lives: float = 2.0,
) -> list[TimelineFrame] | None:
    """Mastery sampled across a learner's history, or None if unknown.

    Steps a causal reference time from the first event to the last event plus a
    forgetting horizon (``horizon_half_lives`` half-lives past the final review,
    so post-practice *decay* is visible). Because evidence is evaluated
    point-in-time (see ``evidence.direct_scores``), early frames see only the
    events that had happened by then — so the scrubber shows mastery grow as
    evidence accumulates, then fade. With forgetting off the horizon collapses
    (there is no decay to show) and the grid spans just the evidence window.
    """
    events = _events_for(learner_id)
    if events is None:
        return None
    g = default_graph() if graph is None else graph

    t0, t1 = event_span(events)
    horizon = config.half_life_days * horizon_half_lives if config.forgetting else 0.0
    t_end = t1 + horizon
    n = max(2, frames)
    step = (t_end - t0) / (n - 1) if t_end > t0 else 0.0

    out: list[TimelineFrame] = []
    for i in range(n):
        t = t0 + step * i
        mastery = mastery_from_events(g, events, config, now=t)
        out.append(TimelineFrame(t=t, mastery=mastery, activated=threshold_activated(mastery, config)))
    return out
