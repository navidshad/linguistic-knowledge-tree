"""Knowledge Graph Tuning (KGT): per-learner edge tuning without retraining.

The static map is the same for every learner; propagation diffuses evidence
along its prerequisite edges with *global* weights. But a prerequisite relation
is a claim about learners in general — an individual can contradict it (know a
dependent while consistently failing its prerequisite). KGT (thesis §3.8, RQ5;
after Sun et al. 2024) personalizes the graph by re-weighting each learner's
*own copy* of the edges from their feedback — a closed-form rule, no gradient
retraining:

  * per node, fold the learner's events into two gated signals,
    ``k`` (demonstrably knows) and ``w`` (demonstrably fails despite trying);
  * per edge ``u -> v`` (u prerequisite of v), agreement (`k_u·k_v`) reinforces
    both propagation messages, while a contradiction weakens exactly the
    message it falsifies:

      - ``k_v·w_u`` — knows the dependent, fails the prereq: the *backward*
        "v known ⇒ u known" inference is wrong for this learner;
      - ``k_u·w_v`` — knows the prereq, fails the dependent: the *forward*
        readiness signal is misleading.

  * multipliers are clamped to ``[kgt_factor_min, kgt_factor_max]``: a factor
    near 0 is edge *removal*, < 1 *weakened*, > 1 *strengthened* — §3.8's
    add/remove/reweight (adding edges outside the curriculum DAG is out of
    scope: the static map is the hypothesis space).

The factors multiply the raw alphas in ``propagation._build_edges`` *before*
per-target normalization, so propagation stays a convex combination and the
``inferred_ceiling`` guarantee is untouched: KGT redistributes inference, it
never fabricates "known". Like the rest of the engine, ``tune_edges`` is a pure
deterministic function of ``(graph, events, config, now)`` — the personal graph
is recomputed from the event stream, never stored.

Evidence gating makes the rule safe on sparse data: with no evidence on a node
both ``k`` and ``w`` are 0 (absence of evidence is never contradiction), and
``kgt_mass0`` is deliberately larger than the mastery score's ``mass0`` so only
*consistent* feedback (§3.8) moves an edge far.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import exp
from typing import TYPE_CHECKING

import networkx as nx

from klg_ai.core.events import Event
from klg_ai.core.evidence import reference_now
from klg_ai.core.forgetting import recency_weight

if TYPE_CHECKING:
    from klg_ai.core.activation import EngineConfig

__all__ = ["NodeFeedback", "EdgeAdjustment", "KgtResult", "node_feedback", "tune_edges"]


@dataclass(frozen=True)
class NodeFeedback:
    """A learner's gated feedback signals for one node.

    ``known`` and ``struggling`` are both accuracy × evidence-gate, so they sum
    to the gate (≤ 1) and are 0 without evidence; ``attempts``/``correct`` are
    raw causal event counts kept for human-readable adjustment reasons.
    """
    known: float       # k_n — demonstrably knows n
    struggling: float  # w_n — demonstrably fails n despite trying
    mass: float        # Σ source-weight · recency
    attempts: int
    correct: int


@dataclass(frozen=True)
class EdgeAdjustment:
    """One personalized edge re-weighting, with the evidence behind it."""
    source: str         # u — the prerequisite
    target: str         # v — the dependent
    factor_back: float  # multiplier on the backward (v -> u) inference message
    factor_fwd: float   # multiplier on the forward (u -> v) readiness message
    kind: str           # "strengthened" | "weakened" | "removed"
    reason: str         # human-readable, cites the evidence counts


@dataclass(frozen=True)
class KgtResult:
    """Personal edge factors (for propagation) + the reportable adjustments.

    ``factors`` covers only edges that deviate from 1; ``adjustments`` only
    those deviating by at least ``config.kgt_min_effect`` (the viewer-facing
    subset).
    """
    factors: dict[tuple[str, str], tuple[float, float]]
    adjustments: list[EdgeAdjustment]


def node_feedback(
    events: Iterable[Event],
    *,
    now: float | None = None,
    half_life_days: float = 30.0,
    forgetting: bool = True,
    mass0: float = 2.0,
) -> dict[str, NodeFeedback]:
    """Fold events into per-node (known, struggling) feedback signals.

    Same causal / source-weighted / recency-decayed discipline as
    ``evidence.direct_scores``, but split into the two signals KGT needs and
    gated harder (``mass0`` here demands repeated evidence, not one review).
    """
    events = list(events)
    if now is None:
        now = reference_now(events)

    pos: dict[str, float] = {}
    mass: dict[str, float] = {}
    attempts: dict[str, int] = {}
    correct: dict[str, int] = {}
    for e in events:
        if e.ts is not None and e.ts > now:
            continue  # causal: not yet observed at `now`
        age = 0.0 if e.ts is None else now - e.ts
        w = e.weight * recency_weight(age, half_life_days, enabled=forgetting)
        for node in e.node_ids:
            mass[node] = mass.get(node, 0.0) + w
            attempts[node] = attempts.get(node, 0) + 1
            if e.correct:
                pos[node] = pos.get(node, 0.0) + w
                correct[node] = correct.get(node, 0) + 1

    out: dict[str, NodeFeedback] = {}
    for node, m in mass.items():
        if m <= 0:
            continue
        acc = pos.get(node, 0.0) / m
        gate = 1.0 - exp(-m / mass0)
        out[node] = NodeFeedback(
            known=acc * gate,
            struggling=(1.0 - acc) * gate,
            mass=m,
            attempts=attempts.get(node, 0),
            correct=correct.get(node, 0),
        )
    return out


_REMOVED_AT = 0.1  # a factor this close to zero counts as edge removal


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _kind(factor: float) -> str:
    if factor <= _REMOVED_AT:
        return "removed"
    return "weakened" if factor < 1.0 else "strengthened"


def _label(g: nx.DiGraph, node: str) -> str:
    return g.nodes[node].get("label", node)


def _reason(
    g: nx.DiGraph, u: str, v: str, fu: NodeFeedback, fv: NodeFeedback,
    m_back: float, m_fwd: float,
) -> str:
    """Explain the dominant adjustment in terms of the learner's evidence."""
    lu, lv = _label(g, u), _label(g, v)
    ev_u = f"{fu.correct}/{fu.attempts} correct on '{lu}'"
    ev_v = f"{fv.correct}/{fv.attempts} correct on '{lv}'"
    dev_back, dev_fwd = abs(m_back - 1.0), abs(m_fwd - 1.0)
    if m_back < 1.0 and dev_back >= dev_fwd:
        return (f"Knows '{lv}' ({ev_v}) but struggles with its prerequisite "
                f"'{lu}' ({ev_u}) — backward inference ×{m_back:.2f}")
    if m_fwd < 1.0 and dev_fwd > dev_back:
        return (f"Knows the prerequisite '{lu}' ({ev_u}) but still struggles "
                f"with '{lv}' ({ev_v}) — readiness signal ×{m_fwd:.2f}")
    return (f"Consistent evidence on both '{lu}' ({ev_u}) and '{lv}' ({ev_v}) "
            f"— edge reinforced ×{max(m_back, m_fwd):.2f}")


def tune_edges(
    g: nx.DiGraph,
    events: Iterable[Event],
    config: "EngineConfig",
    *,
    now: float | None = None,
) -> KgtResult:
    """Per-learner edge factors from the learner's own feedback (closed form).

    Pure and deterministic; returns identity (empty) factors when the learner
    has no evidence on any edge's endpoints.
    """
    feedback = node_feedback(
        events,
        now=now,
        half_life_days=config.half_life_days,
        forgetting=config.forgetting,
        mass0=config.kgt_mass0,
    )

    factors: dict[tuple[str, str], tuple[float, float]] = {}
    adjustments: list[EdgeAdjustment] = []
    for u, v in g.edges:  # u is a prerequisite of v
        fu, fv = feedback.get(u), feedback.get(v)
        if fu is None or fv is None:
            continue  # no contradiction (or agreement) without evidence on both ends
        agree = fu.known * fv.known
        contra_back = fv.known * fu.struggling   # falsifies "v known ⇒ u known"
        contra_fwd = fu.known * fv.struggling    # falsifies "u known ⇒ ready for v"
        m_back = _clamp(1.0 + config.kgt_gain_up * agree - config.kgt_gain_down * contra_back,
                        config.kgt_factor_min, config.kgt_factor_max)
        m_fwd = _clamp(1.0 + config.kgt_gain_up * agree - config.kgt_gain_down * contra_fwd,
                       config.kgt_factor_min, config.kgt_factor_max)
        if m_back == 1.0 and m_fwd == 1.0:
            continue
        factors[(u, v)] = (m_back, m_fwd)
        if max(abs(m_back - 1.0), abs(m_fwd - 1.0)) >= config.kgt_min_effect:
            dominant = m_back if abs(m_back - 1.0) >= abs(m_fwd - 1.0) else m_fwd
            adjustments.append(EdgeAdjustment(
                source=u,
                target=v,
                factor_back=round(m_back, 4),
                factor_fwd=round(m_fwd, 4),
                kind=_kind(dominant),
                reason=_reason(g, u, v, fu, fv, m_back, m_fwd),
            ))
    return KgtResult(factors=factors, adjustments=adjustments)
