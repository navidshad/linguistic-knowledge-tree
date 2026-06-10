"""POST /api/chat — a Gemini tutor conversation that activates the map live.

Stateless (mirrors ``POST /api/status``): the client sends the full message history
+ its current known-set; the server maps each *learner* turn to concept nodes (the
Phase-6 semantic mapper, K-BERT arm), folds them as ``dialog`` evidence into
point-in-time mastery, and returns the tutor reply alongside the recomputed
knowledge state and the per-node text evidence (which turns lit each node — 6-B).

Evidence is **graded**, not assumed: the mapper only *proposes* candidate concepts;
``gemini_grade`` judges per candidate whether the turn really uses it and whether
it is used correctly. Wrong usage becomes negative evidence (``Event.correct=False``
lowers mastery), unused candidates are pruned (the mapper's false positives), and
in mock/keyless mode grading fails open to all-correct so the offline demo and
tests behave as before. Grades are cached per (text, candidates) because the
stateless client resends the whole history each request.
"""
from __future__ import annotations

import os
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache

from fastapi import APIRouter

from klg_ai.activation import (
    DEFAULT_CONFIG,
    EngineConfig,
    mastery_from_events,
    threshold_activated,
)
from klg_ai.events import Event
from klg_ai.semantic.mapper import default_mapper
from klg_ai.status import compute_status

from .. import trace
from ..deps import get_graph, get_map
from ..gemini import gemini_grade, gemini_reply
from ..schemas import ChatIn, ChatOut, NodeEvidenceOut

router = APIRouter(prefix="/api", tags=["chat"])

# Grades per (turn text, candidate ids): the stateless client resends the full
# history every request, so without this only-the-newest-turn caching each turn
# would be re-graded (one Gemini call per historical turn per request).
_GRADE_CACHE: dict[tuple[str, tuple[str, ...]], dict[str, dict]] = {}


def _graded(text: str, node_ids: tuple[str, ...]) -> dict[str, dict]:
    """Grade a turn's candidate nodes (cached); ``KLG_CHAT_GRADE=0`` disables grading."""
    if os.environ.get("KLG_CHAT_GRADE", "1") == "0":
        return {n: {"used": True, "correct": True} for n in node_ids}
    key = (text, node_ids)
    if key not in _GRADE_CACHE:
        m = get_map()
        _GRADE_CACHE[key] = gemini_grade(text, [(n, m.node(n).label) for n in node_ids])
    return _GRADE_CACHE[key]


@lru_cache(maxsize=1)
def _mapper():
    """The K-BERT (knowledge-injected) mapper for the demo, built once.

    Loads the embedder lazily on first use; the graph-aware readiness bias uses
    the learner's carried known-set (``active_nodes``). A free chat turn embeds to
    a "mixed" point, so similarities run lower than for a single tagged token — the
    demo threshold (``KLG_CHAT_THRESHOLD``, default 0.22, with top_k=3) is well below
    the RQ2 default so most turns light up a few plausible candidate concepts.
    """
    threshold = float(os.environ.get("KLG_CHAT_THRESHOLD", "0.22"))
    return default_mapper(knowledge_injection=True, threshold=threshold, top_k=3, syntax_map=get_map())


@router.post("/chat", response_model=ChatOut)
def chat(body: ChatIn) -> ChatOut:
    mapper = _mapper()
    active = set(body.activated)

    events: list[Event] = []
    # node_id -> {"confidence": float, "turns": set[int], "wrong": set[int]}
    evidence: dict[str, dict] = {}
    latest_scores: dict[str, float] = {}
    latest_grades: dict[str, bool] = {}
    latest_idx: int | None = None
    latest_text: str | None = None
    for i, turn in enumerate(body.messages):
        if turn.role != "user":
            continue
        latest_idx, latest_text = i, turn.text
        match = mapper.map_text(turn.text, active_nodes=active)
        grades = _graded(turn.text, match.node_ids)
        used = {n for n in match.node_ids if grades.get(n, {}).get("used", True)}
        right = tuple(n for n in used if grades[n].get("correct", True))
        wrong = tuple(n for n in used if not grades[n].get("correct", True))
        # Event.correct is per event, so correct and incorrect usage split in two.
        if right:
            events.append(Event(learner_id="chat", node_ids=right, correct=True,
                                ts=float(i), source="dialog"))
        if wrong:
            events.append(Event(learner_id="chat", node_ids=wrong, correct=False,
                                ts=float(i), source="dialog"))
        latest_scores = {n: match.scores[n] for n in match.node_ids if n in used}
        latest_grades = {n: n in right for n in latest_scores}
        for nid in used:
            slot = evidence.setdefault(nid, {"confidence": 0.0, "turns": set(), "wrong": set()})
            slot["confidence"] = max(slot["confidence"], match.scores[nid])
            slot["turns"].add(i)
            if nid in wrong:
                slot["wrong"].add(i)

    reply = gemini_reply(body.messages)

    g = get_graph()
    mastery = mastery_from_events(g, events, DEFAULT_CONFIG)
    activated = threshold_activated(mastery)
    statuses = {n: s.value for n, s in compute_status(g, activated).items()}

    if trace.enabled():
        _trace_turn(g, body, events, mastery, statuses,
                    latest_idx, latest_text, latest_scores, latest_grades, reply)

    return ChatOut(
        reply=reply,
        mapped_nodes=list(latest_scores),
        confidences={k: round(v, 3) for k, v in latest_scores.items()},
        grades=latest_grades,
        counts=dict(Counter(statuses.values())),
        statuses=statuses,
        mastery={n: round(m, 3) for n, m in mastery.items()},
        evidence=[
            NodeEvidenceOut(node_id=nid, confidence=round(d["confidence"], 3),
                            turn_indices=sorted(d["turns"]),
                            incorrect_turn_indices=sorted(d["wrong"]))
            for nid, d in sorted(evidence.items())
        ],
    )


def _trace_turn(g, body, events, mastery, statuses,
                latest_idx, latest_text, latest_scores, latest_grades, reply):
    """Append this turn's pipeline journey to the per-session trace file.

    Recomputes the *direct* (GNN-off) scores so the trace shows each touched node's
    evidence → GNN-lift → mastery → status — the full journey of the input.
    """
    direct = mastery_from_events(g, events, EngineConfig(propagation=False))
    touched = sorted({n for ev in events for n in ev.node_ids})

    def _row(n: str) -> dict:
        return {
            "node": n,
            "direct": round(direct.get(n, 0.0), 3),
            "gnn_lift": round(mastery.get(n, 0.0) - direct.get(n, 0.0), 3),
            "mastery": round(mastery.get(n, 0.0), 3),
            "status": statuses.get(n),
        }

    # Nodes the GNN lifted beyond their direct evidence — the interior-gap "glow"
    # the learner never produced directly (this is where propagation's effect shows).
    inferred = sorted(
        (_row(n) for n in mastery if mastery.get(n, 0.0) - direct.get(n, 0.0) > 1e-6),
        key=lambda r: r["gnn_lift"], reverse=True,
    )[:10]

    trace.record(
        body.session_id or "default",
        {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "turn": latest_idx,
            "input": latest_text,
            "mapper": {
                "arm": "kbert",
                "matched": [{"node": k, "cosine": round(v, 3),
                             "correct": latest_grades.get(k)} for k, v in latest_scores.items()],
            },
            "events_total": len(events),
            "touched": [_row(n) for n in touched],          # nodes with direct dialog evidence
            "gnn_inferred": inferred,                        # nodes the GNN lifted (interior gaps)
            "status_counts": dict(Counter(statuses.values())),
            "tutor_reply": reply,
        },
    )
