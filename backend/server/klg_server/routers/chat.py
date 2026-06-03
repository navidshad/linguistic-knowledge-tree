"""POST /api/chat — a Gemini tutor conversation that activates the map live.

Stateless (mirrors ``POST /api/status``): the client sends the full message history
+ its current known-set; the server maps each *learner* turn to concept nodes (the
Phase-6 semantic mapper, K-BERT arm), folds them as ``dialog`` evidence into
point-in-time mastery, and returns the tutor reply alongside the recomputed
knowledge state and the per-node text evidence (which turns lit each node — 6-B).
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
from ..gemini import gemini_reply
from ..schemas import ChatIn, ChatOut, NodeEvidenceOut

router = APIRouter(prefix="/api", tags=["chat"])


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
    evidence: dict[str, dict] = {}  # node_id -> {"confidence": float, "turns": set[int]}
    latest_scores: dict[str, float] = {}
    latest_idx: int | None = None
    latest_text: str | None = None
    for i, turn in enumerate(body.messages):
        if turn.role != "user":
            continue
        latest_idx, latest_text = i, turn.text
        match = mapper.map_text(turn.text, active_nodes=active)
        if match.node_ids:
            events.append(
                Event(learner_id="chat", node_ids=match.node_ids, correct=True,
                      ts=float(i), source="dialog")
            )
        latest_scores = dict(match.scores)
        for nid, conf in match.scores.items():
            slot = evidence.setdefault(nid, {"confidence": 0.0, "turns": set()})
            slot["confidence"] = max(slot["confidence"], conf)
            slot["turns"].add(i)

    reply = gemini_reply(body.messages)

    g = get_graph()
    mastery = mastery_from_events(g, events, DEFAULT_CONFIG)
    activated = threshold_activated(mastery)
    statuses = {n: s.value for n, s in compute_status(g, activated).items()}

    if trace.enabled():
        _trace_turn(g, body, events, mastery, statuses, latest_idx, latest_text, latest_scores, reply)

    return ChatOut(
        reply=reply,
        mapped_nodes=list(latest_scores),
        confidences={k: round(v, 3) for k, v in latest_scores.items()},
        counts=dict(Counter(statuses.values())),
        statuses=statuses,
        mastery={n: round(m, 3) for n, m in mastery.items()},
        evidence=[
            NodeEvidenceOut(node_id=nid, confidence=round(d["confidence"], 3),
                            turn_indices=sorted(d["turns"]))
            for nid, d in sorted(evidence.items())
        ],
    )


def _trace_turn(g, body, events, mastery, statuses, latest_idx, latest_text, latest_scores, reply):
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
                "matched": [{"node": k, "cosine": round(v, 3)} for k, v in latest_scores.items()],
            },
            "events_total": len(events),
            "touched": [_row(n) for n in touched],          # nodes with direct dialog evidence
            "gnn_inferred": inferred,                        # nodes the GNN lifted (interior gaps)
            "status_counts": dict(Counter(statuses.values())),
            "tutor_reply": reply,
        },
    )
