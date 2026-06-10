"""Minimal Gemini client for the chat demo, with a deterministic offline mock.

The tutor reply is the *only* thing Gemini provides; the thesis-relevant part —
the learner's turns activating the map — runs locally regardless. So this never
raises: a missing key, the mock flag, or any network/HTTP error all fall back to a
deterministic canned reply, and the chat still lights up the map.

Set ``GEMINI_API_KEY`` to use the real API; set ``KLG_GEMINI_MOCK=1`` to force the
mock (tests / CI / offline demo).
"""
from __future__ import annotations

import os

_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

_SYSTEM = (
    "You are a friendly English conversation tutor. Keep replies short (1-2 "
    "sentences), respond naturally to the learner, and gently invite them to keep "
    "talking so they produce more English. Do not explicitly correct their grammar."
)


def _mock_reply(messages) -> str:
    """Deterministic canned tutor turn (no network), keyed off the last user text."""
    last = next((m.text for m in reversed(messages) if m.role == "user"), "")
    snippet = last.strip().rstrip(".!?")
    if not snippet:
        return "Hi! Tell me about your day — what did you do today?"
    return f'Nice — you said "{snippet}". Can you tell me a bit more about that?'


def gemini_reply(messages, *, system: str | None = None, timeout: float = 15.0) -> str:
    """A tutor reply for the conversation; falls back to a deterministic mock.

    ``messages`` are ``ChatTurn``-like objects with ``.role`` ("user"/"tutor") and
    ``.text``.
    """
    key = os.environ.get("GEMINI_API_KEY")
    if not key or os.environ.get("KLG_GEMINI_MOCK") == "1":
        return _mock_reply(messages)
    try:
        import httpx

        contents = [
            {"role": "user" if m.role == "user" else "model", "parts": [{"text": m.text}]}
            for m in messages
        ]
        payload = {
            "system_instruction": {"parts": [{"text": system or _SYSTEM}]},
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": 256,
                "temperature": 0.7,
                # Current flash models "think" by default and can spend the whole token
                # budget on hidden thought, truncating the visible reply; disable it for
                # short tutor turns. (Ignored by models without thinking — falls back safely.)
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
        resp = httpx.post(
            _ENDPOINT.format(model=_MODEL), params={"key": key}, json=payload, timeout=timeout
        )
        resp.raise_for_status()
        # Join all text parts (thinking models can split thought/answer across parts).
        parts = resp.json()["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts).strip()
        return text or _mock_reply(messages)
    except Exception:
        # Never fail the chat on a tutor-reply problem — the map activation is the point.
        return _mock_reply(messages)


def _grade_all_correct(candidates) -> dict[str, dict]:
    """Fail-open grading: every candidate used + correct (the pre-grading behavior)."""
    return {nid: {"used": True, "correct": True} for nid, _ in candidates}


def gemini_grade(text: str, candidates: list[tuple[str, str]], *,
                 timeout: float = 15.0) -> dict[str, dict]:
    """Grade a learner turn against the mapper's candidate concepts.

    ``candidates`` is ``[(node_id, label), ...]`` proposed by the semantic mapper.
    Returns ``{node_id: {"used": bool, "correct": bool}}`` — ``used`` prunes
    mapper false positives; ``correct=False`` becomes negative evidence (the
    engine's ``Event.correct``). Falls back to all-used/all-correct when mocked,
    keyless, or on any API/parse error, so grading can only *refine* the demo,
    never break it.
    """
    if not candidates:
        return {}
    key = os.environ.get("GEMINI_API_KEY")
    if not key or os.environ.get("KLG_GEMINI_MOCK") == "1":
        return _grade_all_correct(candidates)
    try:
        import json

        import httpx

        concept_lines = "\n".join(f"- {nid}: {label}" for nid, label in candidates)
        prompt = (
            "You grade English grammar usage. The learner wrote:\n"
            f'"{text}"\n\n'
            "For each concept below, judge:\n"
            '- "used": does the sentence actually attempt this grammatical construction?\n'
            '- "correct": if used, is it formed correctly here? (false when attempted with errors)\n\n'
            f"Concepts:\n{concept_lines}\n\n"
            'Reply with ONLY a JSON object mapping each concept id to {"used": bool, "correct": bool}.'
        )
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": 256,
                "temperature": 0.0,  # grading should be as deterministic as the API allows
                "responseMimeType": "application/json",
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
        resp = httpx.post(
            _ENDPOINT.format(model=_MODEL), params={"key": key}, json=payload, timeout=timeout
        )
        resp.raise_for_status()
        parts = resp.json()["candidates"][0]["content"]["parts"]
        raw = json.loads("".join(p.get("text", "") for p in parts))
        out: dict[str, dict] = {}
        for nid, _ in candidates:
            g = raw.get(nid)
            if isinstance(g, dict):
                used = bool(g.get("used", True))
                out[nid] = {"used": used, "correct": bool(g.get("correct", True)) if used else False}
            else:  # node missing from the response → fail open, don't drop evidence
                out[nid] = {"used": True, "correct": True}
        return out
    except Exception:
        return _grade_all_correct(candidates)
