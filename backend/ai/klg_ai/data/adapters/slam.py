"""Duolingo SLAM adapter: real second-language-acquisition data -> ``Event`` stream.

The 2018 Duolingo Shared Task on Second Language Acquisition Modeling (SLAM)
releases per-token graded exercises from learners' first ~30 days on Duolingo.
We use the **en_es** track — learners *learning English* (L1 Spanish), so the
tokens are English and carry the morphosyntactic annotation we map onto the
syntax map. (es_en / fr_en are Spanish / French tokens and are ignored here.)

Data format (one exercise per blank-line-delimited block), mirrored from the
official ``baseline.py`` loader so our parse matches the shared task exactly::

    # prompt:Yo soy un niño.                                      (L1 prompt, optional)
    # user:XEinXf5+  countries:CO  days:0.003  client:web  session:lesson  format:reverse_translate  time:9
    DRihrVmh0101  I    PRON  Case=Nom|...|fPOS=PRON++PRP   nsubj  4  0
    DRihrVmh0102  am   VERB  ...|Tense=Pres|...|fPOS=VERB++VBP  cop  4  0
    ...

Token line fields: ``instance_id token POS morpho_features dep_label head [label]``.
``label`` is present only in ``.train`` (dev/test labels live in a separate
``.key`` file); **label 1 = the learner made a mistake, 0 = correct** — so a
learner ``Event`` is correct iff the SLAM label is 0.

The 12-char ``instance_id`` decomposes as ``[:8]`` session, ``[8:10]`` exercise
index, ``[10:12]`` token index; ``head`` is the 1-based token index of the
dependency head within the same exercise (0 = ROOT). ``days`` (float, larger =
more recent) maps straight onto ``Event.ts`` (the engine's day-based clock).

This module only *parses*; the morphosyntactic tag -> concept-node mapping lives
in ``slam_mapping.py`` so the rule table can be read, tested, and refined on its
own.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from klg_ai.core.events import Event


@dataclass(frozen=True)
class SlamToken:
    """One annotated token instance (one graded sub-answer) in an exercise."""
    instance_id: str
    token: str
    pos: str                     # Universal POS tag (NOUN, VERB, PRON, DET, ...)
    penn: str                    # fine-grained Penn tag from fPOS (VBP, NNS, MD, ...) or ""
    feats: dict[str, str]        # UD morphological features (Tense=Pres, Number=Plur, ...)
    dep_label: str               # dependency relation (nsubj, det, cop, aux, ...)
    head: int                    # token index of the dependency head (0 = ROOT)
    label: int | None = None     # 1 = mistake, 0 = correct, None = unlabeled

    @property
    def token_index(self) -> int:
        return int(self.instance_id[10:12])

    @property
    def correct(self) -> bool | None:
        """Learner correctness: True iff the SLAM label is 0 (no mistake)."""
        return None if self.label is None else (self.label == 0)


@dataclass(frozen=True)
class SlamExercise:
    """One exercise: shared session metadata plus its ordered token instances."""
    user: str
    days: float
    client: str
    session: str
    format: str
    time: int | None
    countries: tuple[str, ...]
    prompt: str | None
    tokens: tuple[SlamToken, ...]


def _parse_feats(field: str) -> tuple[dict[str, str], str]:
    """Split the morpho-features field into a dict and pull out the Penn tag.

    ``Mood=Ind|...|fPOS=VERB++VBP`` -> ({"Mood": "Ind", ...}, "VBP"). A bare
    ``_`` (no features) yields an empty dict.
    """
    feats: dict[str, str] = {}
    penn = ""
    if field and field != "_":
        for part in field.split("|"):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            if key == "fPOS":
                penn = value.split("++", 1)[1] if "++" in value else value
            else:
                feats[key] = value
    return feats, penn


def parse_key(path: str | Path) -> dict[str, int]:
    """Load a ``.key`` label file: ``{instance_id: label}`` (label 0/1)."""
    out: dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            iid, lab = line.split()
            out[iid] = int(float(lab))
    return out


def iter_exercises(
    path: str | Path, *, key: dict[str, int] | None = None
) -> Iterator[SlamExercise]:
    """Stream ``SlamExercise`` blocks from a SLAM ``.train``/``.dev``/``.test`` file.

    Labels come inline for ``.train``; for dev/test pass the matching ``.key``
    dict via ``key`` to attach them (instances absent from ``key`` keep
    ``label=None``).
    """
    props: dict = {}
    tokens: list[SlamToken] = []

    def flush() -> SlamExercise | None:
        if not tokens:
            return None
        return SlamExercise(
            user=props.get("user", ""),
            days=float(props.get("days", 0.0)),
            client=props.get("client", ""),
            session=props.get("session", ""),
            format=props.get("format", ""),
            time=props.get("time"),
            countries=tuple(props.get("countries", ())),
            prompt=props.get("prompt"),
            tokens=tuple(tokens),
        )

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                ex = flush()
                if ex is not None:
                    yield ex
                props, tokens = {}, []
            elif line[0] == "#":
                if line.startswith("# prompt:") or line[2:].startswith("prompt:"):
                    props["prompt"] = line.split(":", 1)[1]
                else:
                    for param in line[2:].split():
                        k, v = param.split(":", 1)
                        if k == "countries":
                            props[k] = tuple(v.split("|"))
                        elif k == "days":
                            props[k] = float(v)
                        elif k == "time":
                            props[k] = None if v == "null" else int(v)
                        else:
                            props[k] = v
            else:
                parts = line.split()
                iid = parts[0]
                feats, penn = _parse_feats(parts[3])
                label: int | None
                if len(parts) >= 7:
                    label = int(float(parts[6]))
                elif key is not None:
                    label = key.get(iid)
                else:
                    label = None
                tokens.append(SlamToken(
                    instance_id=iid, token=parts[1], pos=parts[2], penn=penn,
                    feats=feats, dep_label=parts[4], head=int(parts[5]), label=label,
                ))
        ex = flush()
        if ex is not None:
            yield ex


def slam_events(
    exercises: Iterable[SlamExercise], *, source: str = "review"
) -> list[Event]:
    """Turn parsed exercises into the engine's ``Event`` stream.

    Each labeled token whose tags map to one or more concept nodes (via
    ``slam_mapping.map_exercise``) becomes one ``Event`` tagged to those nodes,
    correct iff the learner made no mistake, timestamped at the exercise's
    ``days``. Tokens that map to nothing (function words, proper nouns) or have
    no label are skipped. All SLAM interactions are graded, so ``source`` is
    ``"review"`` (the strongest evidence tier) by default.
    """
    from klg_ai.data.adapters.slam_mapping import map_exercise  # local import avoids a cycle

    events: list[Event] = []
    for ex in exercises:
        node_map = map_exercise(ex)
        for tok in ex.tokens:
            nodes = node_map.get(tok.instance_id)
            if not nodes or tok.correct is None:
                continue
            events.append(Event(
                learner_id=ex.user, node_ids=nodes, correct=tok.correct,
                ts=ex.days, source=source,
            ))
    return events
