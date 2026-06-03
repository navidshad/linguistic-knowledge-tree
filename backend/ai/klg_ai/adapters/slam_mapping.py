"""SLAM morphosyntactic tags -> syntax-map concept nodes (the evidence mapping).

Each SLAM token carries a Universal POS tag, a fine-grained Penn tag (from
``fPOS``), UD morphological features, and a dependency relation + head. This
module turns those into the concept node id(s) the token is evidence *about* —
the real-data analogue of the rule-based "evidence -> node" mapping the roadmap
calls for (Phase 2-A, now on Duolingo data). Getting a token right/wrong is then
evidence for/against the mapped node(s).

Two rule layers:

* **Lexical / token-local** — articles, pronouns, plural nouns, comparatives,
  modals, prepositions, linkers, negation … readable straight off one token's
  POS + features + surface form.
* **Verb constructions** — periphrastic forms span several tokens, so we read
  the exercise's dependency tree: *be* + V-ing -> (present/past) continuous,
  *have* + V-en -> (present/past) perfect, *be* + V-en -> passive, modal + V ->
  the modal's concept, copular *be* -> present/past simple. The construction
  node is attributed to every token involved (main verb **and** its auxiliaries)
  so an error on either is evidence about it.

A single token can map to several nodes (e.g. plural *boys* -> plural_nouns; a
finite lexical verb -> present_simple). Function words and proper nouns that
carry no grammar signal map to nothing and are simply dropped as evidence. Every
emitted id is filtered against the live map, so a typo or a map edit can never
inject a phantom node.

The mapping is deliberately conservative: SLAM is beginners' first ~30 days, so
the A1–B1 core dominates. It is the part most worth refining as coverage is
measured (see ``klg_ai.eval``); the thesis treats it as a transparent, auditable
function, not a learned black box.
"""
from __future__ import annotations

from functools import lru_cache

from .slam import SlamExercise, SlamToken

# ---------------------------------------------------------------------------
# Lexical vocabularies (lower-cased surface forms)
# ---------------------------------------------------------------------------
_DEMONSTRATIVES = {"this", "that", "these", "those"}
_QUANTIFIERS = {
    "some", "any", "many", "much", "few", "little", "several", "all", "both",
    "each", "every", "no", "none", "enough", "most", "more", "lot", "lots",
    "plenty", "half",
}
_PERSONAL_PRONOUNS = {
    "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them",
}
_INDEFINITE_PRONOUNS = {
    "someone", "somebody", "something", "somewhere",
    "anyone", "anybody", "anything", "anywhere",
    "everyone", "everybody", "everything", "everywhere",
    "no-one", "nobody", "nothing", "nowhere",
}
_FREQUENCY_ADVERBS = {
    "always", "usually", "often", "sometimes", "never", "rarely", "seldom",
    "normally", "frequently", "occasionally", "ever", "generally",
}
_TIME_PLACE_ADVERBS = {
    "here", "there", "now", "then", "today", "tomorrow", "yesterday", "soon",
    "already", "yet", "still", "tonight", "abroad", "home", "outside", "inside",
    "away", "everyday", "nowadays",
}
_DEGREE_ADVERBS = {
    "very", "too", "quite", "really", "so", "enough", "almost", "nearly",
    "extremely", "fairly", "rather", "absolutely", "completely", "totally",
}
_LINKERS = {
    "because", "so", "then", "also", "first", "next", "finally", "however",
    "although", "though", "while", "therefore", "besides", "moreover",
    "secondly", "firstly",
}
_PREP_TIME = {"since", "until", "till", "during", "ago"}
_PREP_MOVEMENT = {
    "into", "onto", "towards", "toward", "through", "across", "along", "past",
    "up", "down", "out", "off", "away", "round", "around",
}
_PREP_PLACE = {
    "in", "on", "at", "under", "over", "behind", "between", "among", "amongst",
    "near", "beside", "above", "below", "inside", "outside", "by", "beneath",
    "beyond", "against", "opposite", "from", "of", "with", "about",
}
_MODAL_NODE = {
    "can": "can_ability", "cannot": "can_ability", "can't": "can_ability",
    "could": "could_past_ability", "couldn't": "could_past_ability",
    "must": "must_have_to", "mustn't": "must_have_to",
    "should": "should_advice", "shouldn't": "should_advice",
    "ought": "should_advice",
    "will": "will_future", "'ll": "will_future", "won't": "will_future",
    "shall": "will_future",
    "may": "might_may_possibility", "might": "might_may_possibility",
    "would": "would_past_habits", "'d": "would_past_habits",
    "wouldn't": "would_past_habits",
}
_RELATIVIZERS = {"who", "whom", "whose", "which", "that"}
_WH_WORDS = {"what", "where", "when", "why", "who", "whom", "whose", "which", "how"}

_BE_PRESENT = {"am", "is", "are", "'m", "'re"}
_BE_PAST = {"was", "were"}
_BE_NONFINITE = {"be", "been", "being"}
_HAVE_PRESENT = {"have", "has", "'ve"}
_HAVE_PAST = {"had", "'d"}
_NEGATORS = {"not", "n't", "never"}


@lru_cache(maxsize=1)
def _valid_nodes() -> frozenset[str]:
    """Concept ids present in the live map (mapping output is filtered to these)."""
    from ..loader import load_map
    return frozenset(load_map().node_ids)


def _norm(token: str) -> str:
    return token.lower()


def _lexical_nodes(tok: SlamToken) -> list[str]:
    """Concept nodes readable from a single token's tags + surface form."""
    t = _norm(tok.token)
    pos, penn, f = tok.pos, tok.penn, tok.feats
    out: list[str] = []

    # Determiners: articles, demonstratives, quantifiers
    if pos == "DET":
        if f.get("PronType") == "Art" or penn == "DT":
            if f.get("Definite") == "Def" or t == "the":
                out.append("articles_the")
            elif f.get("Definite") == "Ind" or t in {"a", "an"}:
                out.append("articles_a_an")
        if t in _DEMONSTRATIVES:
            out.append("demonstratives")
        if t in _QUANTIFIERS:
            out.append("quantifiers_basic")

    # Pronouns
    if t in _DEMONSTRATIVES and pos in {"PRON", "DET"}:
        out.append("demonstratives")
    if f.get("Reflex") == "Yes" or t.endswith(("self", "selves")):
        out.append("reflexive_pronouns")
    elif penn == "PRP$" or f.get("Poss") == "Yes":
        out.append("possessive_pronouns")
    elif t in _INDEFINITE_PRONOUNS:
        out.append("indefinite_pronouns")
    elif t in _PERSONAL_PRONOUNS and pos == "PRON":
        out.append("personal_pronouns")

    # Nouns: plural + possessive 's clitic
    if pos == "NOUN" and (penn == "NNS" or f.get("Number") == "Plur"):
        out.append("plural_nouns")
    if penn == "POS" or (pos == "PART" and t in {"'s", "'"}):
        out.append("possessive_s")

    # Adjectives: comparative / superlative / basic
    if pos == "ADJ":
        if penn == "JJR" or f.get("Degree") == "Cmp":
            out.append("comparatives")
        elif penn == "JJS" or f.get("Degree") == "Sup":
            out.append("superlatives")
        else:
            out.append("adjective_basic")
    if t == "more":
        out.append("comparatives")
    elif t == "most":
        out.append("superlatives")

    # Adverbs: comparative, frequency, degree, manner, time/place
    if pos == "ADV" or penn.startswith("RB"):
        if penn == "RBR":
            out.append("comparative_adverbs")
        elif t in _FREQUENCY_ADVERBS:
            out.append("adverbs_frequency")
        elif t in _DEGREE_ADVERBS:
            out.append("adverbs_degree")
        elif t in _TIME_PLACE_ADVERBS:
            out.append("adverbs_time_place")
        elif t.endswith("ly"):
            out.append("adverbs_manner")

    # Modals (lexical; constructions also re-attribute these to the main verb)
    if penn == "MD" or t in _MODAL_NODE:
        node = _MODAL_NODE.get(t)
        if node:
            out.append(node)

    # Coordination vs. discourse linkers
    if pos in {"CONJ", "CCONJ"} or penn == "CC":
        if t in {"and", "or", "but"}:
            out.append("coordination")
    if t in _LINKERS:
        out.append("basic_linkers")

    # Prepositions (skip infinitival "to": Penn TO as a clause marker)
    if pos == "ADP" or penn == "IN":
        if t in _PREP_TIME:
            out.append("prepositions_time")
        elif t in _PREP_MOVEMENT:
            out.append("prepositions_movement")
        elif t in _PREP_PLACE:
            out.append("prepositions_place")
        elif t == "to":
            out.append("prepositions_movement")

    # Negation and short answers
    if t in _NEGATORS or tok.dep_label in {"neg"}:
        if t != "never":  # already counted as a frequency adverb above
            out.append("negation_basic")
    if t in {"yes", "no"} and pos == "INTJ":
        out.append("short_answers")

    # Wh-words: relativizer inside a clause vs. question word
    if t in _WH_WORDS or penn in {"WP", "WDT", "WRB", "WP$"}:
        if t in _RELATIVIZERS and tok.token_index > 1 and tok.dep_label != "ROOT":
            out.append("relative_pronouns")
        else:
            out.append("wh_questions")

    return out


def _verb_construction_nodes(ex: SlamExercise) -> dict[int, list[str]]:
    """Nodes from periphrastic verb forms, keyed by token index.

    Reads the dependency tree: a main verb's auxiliary/copula children determine
    tense + aspect/voice. The resulting node is attributed to the main verb and
    each auxiliary token involved.
    """
    by_index: dict[int, SlamToken] = {t.token_index: t for t in ex.tokens}
    children: dict[int, list[int]] = {}
    for t in ex.tokens:
        children.setdefault(t.head, []).append(t.token_index)

    out: dict[int, list[str]] = {}

    def add(idx: int, node: str) -> None:
        out.setdefault(idx, []).append(node)

    has_to_infinitive = any(
        _norm(t.token) == "to" and (t.penn == "TO" or t.dep_label == "mark")
        for t in ex.tokens
    )
    has_negation = any(_norm(t.token) in {"not", "n't"} for t in ex.tokens)
    question = bool(ex.prompt and "?" in ex.prompt)

    for v in ex.tokens:
        if v.pos not in {"VERB", "AUX"}:
            continue
        kids = [by_index[c] for c in children.get(v.token_index, []) if c in by_index]
        aux = [k for k in kids if k.dep_label in {"aux", "auxpass", "cop"}]
        aux_be = [k for k in aux if _norm(k.token) in _BE_PRESENT | _BE_PAST | _BE_NONFINITE]
        aux_have = [k for k in aux if _norm(k.token) in _HAVE_PRESENT | _HAVE_PAST]
        aux_modal = [k for k in aux if k.penn == "MD" or _norm(k.token) in _MODAL_NODE]
        is_pres = lambda k: _norm(k.token) in _BE_PRESENT or _norm(k.token) in _HAVE_PRESENT
        is_past = lambda k: _norm(k.token) in _BE_PAST or _norm(k.token) in _HAVE_PAST

        # going to (future)
        if _norm(v.token) == "going" and (has_to_infinitive or v.penn == "VBG"):
            if has_to_infinitive:
                add(v.token_index, "going_to_future")
                for k in aux_be:
                    add(k.token_index, "going_to_future")
                continue

        # Continuous: be + V-ing
        if (v.penn == "VBG" or v.feats.get("VerbForm") == "Ger") and aux_be:
            node = "past_continuous" if any(is_past(k) for k in aux_be) else "present_continuous"
            add(v.token_index, node)
            for k in aux_be:
                add(k.token_index, node)
            continue

        # Passive: be + V-en  (auxpass marks the be)
        if v.penn == "VBN" and any(k.dep_label == "auxpass" for k in aux_be):
            add(v.token_index, "passive_present_past")
            for k in aux_be:
                add(k.token_index, "passive_present_past")
            continue

        # Perfect: have + V-en
        if v.penn == "VBN" and aux_have:
            node = "past_perfect_simple" if any(is_past(k) for k in aux_have) else "present_perfect_simple"
            add(v.token_index, node)
            for k in aux_have:
                add(k.token_index, node)
            continue

        # Modal + base verb -> the modal's concept (will -> will_future, etc.)
        if aux_modal:
            for k in aux_modal:
                node = _MODAL_NODE.get(_norm(k.token))
                if node:
                    add(v.token_index, node)
            continue

    # Copular / plain finite verbs (only if the verb isn't an auxiliary itself)
    aux_indices = {
        t.token_index for t in ex.tokens if t.dep_label in {"aux", "auxpass"}
    }
    for v in ex.tokens:
        if v.pos not in {"VERB", "AUX"} or v.token_index in aux_indices:
            continue
        if v.token_index in out:
            continue  # already part of a construction
        tense = v.feats.get("Tense")
        lemma_be = _norm(v.token) in _BE_PRESENT | _BE_PAST | _BE_NONFINITE
        if v.dep_label == "cop" or lemma_be:
            if tense == "Past" or _norm(v.token) in _BE_PAST:
                add(v.token_index, "past_simple")
            else:
                add(v.token_index, "present_simple")
        elif v.penn == "VBD" or tense == "Past":
            add(v.token_index, "past_simple")
        elif v.penn in {"VBP", "VBZ"} or tense == "Pres":
            add(v.token_index, "present_simple")

    # do-support: do/does/did as an auxiliary -> negation or yes/no question
    for t in ex.tokens:
        if _norm(t.token) in {"do", "does", "did"} and t.dep_label in {"aux", "auxpass"}:
            if has_negation:
                add(t.token_index, "negation_basic")
            elif question:
                add(t.token_index, "yes_no_questions")

    return out


def map_token(tok: SlamToken, ex: SlamExercise) -> tuple[str, ...]:
    """Concept node ids one token is evidence about (lexical rules only).

    Convenience for token-level callers/tests; ``map_exercise`` additionally
    layers in the dependency-based verb constructions.
    """
    valid = _valid_nodes()
    seen: list[str] = []
    for n in _lexical_nodes(tok):
        if n in valid and n not in seen:
            seen.append(n)
    return tuple(seen)


def map_exercise(ex: SlamExercise) -> dict[str, tuple[str, ...]]:
    """Map every token in an exercise to its concept node ids.

    Returns ``{instance_id: (node_id, ...)}`` for tokens that map to at least one
    valid node; tokens mapping to nothing are omitted. Combines the lexical rules
    with the dependency-aware verb constructions and de-duplicates per token.
    """
    valid = _valid_nodes()
    constructions = _verb_construction_nodes(ex)
    result: dict[str, tuple[str, ...]] = {}
    for tok in ex.tokens:
        nodes: list[str] = []
        for n in _lexical_nodes(tok) + constructions.get(tok.token_index, []):
            if n in valid and n not in nodes:
                nodes.append(n)
        if nodes:
            result[tok.instance_id] = tuple(nodes)
    return result
