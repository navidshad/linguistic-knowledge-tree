"""Duolingo SLAM adapter: parsing + morphosyntactic tag -> concept-node mapping.

Runs against a tiny in-repo fixture in the real SLAM format
(``tests/fixtures/slam/mini.slam.20190204.*``), so it needs none of the gated
dataset. Mapping assertions are traced by hand from the rules.
"""
from pathlib import Path

from klg_ai.data.adapters.slam import (
    SlamToken, iter_exercises, parse_key, slam_events,
)
from klg_ai.data.adapters.slam_mapping import _MODAL_NODE, map_exercise
from klg_ai.core.loader import load_map

FIX = Path(__file__).parent / "fixtures" / "slam"
TRAIN = FIX / "mini.slam.20190204.train"
DEV = FIX / "mini.slam.20190204.dev"
KEY = FIX / "mini.slam.20190204.dev.key"


def _exercises(path=TRAIN, key=None):
    return list(iter_exercises(path, key=key))


def _find(exs, marker):
    """The exercise containing a token whose surface form == marker."""
    return next(ex for ex in exs if any(t.token == marker for t in ex.tokens))


def _nodes(ex, token_text):
    """Mapped node ids for the first token with the given surface form."""
    nm = map_exercise(ex)
    tok = next(t for t in ex.tokens if t.token == token_text)
    return set(nm.get(tok.instance_id, ()))


# --- parsing ---------------------------------------------------------------

def test_parse_counts_and_fields():
    exs = _exercises()
    assert len(exs) == 9  # 5 learnerA + 4 learnerB exercises
    boy = _find(exs, "boy")
    assert boy.user == "learnerA"
    assert boy.days == 1.0
    assert boy.format == "reverse_translate"
    am = next(t for t in boy.tokens if t.token == "am")
    assert am.pos == "VERB" and am.penn == "VBP"
    assert am.feats["Tense"] == "Pres" and am.dep_label == "cop"
    assert am.head == 4


def test_label_semantics_correct_is_label_zero():
    boy_ex = _find(_exercises(), "boy")
    am = next(t for t in boy_ex.tokens if t.token == "am")
    assert am.label == 0 and am.correct is True
    reading = next(t for t in _find(_exercises(), "reading").tokens if t.token == "reading")
    assert reading.label == 1 and reading.correct is False


def test_dev_labels_come_from_key():
    key = parse_key(KEY)
    assert key["A00000200103"] == 1  # "a" marked a mistake in the key
    devs = _exercises(DEV, key=key)
    cat_ex = _find(devs, "cat")
    a_tok = next(t for t in cat_ex.tokens if t.token == "a")
    assert a_tok.label == 1 and a_tok.correct is False


# --- lexical mapping -------------------------------------------------------

def test_articles():
    exs = _exercises()
    assert _nodes(_find(exs, "boy"), "a") == {"articles_a_an"}
    assert _nodes(_find(exs, "books"), "the") == {"articles_the"}


def test_pronouns():
    exs = _exercises()
    assert _nodes(_find(exs, "boy"), "I") == {"personal_pronouns"}
    assert _nodes(_find(exs, "book"), "my") == {"possessive_pronouns"}


def test_plural_noun_and_comparative():
    exs = _exercises()
    assert _nodes(_find(exs, "books"), "books") == {"plural_nouns"}
    assert _nodes(_find(exs, "bigger"), "bigger") == {"comparatives"}


def test_negation_and_time_place_adverb():
    exs = _exercises()
    know_ex = _find(exs, "know")
    assert _nodes(know_ex, "not") == {"negation_basic"}
    assert _nodes(know_ex, "do") == {"negation_basic"}  # do-support under negation
    assert _nodes(_find(exs, "walked"), "home") == {"adverbs_time_place"}


def test_unmapped_tokens_are_dropped():
    boy_ex = _find(_exercises(), "boy")
    nm = map_exercise(boy_ex)
    boy = next(t for t in boy_ex.tokens if t.token == "boy")
    assert boy.instance_id not in nm  # a bare singular noun maps to no grammar node


# --- dependency-aware verb constructions -----------------------------------

def test_copula_and_lexical_present_simple():
    exs = _exercises()
    assert _nodes(_find(exs, "boy"), "am") == {"present_simple"}      # copula
    assert _nodes(_find(exs, "books"), "like") == {"present_simple"}  # lexical finite


def test_present_continuous_be_plus_ving():
    reading_ex = _find(_exercises(), "reading")
    assert _nodes(reading_ex, "reading") == {"present_continuous"}
    assert _nodes(reading_ex, "is") == {"present_continuous"}  # aux attributed too


def test_present_perfect_have_plus_ven():
    eaten_ex = _find(_exercises(), "eaten")
    assert _nodes(eaten_ex, "eaten") == {"present_perfect_simple"}
    assert _nodes(eaten_ex, "have") == {"present_perfect_simple"}


def test_modal_construction_marks_main_verb():
    swim_ex = _find(_exercises(), "swim")
    assert _nodes(swim_ex, "can") == {"can_ability"}
    assert _nodes(swim_ex, "swim") == {"can_ability"}


def test_past_simple():
    assert _nodes(_find(_exercises(), "walked"), "walked") == {"past_simple"}


def test_wh_question_word():
    devs = _exercises(DEV, key=parse_key(KEY))
    assert _nodes(_find(devs, "Where"), "Where") == {"wh_questions"}


# --- events + integrity ----------------------------------------------------

def test_slam_events_only_mapped_tokens_with_correctness():
    events = slam_events(_exercises())
    # "boy" maps to nothing -> no event; "am" -> present_simple, correct, ts=1.0
    am = next(e for e in events if e.node_ids == ("present_simple",) and e.ts == 1.0)
    assert am.correct is True and am.source == "review" and am.learner_id == "learnerA"
    assert all(e.node_ids for e in events)  # every event has at least one node


def test_all_modal_targets_are_valid_map_nodes():
    valid = load_map().node_ids
    assert set(_MODAL_NODE.values()) <= valid


def test_mapping_never_emits_unknown_nodes():
    valid = load_map().node_ids
    for ex in _exercises() + _exercises(DEV, key=parse_key(KEY)):
        for nodes in map_exercise(ex).values():
            assert set(nodes) <= valid
