"""Events -> direct mastery scores: source weighting, accuracy, forgetting."""
from klg_ai.events import Event
from klg_ai.evidence import direct_scores


def _ev(node, correct, ts=0.0, source="review"):
    return Event(learner_id="l", node_ids=(node,), correct=correct, ts=ts, source=source)


def test_no_events_no_scores():
    assert direct_scores([]) == {}


def test_correct_review_clears_known_level():
    s = direct_scores([_ev("present_simple", True)])
    assert s["present_simple"] >= 0.5


def test_review_beats_exposure():
    review = direct_scores([_ev("x", True, source="review")])["x"]
    exposure = direct_scores([_ev("x", True, source="exposure")])["x"]
    assert review > exposure
    assert exposure < 0.5  # a single passive exposure is not mastery


def test_incorrect_only_scores_zero():
    assert direct_scores([_ev("x", False)])["x"] == 0.0


def test_more_correct_evidence_raises_score():
    one = direct_scores([_ev("x", True)])["x"]
    three = direct_scores([_ev("x", True), _ev("x", True), _ev("x", True)])["x"]
    assert three > one


def test_forgetting_decays_old_evidence():
    ev = [_ev("x", True, ts=0.0)]
    fresh = direct_scores(ev, now=0.0)["x"]
    stale = direct_scores(ev, now=120.0, half_life_days=30.0)["x"]  # 4 half-lives
    assert stale < fresh
    assert stale < 0.5  # decayed below known


def test_forgetting_off_ignores_age():
    ev = [_ev("x", True, ts=0.0)]
    assert direct_scores(ev, now=120.0, forgetting=False)["x"] == direct_scores(ev, now=0.0)["x"]
