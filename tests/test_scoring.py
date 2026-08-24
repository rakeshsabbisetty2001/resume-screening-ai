import os
from types import SimpleNamespace

import pydantic
import pytest

from app.extraction.schema import Candidate, Education, Role
from app.scoring import score as score_module
from app.scoring.rubric import CRITERIA, CriterionScore, RubricScore
from app.scoring.score import (CandidateScore, ScoringError, _serialize_candidate,
                                rank_candidates, score_candidate)


def _candidate(name="Jordan Rivera") -> Candidate:
    return Candidate(
        name=name,
        years_experience=5.0,
        skills=["Python", "AWS"],
        education=[Education(degree="B.S. Computer Science", institution="State University")],
        roles=[Role(title="Software Engineer", company="Acme", start_date="2019-01",
                     end_date="2024-01", bullets=["Built internal tools."])],
    )


def _rubric(score=4) -> RubricScore:
    c = CriterionScore(score=score, rationale="ok")
    return RubricScore(skills_match=c, experience_fit=c, education_fit=c, role_relevance=c)


class _FakeResponse:
    def __init__(self, stop_reason, parsed_output, output_tokens=100):
        self.stop_reason = stop_reason
        self.parsed_output = parsed_output
        self.usage = SimpleNamespace(output_tokens=output_tokens)


def test_rubric_weights_sum_to_one():
    assert abs(sum(CRITERIA.values()) - 1.0) < 1e-9


def test_weighted_total_math():
    rubric = _rubric(score=4)
    assert abs(rubric.weighted_total() - 4.0) < 1e-9  # all criteria scored 4, weights sum to 1 -> 4.0

    mixed = RubricScore(
        skills_match=CriterionScore(score=5, rationale="x"),
        experience_fit=CriterionScore(score=3, rationale="x"),
        education_fit=CriterionScore(score=1, rationale="x"),
        role_relevance=CriterionScore(score=2, rationale="x"),
    )
    expected = 0.35 * 5 + 0.30 * 3 + 0.10 * 1 + 0.25 * 2
    assert abs(mixed.weighted_total() - expected) < 1e-9


def test_serialize_candidate_name_blind_by_default():
    text = _serialize_candidate(_candidate(), include_name=False)
    assert "Jordan Rivera" not in text


def test_serialize_candidate_include_name_opt_in():
    text = _serialize_candidate(_candidate(), include_name=True)
    assert "Jordan Rivera" in text


def test_serialize_candidate_scrubs_name_from_free_text():
    # Dropping the `name` field alone isn't enough — a real (or Phase 1
    # synthetic) resume can restate the candidate's name inside a bullet.
    leaky = Candidate(
        name="Jordan Rivera",
        years_experience=3.0,
        skills=["Python"],
        education=[],
        roles=[Role(title="Engineer", company="Acme", start_date="2021-01", end_date="2024-01",
                     bullets=["Jordan Rivera led the migration project."])],
    )
    text = _serialize_candidate(leaky, include_name=False)
    assert "Jordan" not in text
    assert "Rivera" not in text
    assert "[name]" in text


def test_rank_candidates_preserves_caller_order_on_true_ties():
    # weighted_total is now rounded (rubric.py), so mathematically-equal
    # score vectors compare exactly equal rather than differing by float
    # noise — this is what makes "stable sort, caller order on ties" true.
    vec_a = RubricScore(  # 0.35*1 + 0.30*1 + 0.10*1 + 0.25*3 = 1.5
        skills_match=CriterionScore(score=1, rationale="x"),
        experience_fit=CriterionScore(score=1, rationale="x"),
        education_fit=CriterionScore(score=1, rationale="x"),
        role_relevance=CriterionScore(score=3, rationale="x"),
    )
    vec_b = RubricScore(  # 0.35*1 + 0.30*2 + 0.10*3 + 0.25*1 = 1.5, differently composed
        skills_match=CriterionScore(score=1, rationale="x"),
        experience_fit=CriterionScore(score=2, rationale="x"),
        education_fit=CriterionScore(score=3, rationale="x"),
        role_relevance=CriterionScore(score=1, rationale="x"),
    )
    assert vec_a.weighted_total() == vec_b.weighted_total() == 1.5
    scores = [
        CandidateScore(candidate_id="first", job_id="j", rubric=vec_a, weighted_total=vec_a.weighted_total()),
        CandidateScore(candidate_id="second", job_id="j", rubric=vec_b, weighted_total=vec_b.weighted_total()),
    ]
    ranked = rank_candidates(scores)
    assert [s.candidate_id for s in ranked] == ["first", "second"]  # caller order preserved


def test_rank_candidates_sorts_descending():
    scores = [
        CandidateScore(candidate_id="a", job_id="j", rubric=_rubric(2), weighted_total=2.0),
        CandidateScore(candidate_id="b", job_id="j", rubric=_rubric(5), weighted_total=5.0),
        CandidateScore(candidate_id="c", job_id="j", rubric=_rubric(3), weighted_total=3.0),
    ]
    ranked = rank_candidates(scores)
    assert [s.candidate_id for s in ranked] == ["b", "c", "a"]


def test_score_candidate_wraps_validation_error_without_leaking_text(monkeypatch):
    def fake_parse(**kwargs):
        raise pydantic.ValidationError.from_exception_data(
            "RubricScore", [{"type": "json_invalid", "loc": (),
                              "input": '{"skills_match": {"score": 5, "rationale": "Jordan Rivera is',
                              "ctx": {"error": "EOF"}}])
    monkeypatch.setattr(score_module.client.messages, "parse", fake_parse)
    with pytest.raises(ScoringError) as exc_info:
        score_candidate(_candidate(), "some JD text", "cand_1", "job_1")
    assert "Jordan Rivera" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


def test_score_candidate_refusal_raises_cleanly(monkeypatch):
    monkeypatch.setattr(score_module.client.messages, "parse",
                         lambda **kwargs: _FakeResponse("refusal", None))
    with pytest.raises(ScoringError, match="refused"):
        score_candidate(_candidate(), "some JD text", "cand_1", "job_1")


def test_score_candidate_none_output_raises_cleanly(monkeypatch):
    monkeypatch.setattr(score_module.client.messages, "parse",
                         lambda **kwargs: _FakeResponse("end_turn", None))
    with pytest.raises(ScoringError, match="no parsed output"):
        score_candidate(_candidate(), "some JD text", "cand_1", "job_1")


def test_score_candidate_happy_path_omits_name_from_request(monkeypatch):
    captured = {}

    def fake_parse(**kwargs):
        captured["messages"] = kwargs["messages"]
        return _FakeResponse("end_turn", _rubric(4), output_tokens=250)

    monkeypatch.setattr(score_module.client.messages, "parse", fake_parse)
    logged = {}
    monkeypatch.setattr(score_module, "log_scoring",
                         lambda *args, **kwargs: logged.update(total=args[5]))

    result = score_candidate(_candidate(), "some JD text", "cand_1", "job_1")
    assert isinstance(result, CandidateScore)
    assert abs(result.weighted_total - 4.0) < 1e-9
    assert logged["total"] == result.weighted_total
    sent_content = captured["messages"][0]["content"]
    assert "Jordan Rivera" not in sent_content  # name-blind by default, end to end


def test_score_candidate_include_name_true_sends_name(monkeypatch):
    captured = {}

    def fake_parse(**kwargs):
        captured["messages"] = kwargs["messages"]
        return _FakeResponse("end_turn", _rubric(4))

    monkeypatch.setattr(score_module.client.messages, "parse", fake_parse)
    monkeypatch.setattr(score_module, "log_scoring", lambda *a, **k: None)

    score_candidate(_candidate(), "some JD text", "cand_1", "job_1", include_name=True)
    assert "Jordan Rivera" in captured["messages"][0]["content"]


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="live API smoke test")
def test_live_scoring_smoke():
    jd = "Backend Software Engineer. Requires Python, AWS, 3+ years experience."
    result = score_candidate(_candidate(), jd, "live_smoke_1", "live_jd_1")
    assert 1.0 <= result.weighted_total <= 5.0
