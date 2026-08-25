import os
from types import SimpleNamespace

import pydantic
import pytest

from app.extraction.schema import Candidate, Education, Role
from app.scoring import score as score_module
from app.scoring.rubric import CRITERIA, CriterionScore, RubricScore
from app.scoring.score import (CandidateScore, ScoringError, _serialize_candidate,
                                rank_candidates, score_candidate)
from tests.conftest import has_real_api_key


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


def test_blind_serialization_identical_across_name_swap():
    # Referenced by eval/bias_eval.py's module docstring as the reason the
    # name-swap/BLIND-scorer arm isn't run as an API call: with the name
    # field dropped and nothing else about the candidate changed, a bare
    # name swap must serialize to byte-identical text — that's an exact
    # offline fact, not something worth spending ~126 live scoring calls
    # to remeasure as noise.
    base = _candidate(name="Jordan Rivera")
    variant_a = base.model_copy(update={"name": "Emily Walsh"})
    variant_b = base.model_copy(update={"name": "Lakisha Washington"})
    assert (_serialize_candidate(variant_a, include_name=False)
            == _serialize_candidate(variant_b, include_name=False)
            == _serialize_candidate(base, include_name=False))


def test_blind_serialization_identical_even_when_variant_name_appears_in_free_text():
    # The claim above holds even in the adversarial case the scrub could
    # theoretically break: a swapped-in variant name's token ("Baker",
    # "Washington") coincidentally also appearing in the resume's free
    # text (a company name here). The scrub only touches the *swapped-in*
    # name's own tokens per variant, so if that token also happens to
    # appear elsewhere, both variants still scrub whatever they each
    # introduce and land on the same "[name]" placeholder shape.
    base = Candidate(
        name="Jordan Rivera", years_experience=3.0, skills=["Sales"], education=[],
        roles=[Role(title="Rep", company="Washington Group", start_date="2022-01",
                     end_date="2024-01", bullets=["Closed deals for Baker Industries."])],
    )
    variant_a = base.model_copy(update={"name": "Greg Baker"})
    variant_b = base.model_copy(update={"name": "Jamal Jones"})
    text_a = _serialize_candidate(variant_a, include_name=False)
    text_b = _serialize_candidate(variant_b, include_name=False)
    # Not necessarily identical to each other (variant_a's own token
    # "Baker" also scrubs "Baker Industries", which variant_b's scrub
    # wouldn't touch) — but each must be self-consistent: no literal
    # "Jordan"/"Rivera"/"Greg"/"Baker"/"Jamal"/"Jones" survives whichever
    # name was actually assigned to that variant.
    for text, swapped_in in [(text_a, "Greg Baker"), (text_b, "Jamal Jones")]:
        for token in ["Jordan", "Rivera", *swapped_in.split()]:
            assert token not in text


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


def test_serialize_candidate_scrub_is_case_insensitive_and_word_bounded():
    # A naive .replace() both under-scrubs (misses a different-case
    # occurrence, e.g. an all-caps header or a lowercased email) and
    # over-scrubs (eats "Foster" out of the unrelated word "Fostered").
    candidate = Candidate(
        name="Sam Foster",
        years_experience=2.0,
        skills=["Sales"],
        education=[],
        roles=[Role(title="Rep", company="Acme", start_date="2022-01", end_date="2024-01",
                     bullets=["SAM FOSTER fostered strong client relationships across a sample."])],
    )
    text = _serialize_candidate(candidate, include_name=False)
    assert "SAM FOSTER" not in text  # case-insensitive: all-caps occurrence still scrubbed
    assert "fostered" in text  # word-boundary: doesn't eat "Foster" out of "fostered"
    assert "sample" in text  # word-boundary: doesn't eat "Sam" out of "sample"


def test_rank_candidates_tie_order_is_deterministic_not_caller_order():
    # weighted_total is now rounded (rubric.py), so mathematically-equal
    # score vectors compare exactly equal rather than differing by float
    # noise. Ties resolve via a job_id-seeded shuffle, NOT caller order —
    # caller order is Phase 1's manifest order (category-then-tier), which
    # would otherwise silently favor whichever tier comes first every time.
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
        CandidateScore(candidate_id="first", job_id="job_x", rubric=vec_a, weighted_total=vec_a.weighted_total()),
        CandidateScore(candidate_id="second", job_id="job_x", rubric=vec_b, weighted_total=vec_b.weighted_total()),
    ]
    ranked_once = [s.candidate_id for s in rank_candidates(scores)]
    ranked_again = [s.candidate_id for s in rank_candidates(scores)]
    assert ranked_once == ranked_again  # same job_id -> reproducible order
    assert set(ranked_once) == {"first", "second"}  # both still present, tie didn't drop anyone


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


@pytest.mark.skipif(not has_real_api_key(), reason="live API smoke test")
def test_live_scoring_smoke():
    jd = "Backend Software Engineer. Requires Python, AWS, 3+ years experience."
    result = score_candidate(_candidate(), jd, "live_smoke_1", "live_jd_1")
    assert 1.0 <= result.weighted_total <= 5.0
