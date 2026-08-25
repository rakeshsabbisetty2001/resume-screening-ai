"""Resilience tests for eval/run_eval.py and eval/bias_eval.py's
orchestration loops — added after a real crash on the first live run:
`anthropic.BadRequestError: Grammar compilation timed out` (a transient
upstream hiccup, not caused by this project's schema) propagated past
`score_candidate`'s own error contract (which only wraps
`pydantic.ValidationError`) and crashed the whole 121-call eval, discarding
every call already spent. These tests are offline — no API calls."""
import httpx
import pytest

import anthropic
from app.extraction.extract import ExtractionError
from app.extraction.schema import Candidate, Education, Role
from app.scoring.score import ScoringError
from eval import run_eval as run_eval_module
from eval import bias_eval as bias_eval_module


def _fake_api_error() -> anthropic.APIError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIError("Grammar compilation timed out", request, body=None)


def _candidate(name="Jordan Rivera"):
    return Candidate(
        name=name, years_experience=3.0, skills=["Python"],
        education=[Education(degree="B.S. Computer Science", institution="State University")],
        roles=[Role(title="Engineer", company="Acme", start_date="2021-01",
                     end_date="2024-01", bullets=[])],
    )


def test_run_extraction_eval_survives_transient_api_error(monkeypatch):
    resumes = [
        {"candidate_id": "ok_1", "text": "resume text", "category": "software_engineer", "tier": "mid",
         "ground_truth": {"years_experience": 3.0, "skills": ["Python"],
                           "roles": [{"title": "Engineer", "company": "Acme"}]}},
        {"candidate_id": "boom_1", "text": "resume text", "category": "software_engineer", "tier": "mid",
         "ground_truth": {"years_experience": 3.0, "skills": [], "roles": []}},
    ]

    def fake_extract(text, candidate_id):
        if candidate_id == "boom_1":
            raise _fake_api_error()
        return _candidate()

    monkeypatch.setattr(run_eval_module, "extract_candidate", fake_extract)
    results, candidates_by_id = run_eval_module.run_extraction_eval(resumes)

    assert results["n"] == 2
    assert results["n_failed"] == 1  # boom_1 counted as a real failure, not a crash
    assert "ok_1" in candidates_by_id
    assert "boom_1" not in candidates_by_id


def test_run_ranking_eval_survives_transient_api_error(monkeypatch):
    resumes = [
        {"candidate_id": "ok_1", "category": "software_engineer", "tier": "mid", "text": "r1"},
        {"candidate_id": "boom_1", "category": "software_engineer", "tier": "mid", "text": "r2"},
    ]
    jds = [{"job_id": "job_1", "category": "software_engineer", "text": "jd text"}]
    ranking_truth = {"rankings": [{"job_id": "job_1", "tier_rankings": {"mid": ["ok_1", "boom_1"]}}]}
    candidates_by_id = {"ok_1": _candidate(), "boom_1": _candidate()}

    def fake_score(candidate, jd_text, candidate_id, job_id):
        if candidate_id == "boom_1":
            raise _fake_api_error()
        from app.scoring.rubric import CriterionScore, RubricScore
        from app.scoring.score import CandidateScore
        c = CriterionScore(score=4, rationale="ok")
        rubric = RubricScore(skills_match=c, experience_fit=c, education_fit=c, role_relevance=c)
        return CandidateScore(candidate_id=candidate_id, job_id=job_id, rubric=rubric, weighted_total=4.0)

    monkeypatch.setattr(run_eval_module, "score_candidate", fake_score)
    monkeypatch.setattr(run_eval_module, "tfidf_baseline_rank", lambda texts, jd: list(texts.keys()))
    monkeypatch.setattr(run_eval_module, "rank_via_bare_llm", lambda *a, **k: [])

    # Should not raise — the whole point of the fix.
    results = run_eval_module.run_ranking_eval(resumes, jds, ranking_truth, candidates_by_id)
    tier = results["per_job"][0]["per_tier"]["mid"]
    assert tier["n_scored"] == 1  # only ok_1 made it through
    assert tier["n_truth"] == 2


def test_run_bias_eval_survives_transient_api_error(monkeypatch):
    reps = [({"job_id": "job_1", "text": "jd"}, {"candidate_id": "cand_1", "text": "resume"})]

    monkeypatch.setattr(bias_eval_module, "extract_candidate", lambda text, cid: _candidate())
    monkeypatch.setattr(bias_eval_module, "load_variants", lambda name: [{"pair_id": "p1", "variant_a": "A", "variant_b": "B"}])

    def boom(*args, **kwargs):
        raise _fake_api_error()

    monkeypatch.setattr(bias_eval_module, "_run_arm", boom)
    result = bias_eval_module.run_bias_eval(reps)
    assert result["reps"] == []  # the one rep failed cleanly, no crash, no partial garbage
