from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.extraction.extract import ExtractionError
from app.extraction.schema import Candidate, Education, Role
from app.scoring.score import CandidateScore, ScoringError
from app.scoring.rubric import CriterionScore, RubricScore

client = TestClient(main_module.app)
# raise_server_exceptions=False: the whole point of the catch-all handler
# is to turn an unhandled exception into a clean 500 instead of a raw
# traceback — the default TestClient re-raises server exceptions, which
# would make that handler untestable.
client_no_raise = TestClient(main_module.app, raise_server_exceptions=False)

RESUME_TEXT = "Jordan Rivera\n\n5 years experience.\n\nSkills: Python\n\n" + "x" * 20
JD_TEXT = "Software engineer role requiring Python. " + "y" * 20


def _candidate():
    return Candidate(
        name="Jordan Rivera", years_experience=5.0, skills=["Python"],
        education=[Education(degree="B.S. Computer Science", institution="State University")],
        roles=[Role(title="Engineer", company="Acme", start_date="2019-01",
                     end_date="2024-01", bullets=[])],
    )


def _score():
    c = CriterionScore(score=4, rationale="ok")
    rubric = RubricScore(skills_match=c, experience_fit=c, education_fit=c, role_relevance=c)
    return CandidateScore(candidate_id="x", job_id="y", rubric=rubric, weighted_total=4.0)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_root_redirects_to_docs():
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code in (307, 302)
    assert resp.headers["location"] == "/docs"


def test_extract_success(monkeypatch):
    monkeypatch.setattr(main_module, "extract_candidate", lambda text, rid: _candidate())
    resp = client.post("/extract", json={"resume_text": RESUME_TEXT})
    assert resp.status_code == 200
    assert resp.json()["name"] == "Jordan Rivera"


def test_extract_failure_returns_422_without_leaking_text(monkeypatch):
    def fail(text, rid):
        raise ExtractionError(f"extraction failed for {rid}")
    monkeypatch.setattr(main_module, "extract_candidate", fail)
    resp = client.post("/extract", json={"resume_text": RESUME_TEXT})
    assert resp.status_code == 422
    assert RESUME_TEXT not in resp.text


def test_extract_body_too_short_returns_422_without_echoing_input():
    # Pydantic min_length violation — FastAPI's default handler would echo
    # the (short but still real) submitted text back in `input`; the
    # override must strip it. Uses a distinctive value, not "short" — that
    # word is itself a substring of the error type ("string_too_short"),
    # which would false-positive a naive containment check.
    resp = client.post("/extract", json={"resume_text": "zzqxwv123"})
    assert resp.status_code == 422
    assert "zzqxwv123" not in resp.text
    assert "input" not in str(resp.json()["detail"])


def test_score_success(monkeypatch):
    monkeypatch.setattr(main_module, "extract_candidate", lambda text, rid: _candidate())
    monkeypatch.setattr(main_module, "score_candidate",
                         lambda candidate, jd, cid, jid: _score())
    resp = client.post("/score", json={"resume_text": RESUME_TEXT, "job_description": JD_TEXT})
    assert resp.status_code == 200
    assert resp.json()["weighted_total"] == 4.0


def test_score_extraction_failure_returns_422(monkeypatch):
    def fail(text, rid):
        raise ExtractionError(f"extraction failed for {rid}")
    monkeypatch.setattr(main_module, "extract_candidate", fail)
    resp = client.post("/score", json={"resume_text": RESUME_TEXT, "job_description": JD_TEXT})
    assert resp.status_code == 422


def test_score_scoring_failure_returns_422(monkeypatch):
    monkeypatch.setattr(main_module, "extract_candidate", lambda text, rid: _candidate())

    def fail(candidate, jd, cid, jid):
        raise ScoringError(f"scoring failed for {cid}")
    monkeypatch.setattr(main_module, "score_candidate", fail)
    resp = client.post("/score", json={"resume_text": RESUME_TEXT, "job_description": JD_TEXT})
    assert resp.status_code == 422


def test_unhandled_exception_returns_500_not_traceback(monkeypatch):
    def boom(text, rid):
        raise RuntimeError("boom, something internal broke")
    monkeypatch.setattr(main_module, "extract_candidate", boom)
    resp = client_no_raise.post("/extract", json={"resume_text": RESUME_TEXT})
    assert resp.status_code == 500
    assert "boom" not in resp.text
    assert "Traceback" not in resp.text
