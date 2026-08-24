"""Candidate vs. job-description scoring, via Claude structured output.

Name-blind by construction (plan design decision 3): `_serialize_candidate`
never sends the candidate's name field, AND scrubs literal occurrences of
the name (and its individual name-parts) out of the rendered free text —
dropping the field alone isn't enough, since Phase 1's generated resumes
(and real ones) can restate the name inside a role bullet or summary line;
without the scrub, that text would flow into the model input unblinded.
`include_name=True` exists only for the bias eval's name-visible arm
(Phase 5), which needs to measure the *effect* of the name being visible.

One structured call scores every rubric criterion together (not one call
per criterion) — cheaper, lets the model weigh criteria against each other,
and keeps eval-rerun cost affordable for the bias eval's n>=3 reruns.

Error handling mirrors app/extraction/extract.py: `client.messages.parse()`
raises a bare `pydantic.ValidationError` (which can quote raw model output)
from inside the call itself, not just from reading `.parsed_output`
afterward — caught and re-raised clean, `from None`, same PII rationale.
"""
import time

import anthropic
import pydantic

from app.config import settings
from app.extraction.schema import Candidate
from app.logging_config import log_scoring
from app.scoring.rubric import RubricScore

client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

SCORE_PROMPT = (
    "Score this candidate against the job description below using the rubric "
    "criteria (skills_match, experience_fit, education_fit, role_relevance). "
    "Each criterion is 1-5 (5 = excellent fit) with a one-sentence rationale. "
    "Score only what's in the candidate summary — do not assume unstated skills "
    "or experience.\n\n"
    "Job description:\n{job_description}\n\n"
    "Candidate:\n{candidate_summary}"
)


class ScoringError(Exception):
    """Same PII contract as app.extraction.extract.ExtractionError — never
    carries resume/JD text or raw model output in its message, only ids."""


class CandidateScore(pydantic.BaseModel):
    candidate_id: str
    job_id: str
    rubric: RubricScore
    weighted_total: float


def _serialize_candidate(candidate: Candidate, include_name: bool) -> str:
    lines = []
    if include_name:
        lines.append(f"Name: {candidate.name}")
    lines.append(f"Years of experience: {candidate.years_experience}")
    lines.append("Skills: " + ", ".join(candidate.skills))
    if candidate.education:
        lines.append("Education: " + "; ".join(
            f"{e.degree}, {e.institution}" for e in candidate.education))
    lines.append("Experience:")
    for role in candidate.roles:
        lines.append(f"- {role.title}, {role.company} ({role.start_date} - {role.end_date})")
        for bullet in role.bullets:
            lines.append(f"    * {bullet}")
    text = "\n".join(lines)

    if not include_name:
        # Dropping the `name` field isn't enough on its own — the name can
        # still appear inside free text (a role bullet, a summary line).
        # Scrub the full name and each individual token (longest first, so
        # "Jordan Rivera" doesn't leave a dangling "Rivera").
        name_tokens = sorted({candidate.name, *candidate.name.split()}, key=len, reverse=True)
        for token in name_tokens:
            if token:
                text = text.replace(token, "[name]")
    return text


def score_candidate(candidate: Candidate, job_description: str, candidate_id: str,
                     job_id: str, include_name: bool = False) -> CandidateScore:
    candidate_summary = _serialize_candidate(candidate, include_name)
    prompt = SCORE_PROMPT.format(job_description=job_description, candidate_summary=candidate_summary)

    start = time.monotonic()
    try:
        response = client.messages.parse(
            model=settings.scoring_model,
            max_tokens=16000,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": prompt}],
            output_format=RubricScore,
        )
    except pydantic.ValidationError:
        raise ScoringError(f"malformed scoring output for candidate {candidate_id} / job {job_id}") from None
    latency_ms = (time.monotonic() - start) * 1000

    if response.stop_reason == "refusal":
        raise ScoringError(f"scoring refused for candidate {candidate_id} / job {job_id}")
    if response.parsed_output is None:
        raise ScoringError(f"no parsed output for candidate {candidate_id} / job {job_id} "
                            f"(stop_reason={response.stop_reason})")

    rubric = response.parsed_output
    weighted_total = rubric.weighted_total()

    log_scoring(candidate_id, job_id, settings.scoring_model,
                response.usage.output_tokens, latency_ms, weighted_total)
    return CandidateScore(candidate_id=candidate_id, job_id=job_id,
                           rubric=rubric, weighted_total=weighted_total)


def rank_candidates(scores: list[CandidateScore]) -> list[CandidateScore]:
    # Ties: stable sort keeps input order for equal weighted_total (i.e.
    # whatever order the caller passed candidates in) — callers that care
    # about a specific tie-break (e.g. eval/run_eval.py's ranking metric)
    # should sort their input accordingly before calling this.
    return sorted(scores, key=lambda s: s.weighted_total, reverse=True)
