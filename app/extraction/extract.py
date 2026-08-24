"""Resume -> structured Candidate, via Claude structured output.

Per the plan (design decision 2): no generic JSON-repair layer. Current
structured-output mode already constrains the schema server-side, so
malformed JSON isn't the failure mode to defend against. What still needs
handling: `stop_reason` (a 200 response can still be a refusal or a
truncated generation — must be checked before trusting `parsed_output`),
and semantic invariants Pydantic's type system can't express (no future
dates, start before end, stated years roughly matching summed role spans).
"""
import time
from datetime import date

import anthropic

from app.config import settings
from app.extraction.schema import Candidate
from app.logging_config import log_extraction

client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

EXTRACTION_PROMPT = (
    "Extract structured candidate information from the resume text below. "
    "years_experience should be the total professional experience in years "
    "(a number, may have a decimal). List one entry per role/title held, "
    "most recent first. Use the literal string 'present' for end_date on a "
    "role the candidate currently holds; otherwise use 'YYYY-MM'."
)


class ExtractionError(Exception):
    """Raised on anything the extractor can't recover from. Never carries
    resume text in its message — only a candidate id — per the PII
    guardrail in app/logging_config.py: an exception message is exactly
    the kind of string that could end up in a log or an error response."""


def extract_candidate(resume_text: str, candidate_id: str) -> Candidate:
    start = time.monotonic()
    response = client.messages.parse(
        model=settings.extraction_model,
        max_tokens=4096,
        messages=[{"role": "user", "content": f"{EXTRACTION_PROMPT}\n\n{resume_text}"}],
        output_format=Candidate,
    )
    latency_ms = (time.monotonic() - start) * 1000

    if response.stop_reason == "refusal":
        raise ExtractionError(f"extraction refused for candidate {candidate_id}")
    if response.stop_reason == "max_tokens":
        raise ExtractionError(f"extraction truncated (max_tokens) for candidate {candidate_id}")

    candidate = response.parsed_output
    _validate_invariants(candidate, candidate_id)

    log_extraction(candidate_id, settings.extraction_model,
                    response.usage.output_tokens, latency_ms, response.stop_reason)
    return candidate


def _parse_ym(s: str) -> date | None:
    try:
        y, m = s.split("-")
        return date(int(y), int(m), 1)
    except (ValueError, AttributeError):
        return None  # malformed date string; schema only guarantees `str`, not the format


def _months_between(role) -> int | None:
    start = _parse_ym(role.start_date)
    end = date.today() if role.end_date.strip().lower() == "present" else _parse_ym(role.end_date)
    if start is None or end is None:
        return None
    return (end.year - start.year) * 12 + (end.month - start.month)


def _validate_invariants(candidate: Candidate, candidate_id: str) -> None:
    today = date.today()
    role_months = []
    for role in candidate.roles:
        start = _parse_ym(role.start_date)
        end = today if role.end_date.strip().lower() == "present" else _parse_ym(role.end_date)
        if start is None or end is None:
            continue  # can't validate a date this extractor couldn't parse either
        if (end.year, end.month) > (today.year, today.month):
            raise ExtractionError(f"role end date in the future for candidate {candidate_id}")
        if start > end:
            raise ExtractionError(f"role start date after end date for candidate {candidate_id}")
        months = _months_between(role)
        if months is not None:
            role_months.append(months)

    if role_months:
        total_role_months = sum(role_months)
        stated_months = candidate.years_experience * 12
        # Generous tolerance (25%, min 12 months): this catches a wildly
        # wrong extraction, not sub-year rounding on real resumes with
        # overlapping or loosely-dated roles.
        if abs(stated_months - total_role_months) > max(12, total_role_months * 0.25):
            raise ExtractionError(
                f"stated years_experience inconsistent with role dates for candidate {candidate_id}")
