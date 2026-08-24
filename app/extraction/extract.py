"""Resume -> structured Candidate, via Claude structured output.

Per the plan (design decision 2): no generic JSON-repair layer. Current
structured-output mode already constrains the schema server-side, so
malformed JSON isn't the failure mode to defend against. What still needs
handling: `stop_reason` (a clean refusal can return with no text block at
all, so `parsed_output` legitimately comes back `None`), plus semantic
invariants Pydantic's type system can't express (no future dates, date
ordering, years-experience consistency with role dates).

`client.messages.parse()` itself parses the model's text into `Candidate`
*before returning* (see `anthropic.lib._parse._response`) — a truncated or
malformed generation raises `pydantic.ValidationError` out of the `parse()`
call itself, not out of reading `.parsed_output` afterward. That exception's
message embeds the raw (invalid) JSON the model wrote — which can contain
resume text (a candidate's name, in practice) — so it must never propagate
un-wrapped past this module, per the PII guardrail in app/logging_config.py.
"""
import time
from datetime import date

import anthropic
import pydantic

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
    resume text or the model's raw output in its message — only a
    candidate id. Any exception caught inside this module that might embed
    model output (e.g. a `pydantic.ValidationError` on truncated/malformed
    JSON, which quotes the raw text) is re-raised with `from None` so the
    original message can't reach a log or error response through
    `__cause__`/`__context__` either."""


def extract_candidate(resume_text: str, candidate_id: str) -> Candidate:
    start = time.monotonic()
    try:
        response = client.messages.parse(
            model=settings.extraction_model,
            max_tokens=16000,  # extraction_model runs adaptive thinking by default on
            output_config={"effort": "low"},  # claude-sonnet-5; thinking tokens count
            messages=[{"role": "user", "content": f"{EXTRACTION_PROMPT}\n\n{resume_text}"}],
            output_format=Candidate,
        )
    except pydantic.ValidationError:
        # Raised by parse() itself on truncated/malformed model output, and
        # its message quotes the raw (invalid) JSON — which can contain
        # resume text. Never let that string escape this module.
        raise ExtractionError(f"malformed extraction output for candidate {candidate_id}") from None
    latency_ms = (time.monotonic() - start) * 1000

    # Belt-and-suspenders: a clean refusal returns stop_reason="refusal"
    # with no text block, so parsed_output is legitimately None rather than
    # raising — check both explicitly rather than trusting the type hint.
    if response.stop_reason == "refusal":
        raise ExtractionError(f"extraction refused for candidate {candidate_id}")
    if response.parsed_output is None:
        raise ExtractionError(f"no parsed output for candidate {candidate_id} "
                               f"(stop_reason={response.stop_reason})")

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


def _union_months(intervals: list[tuple[int, int]]) -> int:
    # Sum of *covered* months across possibly-overlapping role intervals
    # (each a (start_month_index, end_month_index) pair) — a plain sum
    # double-counts overlapping roles, which is exactly the case a
    # years-experience consistency check needs to get right.
    if not intervals:
        return 0
    merged = [sorted(intervals)[0]]
    for s, e in sorted(intervals)[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return sum(e - s for s, e in merged)


def _validate_invariants(candidate: Candidate, candidate_id: str) -> None:
    today = date.today()
    intervals = []
    for role in candidate.roles:
        start = _parse_ym(role.start_date)
        end = today if role.end_date.strip().lower() == "present" else _parse_ym(role.end_date)
        if start is None or end is None:
            continue  # can't validate a date this extractor couldn't parse either
        if (end.year, end.month) > (today.year, today.month):
            raise ExtractionError(f"role end date in the future for candidate {candidate_id}")
        if start > end:
            raise ExtractionError(f"role start date after end date for candidate {candidate_id}")
        intervals.append((start.year * 12 + start.month, end.year * 12 + end.month))

    if intervals:
        total_months = _union_months(intervals)
        stated_months = candidate.years_experience * 12
        # ponytail: tolerance is a placeholder, not calibrated against real
        # (non-synthetic) resumes — the corpus's worst gap is <1 month, so
        # this hasn't actually been exercised yet. Revisit once eval/
        # extraction_dataset.json has real disagreement to measure against.
        if abs(stated_months - total_months) > max(3, total_months * 0.10):
            raise ExtractionError(
                f"stated years_experience inconsistent with role dates for candidate {candidate_id}")
