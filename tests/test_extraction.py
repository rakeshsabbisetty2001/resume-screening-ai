import os
from datetime import date
from types import SimpleNamespace

import pydantic
import pytest

from app.extraction import extract as extract_module
from app.extraction.extract import ExtractionError, _validate_invariants, extract_candidate
from app.extraction.schema import Candidate, Education, Role
from tests.conftest import has_real_api_key

THIS_YEAR = date.today().year


def _candidate(**overrides) -> Candidate:
    defaults = dict(
        name="Test Candidate",
        years_experience=5.0,
        skills=["Python"],
        education=[Education(degree="B.S. Computer Science", institution="State University")],
        roles=[Role(title="Software Engineer", company="Acme", start_date="2019-01",
                     end_date="2024-01", bullets=["Did things."])],
    )
    defaults.update(overrides)
    return Candidate(**defaults)


def test_consistent_candidate_passes():
    _validate_invariants(_candidate(), "cand_1")  # no exception


def test_future_end_date_rejected():
    bad = _candidate(roles=[Role(title="X", company="Y", start_date="2024-01",
                                  end_date=f"{THIS_YEAR + 1}-01", bullets=[])])
    with pytest.raises(ExtractionError, match="future"):
        _validate_invariants(bad, "cand_2")


def test_start_after_end_rejected():
    bad = _candidate(roles=[Role(title="X", company="Y", start_date="2024-06",
                                  end_date="2024-01", bullets=[])])
    with pytest.raises(ExtractionError, match="start date after end date"):
        _validate_invariants(bad, "cand_3")


def test_years_inconsistent_with_roles_rejected():
    # 5 years stated, role only spans 1 year — well outside the 10%/3mo tolerance
    bad = _candidate(years_experience=5.0,
                      roles=[Role(title="X", company="Y", start_date="2023-01",
                                  end_date="2024-01", bullets=[])])
    with pytest.raises(ExtractionError, match="inconsistent"):
        _validate_invariants(bad, "cand_4")


def test_present_end_date_uses_today():
    today = date.today()
    start = date(today.year - 1, today.month, 1)  # ~1 year ago, whatever "today" is
    ok = _candidate(years_experience=1.0,
                     roles=[Role(title="X", company="Y", start_date=f"{start.year}-{start.month:02d}",
                                  end_date="present", bullets=[])])
    _validate_invariants(ok, "cand_5")  # no exception — "present" resolves to today


def test_unparseable_date_is_skipped_not_crashed():
    # Schema only guarantees `str`, not a valid YYYY-MM — invariant check
    # should skip what it can't parse, not raise on a malformed string.
    weird = _candidate(roles=[Role(title="X", company="Y", start_date="not-a-date",
                                    end_date="also-not-a-date", bullets=[])])
    _validate_invariants(weird, "cand_6")  # no exception


def test_empty_roles_skips_consistency_check_entirely():
    # Documented as intentional, not accidental: with no parseable roles
    # there's nothing to check years_experience against, so a candidate
    # with an implausible number but zero roles still passes. Negative
    # values are rejected at the schema level instead (Field(ge=0)).
    weird = _candidate(years_experience=40.0, roles=[])
    _validate_invariants(weird, "cand_7")  # no exception — nothing to compare against


def test_overlapping_roles_use_union_not_sum():
    # Two roles overlapping Jan-Jun 2023 should count that overlap once —
    # a plain sum would double-count it and reject a truthful resume.
    ok = _candidate(
        years_experience=1.0,
        roles=[
            Role(title="X", company="A", start_date="2023-01", end_date="2024-01", bullets=[]),
            Role(title="Y", company="B", start_date="2023-01", end_date="2024-01", bullets=[]),
        ],
    )
    _validate_invariants(ok, "cand_8")  # no exception — union is 12mo, not 24mo (a sum would fail this)


def test_negative_years_rejected_at_schema_level():
    with pytest.raises(pydantic.ValidationError):
        _candidate(years_experience=-1.0)


class _FakeResponse:
    def __init__(self, stop_reason, parsed_output, output_tokens=100):
        self.stop_reason = stop_reason
        self.parsed_output = parsed_output
        self.usage = SimpleNamespace(output_tokens=output_tokens)


def test_extract_candidate_wraps_validation_error_without_leaking_text(monkeypatch):
    def fake_parse(**kwargs):
        # Reproduce the SDK's actual failure mode: parse() raises
        # ValidationError internally, and its message can quote raw model
        # output (a candidate's name, in practice).
        raise pydantic.ValidationError.from_exception_data(
            "Candidate", [{"type": "json_invalid", "loc": (),
                            "input": '{"name": "Jordan Rivera", "years_experience": 5',
                            "ctx": {"error": "EOF"}}])

    monkeypatch.setattr(extract_module.client.messages, "parse", fake_parse)
    with pytest.raises(ExtractionError) as exc_info:
        extract_candidate("resume text", "cand_leak_test")
    assert "Jordan Rivera" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None  # `from None` — no chained leak either


def test_extract_candidate_refusal_with_no_parsed_output_raises_cleanly(monkeypatch):
    monkeypatch.setattr(extract_module.client.messages, "parse",
                         lambda **kwargs: _FakeResponse("refusal", None))
    with pytest.raises(ExtractionError, match="refused"):
        extract_candidate("resume text", "cand_refusal_test")


def test_extract_candidate_none_output_without_refusal_still_raises(monkeypatch):
    # Defensive case: parsed_output is None but stop_reason isn't literally
    # "refusal" — must not fall through and hand Phase 3 a None.
    monkeypatch.setattr(extract_module.client.messages, "parse",
                         lambda **kwargs: _FakeResponse("end_turn", None))
    with pytest.raises(ExtractionError, match="no parsed output"):
        extract_candidate("resume text", "cand_none_test")


def test_extract_candidate_happy_path_logs(monkeypatch):
    good = _candidate()
    monkeypatch.setattr(extract_module.client.messages, "parse",
                         lambda **kwargs: _FakeResponse("end_turn", good, output_tokens=250))
    logged = {}
    monkeypatch.setattr(extract_module, "log_extraction",
                         lambda *args, **kwargs: logged.update(tokens=args[2]))
    result = extract_candidate("resume text", "cand_happy_test")
    assert result is good
    assert logged["tokens"] == 250


@pytest.mark.skipif(not has_real_api_key(), reason="live API smoke test")
def test_live_extraction_smoke():
    resume = (
        "Jordan Rivera\n\n5.0 years of experience in software engineer roles.\n\n"
        "Skills: Python, AWS, Docker\n\nExperience:\n"
        "- Software Engineer, Acme Corp (2020-01 - present)\n"
        "    * Built internal tools using Python.\n\n"
        "Education: B.S. Computer Science, State University"
    )
    candidate = extract_candidate(resume, "live_smoke_1")
    assert candidate.years_experience > 0
    assert "Python" in candidate.skills
    assert len(candidate.roles) >= 1
