import os
from datetime import date

import pytest

from app.extraction.extract import ExtractionError, _validate_invariants, extract_candidate
from app.extraction.schema import Candidate, Education, Role

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
    # 5 years stated, role only spans 1 year — well outside the 25%/12mo tolerance
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


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="live API smoke test")
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
