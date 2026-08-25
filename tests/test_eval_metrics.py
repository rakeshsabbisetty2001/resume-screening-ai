from app.extraction.schema import Candidate, Education, Role
from eval.metrics import (extraction_accuracy, pairwise_agreement, ranking_kendall_tau_b,
                           skill_set_f1, tfidf_baseline_rank, top_k_precision)


def test_skill_set_f1_perfect_match():
    assert skill_set_f1(["Python", "AWS"], ["python", "aws"]) == 1.0  # case-insensitive


def test_skill_set_f1_no_overlap():
    assert skill_set_f1(["Python"], ["Java"]) == 0.0


def test_skill_set_f1_partial():
    f1 = skill_set_f1(["Python", "AWS", "Go"], ["Python", "AWS"])
    assert 0.0 < f1 < 1.0


def test_skill_set_f1_both_empty():
    assert skill_set_f1([], []) == 1.0


def test_extraction_accuracy_shape():
    candidate = Candidate(
        name="X", years_experience=5.0, skills=["Python", "AWS"],
        education=[Education(degree="B.S. Computer Science", institution="State University")],
        roles=[Role(title="Engineer", company="Acme", start_date="2019-01",
                     end_date="2024-01", bullets=[])],
    )
    ground_truth = {"years_experience": 5.0, "skills": ["Python", "AWS"],
                     "roles": [{"title": "Engineer", "company": "Acme",
                                "start_date": "2019-01", "end_date": "2024-01"}]}
    result = extraction_accuracy(candidate, ground_truth)
    assert result["skill_f1"] == 1.0
    assert result["years_within_tolerance"] is True
    assert result["role_count_match"] is True


def test_extraction_accuracy_years_outside_tolerance():
    candidate = Candidate(name="X", years_experience=1.0, skills=[], education=[], roles=[])
    ground_truth = {"years_experience": 5.0, "skills": [], "roles": []}
    result = extraction_accuracy(candidate, ground_truth)
    assert result["years_within_tolerance"] is False


def test_top_k_precision_perfect():
    assert top_k_precision(["a", "b", "c"], ["a", "b", "c"], k=2) == 1.0


def test_top_k_precision_no_overlap():
    assert top_k_precision(["x", "y"], ["a", "b"], k=2) == 0.0


def test_top_k_precision_empty_truth():
    assert top_k_precision(["a"], [], k=3) == 1.0  # nothing to be wrong about


def test_pairwise_agreement_identical_order():
    assert pairwise_agreement(["a", "b", "c"], ["a", "b", "c"]) == 1.0


def test_pairwise_agreement_fully_reversed():
    assert pairwise_agreement(["c", "b", "a"], ["a", "b", "c"]) == 0.0


def test_pairwise_agreement_partial():
    agreement = pairwise_agreement(["a", "c", "b"], ["a", "b", "c"])
    assert 0.0 < agreement < 1.0


def test_pairwise_agreement_single_candidate_is_vacuously_perfect():
    assert pairwise_agreement(["a"], ["a"]) == 1.0


def test_ranking_kendall_tau_b_perfect_agreement():
    tau = ranking_kendall_tau_b(["a", "b", "c"], ["a", "b", "c"])
    assert abs(tau - 1.0) < 1e-9


def test_ranking_kendall_tau_b_full_disagreement():
    tau = ranking_kendall_tau_b(["c", "b", "a"], ["a", "b", "c"])
    assert abs(tau - (-1.0)) < 1e-9


def test_ranking_kendall_tau_b_too_few_common_returns_none():
    assert ranking_kendall_tau_b(["a"], ["a"]) is None


def test_tfidf_baseline_rank_prefers_more_overlap():
    jd = "Python backend engineer with AWS and Docker experience"
    candidates = {
        "strong": "Python AWS Docker backend engineer with 5 years experience",
        "weak": "Sales representative with CRM and negotiation skills",
    }
    ranked = tfidf_baseline_rank(candidates, jd)
    assert ranked[0] == "strong"
