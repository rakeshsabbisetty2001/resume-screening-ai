"""Pure, offline-testable metric functions for eval/run_eval.py.

Kept separate from run_eval.py's orchestration (live API calls, dataset
loading) so the actual scoring math has unit tests that don't require a
network call or an API key.
"""
from scipy.stats import kendalltau
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.extraction.schema import Candidate


def skill_set_f1(predicted: list[str], truth: list[str]) -> float:
    pred_set = {s.lower() for s in predicted}
    truth_set = {s.lower() for s in truth}
    if not pred_set and not truth_set:
        return 1.0
    if not pred_set or not truth_set:
        return 0.0
    tp = len(pred_set & truth_set)
    precision = tp / len(pred_set)
    recall = tp / len(truth_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def extraction_accuracy(candidate: Candidate, ground_truth: dict) -> dict:
    years_ok = abs(candidate.years_experience - ground_truth["years_experience"]) <= 1.0
    role_count_ok = len(candidate.roles) == len(ground_truth["roles"])
    return {
        "skill_f1": skill_set_f1(candidate.skills, ground_truth["skills"]),
        "years_within_tolerance": years_ok,
        "role_count_match": role_count_ok,
    }


def top_k_precision(predicted_ids: list[str], truth_ids: list[str], k: int) -> float:
    k = min(k, len(truth_ids))
    if k == 0:
        return 1.0
    pred_top_k = set(predicted_ids[:k])
    truth_top_k = set(truth_ids[:k])
    return len(pred_top_k & truth_top_k) / k


def pairwise_agreement(predicted_ids: list[str], truth_ids: list[str]) -> float:
    # Fraction of candidate pairs (both present in both lists) whose
    # relative order agrees between predicted and truth ranking.
    common = [c for c in truth_ids if c in predicted_ids]
    if len(common) < 2:
        return 1.0
    truth_rank = {c: i for i, c in enumerate(truth_ids) if c in common}
    pred_rank = {c: i for i, c in enumerate(predicted_ids) if c in common}
    agree, total = 0, 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            a, b = common[i], common[j]
            total += 1
            truth_order = truth_rank[a] < truth_rank[b]
            pred_order = pred_rank[a] < pred_rank[b]
            if truth_order == pred_order:
                agree += 1
    return agree / total if total else 1.0


def ranking_kendall_tau_b(predicted_ids: list[str], truth_ids: list[str]) -> float | None:
    common = [c for c in truth_ids if c in predicted_ids]
    if len(common) < 2:
        return None  # not enough overlap to compute a correlation
    truth_rank = [i for i, _ in enumerate(common)]
    pred_rank = [predicted_ids.index(c) for c in common]
    tau, _ = kendalltau(truth_rank, pred_rank, variant="b")
    return tau


def tfidf_baseline_rank(candidate_texts: dict[str, str], jd_text: str) -> list[str]:
    # Deterministic, non-LLM baseline: "would keyword overlap alone have
    # produced a similar ranking?" No name-blindness concern — TF-IDF has
    # no notion of identity, only term overlap.
    ids = list(candidate_texts.keys())
    corpus = [candidate_texts[i] for i in ids] + [jd_text]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(corpus)
    jd_vec = matrix[-1]
    sims = cosine_similarity(matrix[:-1], jd_vec).flatten()
    return [ids[i] for i in sims.argsort()[::-1]]
