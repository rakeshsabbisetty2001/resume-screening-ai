"""Quality eval: extraction accuracy + ranking quality vs. two baselines.

Cost gate: run with no flags (or --estimate) to print the call/cost estimate
without spending anything. Only `--run` actually calls the API — refuses to
run without ANTHROPIC_API_KEY set, per the plan's pre-flight budget check.

Pinned models: extraction_model / scoring_model from app.config.settings —
recorded into results.md so every number here stays attributable to a
specific model version (see app/config.py's comment on why these carry no
date suffix).

Rough cost: ~$0.016/call at claude-sonnet-5 intro pricing ($3/$15 per 1M
in/out tokens, ~1200 input + ~800 output tokens/call — extraction and
scoring prompts are short, effort="low"). This project's total estimate:
run_eval.py 121 calls (~$1.90), bias_eval.py 215 calls (~$3.40) — the call
counts are exact (both scripts compute and print them from the real
manifests before spending anything); the $ figures are the approximate part.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean

from app.config import settings
from app.extraction.extract import ExtractionError, extract_candidate
from app.scoring.score import ScoringError, rank_candidates, score_candidate
from eval.baseline_llm import BaselineLLMError, rank_via_bare_llm
from eval.metrics import (extraction_accuracy, pairwise_agreement, ranking_kendall_tau_b,
                           tfidf_baseline_rank, top_k_precision)

ROOT = Path(__file__).resolve().parent.parent
RESUMES_DIR = ROOT / "data" / "synthetic" / "resumes"
JDS_DIR = ROOT / "data" / "synthetic" / "jds"
RESULTS_JSON = Path(__file__).resolve().parent / "results.json"
RESULTS_MD = Path(__file__).resolve().parent / "results.md"

# Sonnet-drafts/Opus-reviews estimate, per candidate — extraction (1 call)
# + scoring (1 call, name-blind) + the rubric-free baseline is one call
# PER JOB not per candidate. Actual n's come from the real manifests below;
# this constant is just the shape of the estimate.
N_EXTRACTION_RERUNS = 1  # not bias-sensitive; n>=3 reruns are Phase 5's job


def load_resumes() -> list[dict]:
    manifest = json.loads((RESUMES_DIR / "manifest.json").read_text())
    for entry in manifest:
        entry["text"] = (RESUMES_DIR / f"{entry['candidate_id']}.txt").read_text(encoding="utf-8")
    return manifest


def load_jds() -> list[dict]:
    manifest = json.loads((JDS_DIR / "manifest.json").read_text())
    for entry in manifest:
        entry["text"] = (JDS_DIR / f"{entry['job_id']}.txt").read_text(encoding="utf-8")
    return manifest


def load_ranking_ground_truth() -> dict:
    path = Path(__file__).resolve().parent / "ranking_dataset.json"
    return json.loads(path.read_text())


def estimate_calls(resumes: list[dict], jds: list[dict]) -> dict:
    extraction_calls = len(resumes) * N_EXTRACTION_RERUNS
    scoring_calls = sum(
        sum(1 for r in resumes if r["category"] == jd["category"]) for jd in jds
    )
    baseline_llm_calls = len(jds)  # one rubric-free ranking call per JD
    total = extraction_calls + scoring_calls + baseline_llm_calls
    return {"extraction_calls": extraction_calls, "scoring_calls": scoring_calls,
            "baseline_llm_calls": baseline_llm_calls, "total_calls": total}


def run_extraction_eval(resumes: list[dict]) -> tuple[dict, dict]:
    # Returns (results, candidates_by_id) — the caller (run_ranking_eval)
    # reuses these Candidate objects rather than re-extracting, since each
    # resume belongs to exactly one category and therefore feeds every JD
    # in that category. Re-extracting per JD was the estimate_calls bug:
    # it silently tripled/doubled the real call count beyond what the
    # printed estimate promised.
    per_candidate = []
    candidates_by_id = {}
    start = time.monotonic()
    for entry in resumes:
        try:
            candidate = extract_candidate(entry["text"], entry["candidate_id"])
            candidates_by_id[entry["candidate_id"]] = candidate
            acc = extraction_accuracy(candidate, entry["ground_truth"])
            per_candidate.append({"candidate_id": entry["candidate_id"], "error": None, **acc})
        except ExtractionError as e:
            per_candidate.append({"candidate_id": entry["candidate_id"], "error": str(e)})
    elapsed = time.monotonic() - start

    ok = [r for r in per_candidate if r.get("error") is None]
    results = {
        "n": len(resumes),
        "n_failed": len(resumes) - len(ok),
        "mean_skill_f1": mean(r["skill_f1"] for r in ok) if ok else None,
        "years_within_tolerance_rate": mean(r["years_within_tolerance"] for r in ok) if ok else None,
        "role_count_match_rate": mean(r["role_count_match"] for r in ok) if ok else None,
        "per_candidate": per_candidate,
        "wall_clock_seconds": round(elapsed, 1),
        "resumes_per_hour": round(len(resumes) / (elapsed / 3600), 1) if elapsed > 0 else None,
    }
    return results, candidates_by_id


def run_ranking_eval(resumes: list[dict], jds: list[dict], ranking_truth: dict,
                      candidates_by_id: dict) -> dict:
    truth_by_job = {r["job_id"]: r for r in ranking_truth["rankings"]}
    per_job = []

    for jd in jds:
        job_candidates = [r for r in resumes if r["category"] == jd["category"]]
        if not job_candidates:
            continue
        truth = truth_by_job.get(jd["job_id"])
        if truth is None:
            continue

        # AI scorer (name-blind by default) — one call per candidate,
        # reusing the already-extracted Candidate (no re-extraction).
        scores = []
        for r in job_candidates:
            candidate_obj = candidates_by_id.get(r["candidate_id"])
            if candidate_obj is None:
                continue  # extraction already failed for this candidate
            try:
                scores.append(score_candidate(candidate_obj, jd["text"], r["candidate_id"], jd["job_id"]))
            except ScoringError:
                continue
        ranked_ids = [s.candidate_id for s in rank_candidates(scores)]
        score_by_id = {s.candidate_id: s.weighted_total for s in scores}

        # Baseline 1: deterministic TF-IDF cosine, no LLM call.
        tfidf_ranked = tfidf_baseline_rank(
            {r["candidate_id"]: r["text"] for r in job_candidates}, jd["text"])

        # Baseline 2: rubric-free bare LLM ranking, one call for the whole
        # JD (not per candidate) — isolates whether the rubric earns its cost.
        try:
            bare_llm_ranked = rank_via_bare_llm(
                jd["job_id"], jd["text"], {r["candidate_id"]: r["text"] for r in job_candidates})
        except BaselineLLMError:
            bare_llm_ranked = []

        per_tier = {}
        for tier, truth_ids in truth["tier_rankings"].items():
            if not truth_ids:
                continue
            tier_pred = [c for c in ranked_ids if c in truth_ids]
            tier_tfidf = [c for c in tfidf_ranked if c in truth_ids]
            tier_bare_llm = [c for c in bare_llm_ranked if c in truth_ids]
            per_tier[tier] = {
                "n_truth": len(truth_ids), "n_scored": len(tier_pred),  # visible when a tier partly failed
                "rubric_top_k_precision": top_k_precision(tier_pred, truth_ids, k=max(1, len(truth_ids) // 2)),
                "rubric_pairwise_agreement": pairwise_agreement(tier_pred, truth_ids),
                "rubric_kendall_tau_b": ranking_kendall_tau_b(tier_pred, truth_ids, score_by_id),
                "tfidf_pairwise_agreement": pairwise_agreement(tier_tfidf, truth_ids),
                "bare_llm_pairwise_agreement": pairwise_agreement(tier_bare_llm, truth_ids),
            }

        per_job.append({"job_id": jd["job_id"], "category": jd["category"],
                         "n_candidates": len(job_candidates), "per_tier": per_tier})

    return {"per_job": per_job}


def write_results_md(extraction: dict, ranking: dict, call_estimate: dict) -> None:
    lines = [
        "# Project 3 Eval Results — Quality",
        "",
        f"Models: extraction=`{settings.extraction_model}`, scoring=`{settings.scoring_model}` "
        "(pinned in app/config.py — see that file's comment on why these IDs carry no date suffix).",
        "",
        "## Extraction accuracy",
        f"- n = {extraction['n']} resumes, {extraction['n_failed']} failed",
        f"- mean skill-set F1: {extraction['mean_skill_f1']}",
        f"- years-within-1yr-tolerance rate: {extraction['years_within_tolerance_rate']}",
        f"- role-count-match rate: {extraction['role_count_match_rate']}",
        f"- throughput: {extraction['resumes_per_hour']} resumes/hour "
        f"({extraction['wall_clock_seconds']}s wall-clock for {extraction['n']} resumes)",
        "",
        "## Ranking quality (per job, stratified within tier — see methodology notes)",
    ]
    def fmt(v):
        return f"{v:.2f}" if isinstance(v, float) else "n/a"

    for job in ranking["per_job"]:
        lines.append(f"### {job['job_id']} ({job['category']}, n={job['n_candidates']})")
        for tier, m in job["per_tier"].items():
            lines.append(
                f"- {tier} (scored {m['n_scored']}/{m['n_truth']}): "
                f"top-k precision={fmt(m['rubric_top_k_precision'])}, "
                f"rubric pairwise-agreement={fmt(m['rubric_pairwise_agreement'])}, "
                f"tau-b={fmt(m['rubric_kendall_tau_b'])}, "
                f"TF-IDF baseline={fmt(m['tfidf_pairwise_agreement'])}, "
                f"bare-LLM baseline={fmt(m['bare_llm_pairwise_agreement'])}")
    lines += [
        "",
        "## Methodology notes",
        "- Ranking ground truth: one non-recruiter labeler (see eval/ranking_dataset.json's "
        "own methodology field), reading resume text blind to the corpus's generation "
        "parameters, ranked within tier (not across tiers) per JD to avoid the tier "
        "confound design decision 1 flagged.",
        "- Headline ranking metric is per-JD, per-tier pairwise agreement, not a single "
        "blended number — same convention as Projects 1 and 2 (don't blend distinct axes).",
        "- Kendall tau-b is reported as a secondary/supporting number only — at this n "
        "(a handful of candidates per tier per JD) it has no usable confidence interval.",
        "- Extraction eval uses n=1 per resume (not bias-sensitive); the bias eval "
        "(eval/bias_eval.py, eval/bias_results.md) uses n>=3 reruns where sampling "
        "variance actually matters.",
        f"- Call estimate before this run: {call_estimate}",
    ]
    RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true",
                         help="Actually call the API. Without this flag, only prints the cost estimate.")
    args = parser.parse_args()

    resumes = load_resumes()
    jds = load_jds()
    estimate = estimate_calls(resumes, jds)

    print(f"Call estimate: {estimate}")
    if not args.run:
        print("Dry run (default) — pass --run to actually call the API. "
              "No requests were made.")
        return

    if not settings.anthropic_api_key:
        print("ANTHROPIC_API_KEY is not set — refusing to run with --run. "
              "Add it to .env first.", file=sys.stderr)
        sys.exit(1)

    ranking_truth = load_ranking_ground_truth()
    extraction_results, candidates_by_id = run_extraction_eval(resumes)
    ranking_results = run_ranking_eval(resumes, jds, ranking_truth, candidates_by_id)

    RESULTS_JSON.write_text(json.dumps(
        {"extraction": extraction_results, "ranking": ranking_results, "call_estimate": estimate},
        indent=2), encoding="utf-8")
    write_results_md(extraction_results, ranking_results, estimate)
    print(f"Wrote {RESULTS_JSON} and {RESULTS_MD}")


if __name__ == "__main__":
    main()
