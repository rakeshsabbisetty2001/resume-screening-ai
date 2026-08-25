# Project 3 Eval Results — Quality

Models: extraction=`claude-sonnet-5`, scoring=`claude-sonnet-5` (pinned in app/config.py — see that file's comment on why these IDs carry no date suffix).

## Extraction accuracy
- n = 40 resumes, 0 failed
- mean skill-set F1: 1.0
- years-within-1yr-tolerance rate: 1
- role-count-match rate: 1
- throughput: 779.4 resumes/hour (184.8s wall-clock for 40 resumes)

## Ranking quality (per job, stratified within tier — see methodology notes)
### swe_backend (software_engineer, n=8)
- junior (scored 3/3): top-k precision=1.00, rubric pairwise-agreement=1.00, tau-b=1.00, TF-IDF baseline=0.67, bare-LLM baseline=0.67
- mid (scored 3/3): top-k precision=0.00, rubric pairwise-agreement=0.33, tau-b=0.00, TF-IDF baseline=0.33, bare-LLM baseline=1.00
- senior (scored 2/2): top-k precision=1.00, rubric pairwise-agreement=1.00, tau-b=1.00, TF-IDF baseline=1.00, bare-LLM baseline=1.00
### swe_fullstack (software_engineer, n=8)
- junior (scored 3/3): top-k precision=0.00, rubric pairwise-agreement=0.33, tau-b=-0.82, TF-IDF baseline=0.67, bare-LLM baseline=0.33
- mid (scored 3/3): top-k precision=1.00, rubric pairwise-agreement=1.00, tau-b=1.00, TF-IDF baseline=1.00, bare-LLM baseline=1.00
- senior (scored 2/2): top-k precision=0.00, rubric pairwise-agreement=0.00, tau-b=-1.00, TF-IDF baseline=0.00, bare-LLM baseline=0.00
### data_analyst_bi (data_analyst, n=8)
- junior (scored 3/3): top-k precision=1.00, rubric pairwise-agreement=0.67, tau-b=0.33, TF-IDF baseline=0.67, bare-LLM baseline=0.33
- mid (scored 3/3): top-k precision=1.00, rubric pairwise-agreement=0.67, tau-b=0.82, TF-IDF baseline=0.67, bare-LLM baseline=1.00
- senior (scored 2/2): top-k precision=1.00, rubric pairwise-agreement=1.00, tau-b=n/a, TF-IDF baseline=1.00, bare-LLM baseline=1.00
### data_analyst_growth (data_analyst, n=8)
- junior (scored 3/3): top-k precision=1.00, rubric pairwise-agreement=1.00, tau-b=1.00, TF-IDF baseline=0.33, bare-LLM baseline=0.67
- mid (scored 3/3): top-k precision=1.00, rubric pairwise-agreement=1.00, tau-b=1.00, TF-IDF baseline=1.00, bare-LLM baseline=1.00
- senior (scored 2/2): top-k precision=1.00, rubric pairwise-agreement=1.00, tau-b=1.00, TF-IDF baseline=1.00, bare-LLM baseline=1.00
### pm_core (product_manager, n=8)
- junior (scored 3/3): top-k precision=1.00, rubric pairwise-agreement=1.00, tau-b=1.00, TF-IDF baseline=0.67, bare-LLM baseline=1.00
- mid (scored 3/3): top-k precision=1.00, rubric pairwise-agreement=1.00, tau-b=1.00, TF-IDF baseline=1.00, bare-LLM baseline=0.67
- senior (scored 2/2): top-k precision=1.00, rubric pairwise-agreement=1.00, tau-b=1.00, TF-IDF baseline=1.00, bare-LLM baseline=1.00
### pm_associate (product_manager, n=8)
- junior (scored 3/3): top-k precision=1.00, rubric pairwise-agreement=1.00, tau-b=0.82, TF-IDF baseline=1.00, bare-LLM baseline=1.00
- mid (scored 3/3): top-k precision=1.00, rubric pairwise-agreement=0.67, tau-b=0.82, TF-IDF baseline=0.33, bare-LLM baseline=1.00
- senior (scored 2/2): top-k precision=1.00, rubric pairwise-agreement=1.00, tau-b=n/a, TF-IDF baseline=0.00, bare-LLM baseline=0.00
### sales_ae (sales, n=8)
- junior (scored 3/3): top-k precision=0.00, rubric pairwise-agreement=0.00, tau-b=-0.82, TF-IDF baseline=1.00, bare-LLM baseline=0.33
- mid (scored 3/3): top-k precision=0.00, rubric pairwise-agreement=0.33, tau-b=0.00, TF-IDF baseline=0.67, bare-LLM baseline=1.00
- senior (scored 2/2): top-k precision=0.00, rubric pairwise-agreement=0.00, tau-b=n/a, TF-IDF baseline=1.00, bare-LLM baseline=1.00
### sales_bdr (sales, n=8)
- junior (scored 3/3): top-k precision=0.00, rubric pairwise-agreement=0.33, tau-b=-0.33, TF-IDF baseline=1.00, bare-LLM baseline=0.00
- mid (scored 3/3): top-k precision=0.00, rubric pairwise-agreement=0.33, tau-b=-0.33, TF-IDF baseline=1.00, bare-LLM baseline=0.33
- senior (scored 2/2): top-k precision=0.00, rubric pairwise-agreement=0.00, tau-b=-1.00, TF-IDF baseline=1.00, bare-LLM baseline=1.00
### rn_medsurg (registered_nurse, n=8)
- junior (scored 3/3): top-k precision=0.00, rubric pairwise-agreement=0.33, tau-b=0.00, TF-IDF baseline=1.00, bare-LLM baseline=0.00
- mid (scored 3/3): top-k precision=0.00, rubric pairwise-agreement=0.33, tau-b=n/a, TF-IDF baseline=1.00, bare-LLM baseline=0.67
- senior (scored 2/2): top-k precision=1.00, rubric pairwise-agreement=1.00, tau-b=1.00, TF-IDF baseline=1.00, bare-LLM baseline=1.00

## Methodology notes
- Ranking ground truth was labeled by Claude itself (see eval/ranking_dataset.json's own methodology field), reading resume/JD text directly and blind to the corpus's generation parameters, ranked within tier (not across tiers) per JD to avoid the tier confound design decision 1 flagged. This is a real circularity, not a human/'non-recruiter' labeler as originally planned — an LLM judging the data used to grade an LLM scorer. Disclosed as a limitation, not glossed over.
- Headline ranking metric is per-JD, per-tier pairwise agreement, not a single blended number — same convention as Projects 1 and 2 (don't blend distinct axes).
- Kendall tau-b is reported as a secondary/supporting number only — at this n (a handful of candidates per tier per JD) it has no usable confidence interval.
- Extraction eval uses n=1 per resume (not bias-sensitive); the bias eval (eval/bias_eval.py, eval/bias_results.md) uses n>=3 reruns where sampling variance actually matters.
- Call estimate before this run: {'extraction_calls': 40, 'scoring_calls': 72, 'baseline_llm_calls': 9, 'total_calls': 121, 'estimated_cost_usd': 1.94}