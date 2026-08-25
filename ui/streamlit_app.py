import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Resume Screening & Ranking", page_icon="📋")
st.title("📋 Resume Screening & Ranking")
st.caption(
    "Paste a job description and upload candidate resumes (.txt) to get a "
    "ranked list with a per-criterion rationale for each score. "
    "Scoring is name-blind by default — the model never sees a candidate's "
    "name, only their skills, experience, and education."
)
st.info(
    "🔒 **Don't paste real personal data.** This is a portfolio demo — "
    "resumes are sent to the Anthropic API for scoring and are never "
    "written to disk by this app, but treat it like any other public demo: "
    "use a synthetic/sample resume, not your own or anyone else's.",
    icon="🔒",
)

job_description = st.text_area("Job description", height=150,
                                placeholder="Paste the job description here...")

uploaded_files = st.file_uploader("Candidate resumes (.txt)", type=["txt"],
                                   accept_multiple_files=True)

if st.button("Rank candidates", type="primary") and job_description and uploaded_files:
    resumes = []
    for f in uploaded_files:
        # Decode immediately and drop the reference — this app never writes
        # upload bytes to disk itself. (Streamlit's own file_uploader keeps
        # small files like these in memory; very large uploads can be
        # transiently spooled to a temp file by Streamlit internally,
        # outside this app's control — not a realistic concern for
        # resume-sized text files, disclosed here for completeness.)
        resumes.append({"name": f.name, "text": f.read().decode("utf-8", errors="replace")})
    uploaded_files = None  # drop references now that text is extracted

    results = []
    progress = st.progress(0.0)
    for i, r in enumerate(resumes):
        try:
            resp = requests.post(f"{API_URL}/score",
                                  json={"resume_text": r["text"], "job_description": job_description},
                                  timeout=100)
        except requests.Timeout:
            st.error(f"{r['name']}: API timed out (likely a free-tier cold start) — try again.")
            continue
        except requests.RequestException as e:
            st.error(f"{r['name']}: could not reach the API ({e}).")
            continue
        if resp.status_code != 200:
            try:
                detail = resp.json().get("detail", "Something went wrong.")
            except ValueError:
                detail = f"Something went wrong (HTTP {resp.status_code})."
            st.warning(f"{r['name']}: {detail}")
            continue
        results.append({"file_name": r["name"], **resp.json()})
        progress.progress((i + 1) / len(resumes))
    progress.empty()

    if results:
        ranked = sorted(results, key=lambda x: x["weighted_total"], reverse=True)
        st.subheader("Ranked candidates")
        st.table([{"Rank": i + 1, "File": r["file_name"], "Score": round(r["weighted_total"], 2)}
                   for i, r in enumerate(ranked)])

        for i, r in enumerate(ranked):
            with st.expander(f"#{i + 1} — {r['file_name']} (score: {r['weighted_total']:.2f})"):
                for criterion, detail in r["rubric"].items():
                    st.markdown(f"**{criterion.replace('_', ' ').title()}**: {detail['score']}/5")
                    st.caption(detail["rationale"])
