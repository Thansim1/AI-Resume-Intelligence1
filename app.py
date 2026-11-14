"""AI Resume Intelligence — educational resume/job-description matching demo."""
from __future__ import annotations

import csv
import html
import io
from datetime import datetime
from pathlib import Path
import streamlit as st

from model import compute_match_score, generate_suggestions, score_label
from parser import basic_clean, parse_raw_text, parse_uploaded_file
from utils import (
    extract_skills,
    extract_skills_flat,
    get_matching_skills,
    get_missing_skills,
    skill_match_rate,
)

st.set_page_config(
    page_title="AI Resume Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

SAMPLE_RESUME = (Path(__file__).parent / "data" / "sample_resume.txt").read_text(encoding="utf-8")
SAMPLE_JOB = (Path(__file__).parent / "data" / "sample_job.txt").read_text(encoding="utf-8")


def analyze_resume(resume_text: str, job_text: str) -> dict:
    """Run text similarity and dictionary-based skill comparison."""
    resume_clean = basic_clean(resume_text)
    job_clean = basic_clean(job_text)
    if not resume_clean or not job_clean:
        raise ValueError("Both the resume and job description must contain readable text.")

    tfidf_score = compute_match_score(resume_clean, job_clean)
    resume_skills = extract_skills_flat(resume_clean)
    job_skills = extract_skills_flat(job_clean)
    job_categories = extract_skills(job_clean)

    skill_score = skill_match_rate(resume_skills, job_skills)
    combined_score = round((tfidf_score * 0.60) + (skill_score * 0.40), 1)
    matched = get_matching_skills(resume_skills, job_skills)
    missing = get_missing_skills(resume_skills, job_skills)
    additional = resume_skills - job_skills
    label, _ = score_label(combined_score)

    return {
        "tfidf_score": tfidf_score,
        "skill_score": skill_score,
        "combined_score": combined_score,
        "label": label,
        "matched": sorted(matched),
        "missing": sorted(missing),
        "additional": sorted(additional),
        "suggestions": generate_suggestions(missing, combined_score, job_categories),
    }


def create_pdf_report(result: dict) -> bytes:
    """Build a downloadable PDF report in memory."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=17 * mm, bottomMargin=17 * mm,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitleCustom", parent=styles["Title"], alignment=TA_CENTER,
        textColor=colors.HexColor("#17365D"), spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        name="SmallNoteCustom", parent=styles["BodyText"], fontSize=8,
        leading=11, textColor=colors.HexColor("#555555"),
    ))

    story = [
        Paragraph("AI Resume Intelligence", styles["ReportTitleCustom"]),
        Paragraph("Educational resume and job-description matching report", styles["Normal"]),
        Spacer(1, 7 * mm),
        Paragraph("Analysis summary", styles["Heading2"]),
    ]
    rows = [
        ["Measure", "Result"],
        ["Combined matching score", f'{result["combined_score"]}%'],
        ["TF-IDF cosine similarity", f'{result["tfidf_score"]}%'],
        ["Required-skill match rate", f'{result["skill_score"]}%'],
        ["Match category", html.escape(result["label"])],
    ]
    table = Table(rows, colWidths=[95 * mm, 65 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#C9D2E3")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F3F6FA")),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.extend([table, Spacer(1, 5 * mm)])

    for heading, values in (
        ("Matched skills", result["matched"]),
        ("Missing job skills", result["missing"]),
        ("Additional resume skills", result["additional"]),
    ):
        story.append(Paragraph(heading, styles["Heading2"]))
        story.append(Paragraph(html.escape(", ".join(values) if values else "None detected"), styles["BodyText"]))
        story.append(Spacer(1, 3 * mm))

    story.append(Paragraph("Improvement suggestions", styles["Heading2"]))
    for suggestion in result["suggestions"]:
        safe = html.escape(suggestion.replace("**", "").replace("`", ""))
        story.append(Paragraph(f"&bull; {safe}", styles["BodyText"]))
        story.append(Spacer(1, 2 * mm))

    story.extend([
        Spacer(1, 5 * mm),
        Paragraph("Method and limitations", styles["Heading2"]),
        Paragraph(
            "Combined score = TF-IDF cosine similarity (60%) + required-skill match rate (40%). "
            "This is a heuristic text-matching indicator, not model accuracy, hiring probability, "
            "or a validated measure of candidate quality. Dictionary matching may miss synonyms "
            "and context. A human reviewer must verify all results.",
            styles["SmallNoteCustom"],
        ),
        Spacer(1, 3 * mm),
        Paragraph(f'Generated: {datetime.now().strftime("%d %b %Y, %I:%M %p")}', styles["SmallNoteCustom"]),
    ])
    doc.build(story)
    return buffer.getvalue()


def render_skill_list(title: str, skills: list[str], kind: str) -> None:
    st.markdown(f"#### {title}")
    if not skills:
        st.caption("None detected.")
        return
    pills = " ".join(
        f'<span class="skill {kind}">{html.escape(skill)}</span>' for skill in skills
    )
    st.markdown(pills, unsafe_allow_html=True)


def render_analysis(result: dict) -> None:
    st.divider()
    st.subheader("Analysis results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Combined match", f'{result["combined_score"]}%')
    c1.caption(result["label"])
    c2.metric("Matched skills", len(result["matched"]))
    c3.metric("Missing skills", len(result["missing"]))
    c4.metric("Additional skills", len(result["additional"]))
    st.progress(max(0.0, min(result["combined_score"] / 100, 1.0)))
    st.caption(
        f'TF-IDF similarity: {result["tfidf_score"]}%  |  '
        f'Required-skill match: {result["skill_score"]}%'
    )

    left, right = st.columns(2)
    with left:
        render_skill_list("✅ Matched skills", result["matched"], "match")
        render_skill_list("➕ Additional resume skills", result["additional"], "extra")
    with right:
        render_skill_list("⚠️ Missing job skills", result["missing"], "missing")

    st.markdown("#### Improvement suggestions")
    for suggestion in result["suggestions"]:
        st.info(suggestion)

    with st.expander("How is the score calculated?"):
        st.write(
            f'{result["tfidf_score"]}% × 0.60 + {result["skill_score"]}% × 0.40 '
            f'= **{result["combined_score"]}%**'
        )
        st.caption("The weights are illustrative and have not been calibrated against recruitment outcomes.")

    chart = {
        "Skill group": ["Matched", "Missing", "Additional"],
        "Count": [len(result["matched"]), len(result["missing"]), len(result["additional"])],
    }
    st.bar_chart(chart, x="Skill group", y="Count")
    st.download_button(
        "Download PDF analysis report",
        data=create_pdf_report(result),
        file_name="AI_Resume_Intelligence_Report.pdf",
        mime="application/pdf",
        type="primary",
    )


# Visual styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: Inter, sans-serif; }
.stApp { background: #0b1020; color: #edf2ff; }
.hero { padding: 1.5rem 1.7rem; border-radius: 18px;
 background: linear-gradient(120deg,#102b46,#172044 60%,#28204a);
 border: 1px solid #33466d; margin-bottom: 1rem; }
.hero h1 { margin: 0; color: #f4f7ff; font-size: 2.25rem; }
.hero p { margin: .45rem 0 0; color: #bdc9e5; }
.skill { display:inline-block; padding:.28rem .65rem; margin:.18rem;
 border-radius:999px; font-size:.82rem; border:1px solid; }
.match { color:#4adeb2; background:#10352f; border-color:#208a70; }
.missing { color:#ff9999; background:#3a202b; border-color:#a34a60; }
.extra { color:#91c5ff; background:#172e4b; border-color:#4278ad; }
div[data-testid="stMetric"] { background:#131d33; padding:14px;
 border-radius:12px; border:1px solid #2a3a5b; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🧠 AI Resume Intelligence")
    st.caption("Resume screening and job matching demonstration")
    page = st.radio("Navigation", ["Resume Analyzer", "Batch Resume Ranking", "How It Works", "About"])
    st.divider()
    st.caption("Developed by V. Abdul Rahim Thansim")
    st.caption("Educational portfolio project")

st.markdown("""
<div class="hero">
  <h1>🧠 AI Resume Intelligence</h1>
  <p>Resume screening, skill-gap analysis and job-description matching</p>
</div>
""", unsafe_allow_html=True)

if page == "How It Works":
    st.subheader("System workflow")
    st.markdown("""
1. **Input:** Provide a resume and a target job description.
2. **Text extraction:** Read PDF/TXT files or accept pasted text.
3. **Preprocessing:** Normalize text, tokenize it and remove common stopwords.
4. **Text similarity:** Build TF-IDF vectors and calculate cosine similarity.
5. **Skill comparison:** Find skills using a curated dictionary.
6. **Combined score:** Weight text similarity at 60% and skill match at 40%.
7. **Feedback:** Generate rule-based improvement suggestions.
8. **Export:** Download a PDF report or batch-ranking CSV.
""")
    st.subheader("Technology stack")
    st.write("Python · Streamlit · PyMuPDF · ReportLab · TF-IDF · Cosine Similarity · Regular Expressions")
    st.subheader("Limitations")
    st.markdown("""
- TF-IDF measures word overlap; it does not understand meaning like a modern language model.
- Skill extraction depends on a curated dictionary and may miss synonyms, abbreviations or context.
- Scanned/image-only PDFs are not OCR-processed.
- The score is a heuristic, not hiring probability, model accuracy or a validated candidate assessment.
- Never use this tool as the sole basis for recruitment decisions.
- Use fictional/sample resumes for public demonstrations.
""")

elif page == "About":
    st.subheader("About this project")
    st.write(
        "AI Resume Intelligence is an educational Python application that compares resume content "
        "with a job description, identifies skill overlap and gaps, and produces explainable feedback."
    )
    st.markdown("**Developer:** V. Abdul Rahim Thansim")
    st.markdown("**Academic background:** B.Tech Computer Science and Engineering student")
    st.markdown("**Project focus:** Python, text processing, information retrieval and data analysis")
    st.markdown("**Methods:** Custom TF-IDF, cosine similarity and dictionary-based skill extraction")
    st.warning(
        "This is a portfolio demonstration, not a production hiring system. A human reviewer must "
        "check the source documents and all extracted results."
    )
    with st.expander("Acknowledgements and source-code notes"):
        st.write(
            "The project files in this package are provided as a consolidated educational implementation. "
            "If you reuse code from another repository, retain its original author, licence and attribution "
            "in NOTICE.md before publishing."
        )

elif page == "Resume Analyzer":
    st.subheader("Analyze a resume")
    st.write("Upload a PDF/TXT resume and job description, or paste text directly.")

    if "resume_text_box" not in st.session_state:
        st.session_state.resume_text_box = ""
    if "job_text_box" not in st.session_state:
        st.session_state.job_text_box = ""

    sample_a, sample_b = st.columns(2)
    with sample_a:
        if st.button("Load fictional sample resume", use_container_width=True):
            st.session_state.resume_text_box = SAMPLE_RESUME
            st.rerun()
    with sample_b:
        if st.button("Load sample job description", use_container_width=True):
            st.session_state.job_text_box = SAMPLE_JOB
            st.rerun()

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("### 📄 Candidate resume")
        resume_file = st.file_uploader("Upload resume (PDF or TXT)", type=["pdf", "txt"], key="resume_upload")
        resume_input = st.text_area("Or paste resume text", height=250, key="resume_text_box")
    with right:
        st.markdown("### 💼 Job description")
        job_file = st.file_uploader("Upload job description (PDF or TXT)", type=["pdf", "txt"], key="job_upload")
        job_input = st.text_area("Or paste job description", height=250, key="job_text_box")

    if st.button("🔎 Analyze match", type="primary", use_container_width=True):
        try:
            resume_text = parse_uploaded_file(resume_file) if resume_file else parse_raw_text(resume_input)
            job_text = parse_uploaded_file(job_file) if job_file else parse_raw_text(job_input)
            if not resume_text.strip():
                st.error("Please upload or paste a resume.")
            elif not job_text.strip():
                st.error("Please upload or paste a job description.")
            else:
                with st.spinner("Analyzing resume and job requirements..."):
                    st.session_state.latest_analysis = analyze_resume(resume_text, job_text)
        except Exception as exc:
            st.error(f"Could not complete the analysis: {exc}")

    if st.session_state.get("latest_analysis"):
        render_analysis(st.session_state.latest_analysis)
    st.divider()
    st.caption("Privacy: use fictional data for public demos. Avoid uploading confidential or sensitive resumes.")

elif page == "Batch Resume Ranking":
    st.subheader("Compare multiple resumes")
    st.write("Compare several PDF/TXT resumes against one job description using the same heuristic score.")
    job_description = st.text_area("Paste the target job description", height=180)
    resume_files = st.file_uploader(
        "Upload multiple resumes (PDF or TXT)", type=["pdf", "txt"],
        accept_multiple_files=True, key="batch_resumes",
    )
    if st.button("Rank uploaded resumes", type="primary", use_container_width=True):
        if not job_description.strip():
            st.error("Please provide a job description.")
        elif not resume_files:
            st.error("Please upload at least one resume.")
        else:
            ranking_rows, errors = [], []
            progress = st.progress(0)
            for index, uploaded in enumerate(resume_files):
                try:
                    resume_text = parse_uploaded_file(uploaded)
                    if not resume_text.strip():
                        raise ValueError("No readable text found.")
                    result = analyze_resume(resume_text, job_description)
                    ranking_rows.append({
                        "Resume file": uploaded.name,
                        "Match score (%)": result["combined_score"],
                        "TF-IDF (%)": result["tfidf_score"],
                        "Skill match (%)": result["skill_score"],
                        "Matched skills": ", ".join(result["matched"]),
                        "Missing skills": ", ".join(result["missing"]),
                    })
                except Exception as exc:
                    errors.append(f"{uploaded.name}: {exc}")
                progress.progress((index + 1) / len(resume_files))
            ranking_rows.sort(key=lambda row: row["Match score (%)"], reverse=True)
            st.session_state.batch_results = ranking_rows
            st.session_state.batch_errors = errors

    rows = st.session_state.get("batch_results", [])
    if rows:
        st.divider()
        st.subheader("Candidate comparison")
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.bar_chart(
            {"Resume": [r["Resume file"] for r in rows],
             "Match score (%)": [r["Match score (%)"] for r in rows]},
            x="Resume", y="Match score (%)",
        )
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        st.download_button("Download ranking as CSV", output.getvalue(),
                           file_name="resume_ranking.csv", mime="text/csv")
        for error in st.session_state.get("batch_errors", []):
            st.warning(error)
    elif st.session_state.get("batch_errors"):
        for error in st.session_state.batch_errors:
            st.warning(error)
