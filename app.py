
# ============================================================
# AI RESUME SCREENING & CANDIDATE RANKING SYSTEM
# ============================================================

import streamlit as st
import pandas as pd
import joblib
import re
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR
OUTPUTS_DIR = BASE_DIR


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    tfidf = None
    encoder = None

    tfidf_path = MODELS_DIR / "tfidf.pkl"
    encoder_path = MODELS_DIR / "label_encoder.pkl"

    if tfidf_path.exists():
        try:
            tfidf = joblib.load(tfidf_path)
        except Exception:
            pass

    if encoder_path.exists():
        try:
            encoder = joblib.load(encoder_path)
        except Exception:
            pass

    return tfidf, encoder


tfidf, encoder = load_models()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    data = {}

    files = {
        "ranking": "final_candidate_ranking.csv",
        "explainable": "explainable_candidate_profiles.csv",
        "recruiter": "recruiter_candidate_profiles.csv",
        "dashboard": "recruiter_dashboard_data.csv",
        "role_analytics": "role_wise_analytics.csv",
        "top_role": "top_candidate_per_role.csv",
        "skill_gap": "skill_gap_analysis.csv",
        "comparison": "candidate_comparison.csv"
    }

    for key, filename in files.items():

        path = OUTPUTS_DIR / filename

        if path.exists():
            try:
                data[key] = pd.read_csv(path)
            except Exception:
                data[key] = pd.DataFrame()
        else:
            data[key] = pd.DataFrame()

    return data


data = load_data()

ranking_df = data["ranking"]
explainable_df = data["explainable"]
role_df = data["role_analytics"]
top_role_df = data["top_role"]
skill_gap_df = data["skill_gap"]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_value(value):

    if pd.isna(value):
        return "None"

    value = str(value)

    if value.lower() in ["nan", "none", ""]:
        return "None"

    return value


def clean_text(text):

    if text is None:
        return ""

    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9+#.\- ]",
        " ",
        text
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# SKILLS
# ============================================================

screening_skills = [
    "python",
    "sql",
    "machine learning",
    "deep learning",
    "data analysis",
    "data visualization",
    "statistics",
    "nlp",
    "tensorflow",
    "pytorch",
    "spark",
    "hadoop",
    "aws",
    "azure",
    "gcp",
    "google cloud",
    "docker",
    "kubernetes",
    "git",
    "gitlab",
    "java",
    "javascript",
    "power bi",
    "tableau",
    "excel",
    "r",
    "api",
    "mongodb",
    "mysql",
    "postgresql",
    "etl",
    "mlops",
    "computer vision",
    "feature engineering",
    "data science",
    "artificial intelligence"
]


def extract_skills(text):

    text = clean_text(text)

    found = []

    for skill in screening_skills:
        if skill in text:
            found.append(skill)

    return sorted(set(found))


# ============================================================
# REAL-TIME SCREENING
# ============================================================

def screen_candidate(resume_text, job_description):

    resume = clean_text(resume_text)
    job = clean_text(job_description)

    resume_skills = set(extract_skills(resume))
    required_skills = set(extract_skills(job))

    matched = sorted(
        resume_skills.intersection(required_skills)
    )

    missing = sorted(
        required_skills.difference(resume_skills)
    )

    if required_skills:
        skill_score = (
            len(matched) / len(required_skills)
        ) * 100
    else:
        skill_score = 0

    similarity = 0

    if tfidf is not None:

        try:

            vectors = tfidf.transform(
                [resume, job]
            )

            similarity = (
                cosine_similarity(
                    vectors[0],
                    vectors[1]
                )[0][0]
                * 100
            )

        except Exception:
            similarity = 0

    final_score = (
        0.60 * skill_score
        +
        0.40 * similarity
    )

    if final_score >= 75:
        recommendation = "Highly Recommended"

    elif final_score >= 60:
        recommendation = "Recommended"

    elif final_score >= 40:
        recommendation = "Consider"

    else:
        recommendation = "Not Recommended"

    if len(matched) >= 3 and skill_score >= 75:
        evidence = "Strong Evidence"

    elif matched or similarity >= 20:
        evidence = "Moderate Evidence"

    else:
        evidence = "Weak Evidence"

    return {
        "skill_match_score": round(skill_score, 2),
        "text_similarity_score": round(similarity, 2),
        "final_match_score": round(final_score, 2),
        "recommendation": recommendation,
        "matched_skills": ", ".join(matched) if matched else "None",
        "missing_skills": ", ".join(missing) if missing else "None",
        "evidence_level": evidence
    }


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🤖 Recruiter Controls")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Dashboard",
        "🏆 Candidate Ranking",
        "🔎 Candidate Search",
        "⚖️ Compare Candidates",
        "👤 Candidate Profile",
        "🤖 Real-Time Screening",
        "📄 Resume Upload",
        "📊 Role Analytics",
        "📈 Skill Gap Analysis",
        "🧠 Explainable AI"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title("🤖 AI Resume Screening & Candidate Ranking System")

    st.caption(
        "Intelligent recruitment analytics powered by NLP, "
        "Machine Learning and Explainable AI"
    )

    if ranking_df.empty:

        st.error("Final ranking file could not be loaded.")

    else:

        total_candidates = len(ranking_df)

        average_score = ranking_df[
            "final_match_score"
        ].mean()

        best_score = ranking_df[
            "final_match_score"
        ].max()

        highly = (
            ranking_df["final_recommendation"]
            .astype(str)
            .eq("Highly Recommended")
            .sum()
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "👥 Total Candidates",
            f"{total_candidates:,}"
        )

        c2.metric(
            "📈 Average Match",
            f"{average_score:.2f}%"
        )

        c3.metric(
            "🏆 Best Match",
            f"{best_score:.2f}%"
        )

        c4.metric(
            "⭐ Highly Recommended",
            highly
        )

        st.divider()

        st.subheader("📌 Recommendation Distribution")

        distribution = (
            ranking_df["final_recommendation"]
            .value_counts()
        )

        st.bar_chart(distribution)

        st.subheader("🏆 Top 10 Candidates")

        top10 = (
            ranking_df
            .sort_values(
                "final_match_score",
                ascending=False
            )
            .head(10)
            .copy()
        )

        top10.insert(
            0,
            "Rank",
            range(1, len(top10) + 1)
        )

        columns = [
            "Rank",
            "Name",
            "Role",
            "skill_match_score",
            "text_similarity_score",
            "final_match_score",
            "final_recommendation"
        ]

        columns = [
            col for col in columns
            if col in top10.columns
        ]

        st.dataframe(
            top10[columns],
            width="stretch",
            hide_index=True
        )


# ============================================================
# CANDIDATE RANKING
# ============================================================

elif page == "🏆 Candidate Ranking":

    st.title("🏆 Candidate Ranking")

    if ranking_df.empty:

        st.error("Ranking data unavailable.")

    else:

        min_score = st.slider(
            "Minimum Final Match Score",
            0.0,
            100.0,
            0.0
        )

        recommendations = st.multiselect(
            "Recommendation",
            sorted(
                ranking_df[
                    "final_recommendation"
                ]
                .dropna()
                .unique()
                .tolist()
            )
        )

        filtered = ranking_df[
            ranking_df["final_match_score"] >= min_score
        ].copy()

        if recommendations:

            filtered = filtered[
                filtered["final_recommendation"]
                .isin(recommendations)
            ]

        filtered = (
            filtered
            .sort_values(
                "final_match_score",
                ascending=False
            )
            .reset_index(drop=True)
        )

        filtered.insert(
            0,
            "Rank",
            range(1, len(filtered) + 1)
        )

        st.write(
            f"Showing **{len(filtered):,} candidates**"
        )

        st.dataframe(
            filtered,
            width="stretch",
            hide_index=True
        )

        st.download_button(
            "📥 Download Ranking",
            filtered.to_csv(index=False),
            "candidate_ranking.csv",
            "text/csv"
        )


# ============================================================
# CANDIDATE SEARCH
# ============================================================

elif page == "🔎 Candidate Search":

    st.title("🔎 Advanced Candidate Search")

    if ranking_df.empty:

        st.error("Ranking data unavailable.")

    else:

        search = st.text_input(
            "Search candidate name, ID, role or skill"
        )

        role_options = sorted(
            ranking_df["Role"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_role = st.selectbox(
            "Role",
            ["All Roles"] + role_options
        )

        result = ranking_df.copy()

        if search:

            mask = pd.Series(
                False,
                index=result.index
            )

            search_columns = [
                "Name",
                "ID",
                "Role",
                "matched_skills_text",
                "missing_skills_text"
            ]

            for col in search_columns:

                if col in result.columns:

                    mask = (
                        mask
                        |
                        result[col]
                        .astype(str)
                        .str.contains(
                            search,
                            case=False,
                            na=False
                        )
                    )

            result = result[mask]

        if selected_role != "All Roles":

            result = result[
                result["Role"]
                .astype(str)
                .str.lower()
                ==
                selected_role.lower()
            ]

        result = (
            result
            .sort_values(
                "final_match_score",
                ascending=False
            )
            .head(100)
        )

        st.metric(
            "Matching Candidates",
            len(result)
        )

        st.dataframe(
            result,
            width="stretch",
            hide_index=True
        )


# ============================================================
# COMPARE CANDIDATES
# ============================================================

elif page == "⚖️ Compare Candidates":

    st.title("⚖️ Candidate Comparison")

    if ranking_df.empty:

        st.error("Ranking data unavailable.")

    else:

        options_df = ranking_df[
            ["ID", "Name", "Role"]
        ].copy()

        options_df["label"] = (
            options_df["Name"].astype(str)
            + " | "
            + options_df["Role"].astype(str)
            + " | "
            + options_df["ID"].astype(str)
        )

        selected = st.multiselect(
            "Select 2–5 candidates",
            options_df["label"].tolist(),
            max_selections=5
        )

        if len(selected) >= 2:

            selected_ids = []

            for item in selected:

                row = options_df[
                    options_df["label"] == item
                ]

                if not row.empty:
                    selected_ids.append(
                        row.iloc[0]["ID"]
                    )

            comparison = ranking_df[
                ranking_df["ID"].isin(selected_ids)
            ].copy()

            comparison = comparison.sort_values(
                "final_match_score",
                ascending=False
            )

            columns = [
                "Name",
                "Role",
                "skill_match_score",
                "text_similarity_score",
                "selection_probability",
                "final_match_score",
                "final_recommendation",
                "matched_skills_text",
                "missing_skills_text"
            ]

            columns = [
                col for col in columns
                if col in comparison.columns
            ]

            st.dataframe(
                comparison[columns],
                width="stretch",
                hide_index=True
            )

        else:

            st.info("Select at least 2 candidates.")


# ============================================================
# CANDIDATE PROFILE
# ============================================================

elif page == "👤 Candidate Profile":

    st.title("👤 Candidate Profile")

    if ranking_df.empty:

        st.error("Candidate data unavailable.")

    else:

        names = sorted(
            ranking_df["Name"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_name = st.selectbox(
            "Select Candidate",
            names
        )

        candidate = ranking_df[
            ranking_df["Name"]
            .astype(str)
            == selected_name
        ]

        if not candidate.empty:

            row = candidate.iloc[0]

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Final Score",
                f"{row['final_match_score']:.2f}%"
            )

            c2.metric(
                "Skill Match",
                f"{row['skill_match_score']:.2f}%"
            )

            c3.metric(
                "Text Similarity",
                f"{row['text_similarity_score']:.2f}%"
            )

            st.subheader(
                f"{row['Name']} — {row['Role']}"
            )

            st.write(
                "**Recommendation:**",
                safe_value(
                    row.get(
                        "final_recommendation",
                        "None"
                    )
                )
            )

            st.write(
                "**Matched Skills:**",
                safe_value(
                    row.get(
                        "matched_skills_text",
                        "None"
                    )
                )
            )

            st.write(
                "**Missing Skills:**",
                safe_value(
                    row.get(
                        "missing_skills_text",
                        "None"
                    )
                )
            )


# ============================================================
# REAL-TIME SCREENING
# ============================================================

elif page == "🤖 Real-Time Screening":

    st.title("🤖 Real-Time Resume Screening")

    resume_text = st.text_area(
        "📄 Resume",
        height=250,
        placeholder="Paste candidate resume here..."
    )

    job_text = st.text_area(
        "💼 Job Description",
        height=250,
        placeholder="Paste job description here..."
    )

    if st.button(
        "🚀 Screen Candidate",
        type="primary"
    ):

        if not resume_text.strip():

            st.warning("Please enter a resume.")

        elif not job_text.strip():

            st.warning("Please enter a job description.")

        else:

            result = screen_candidate(
                resume_text,
                job_text
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Skill Match",
                f"{result['skill_match_score']:.2f}%"
            )

            c2.metric(
                "Text Similarity",
                f"{result['text_similarity_score']:.2f}%"
            )

            c3.metric(
                "Final Score",
                f"{result['final_match_score']:.2f}%"
            )

            st.subheader(
                f"Recommendation: {result['recommendation']}"
            )

            st.write(
                "**Evidence Level:**",
                result["evidence_level"]
            )

            col1, col2 = st.columns(2)

            with col1:

                st.success("✅ Matched Skills")

                st.write(
                    result["matched_skills"]
                )

            with col2:

                st.warning("⚠️ Missing Skills")

                st.write(
                    result["missing_skills"]
                )

            st.subheader("🧠 AI Explanation")

            st.info(
                f"Final match score: "
                f"{result['final_match_score']:.2f}%. "
                f"Evidence level: "
                f"{result['evidence_level']}."
            )


# ============================================================
# RESUME UPLOAD
# ============================================================

elif page == "📄 Resume Upload":

    st.title("📄 Resume Upload & AI Screening")

    col1, col2 = st.columns(2)

    with col1:

        resume_file = st.file_uploader(
            "Upload Resume (.txt)",
            type=["txt"]
        )

    with col2:

        jd_file = st.file_uploader(
            "Upload Job Description (.txt)",
            type=["txt"]
        )

    resume_text = ""
    jd_text = ""

    if resume_file is not None:

        resume_text = (
            resume_file.getvalue()
            .decode(
                "utf-8",
                errors="ignore"
            )
        )

        st.success(
            f"Resume uploaded: {resume_file.name}"
        )

    if jd_file is not None:

        jd_text = (
            jd_file.getvalue()
            .decode(
                "utf-8",
                errors="ignore"
            )
        )

        st.success(
            f"Job description uploaded: {jd_file.name}"
        )

    if st.button(
        "🚀 Analyze Candidate",
        type="primary"
    ):

        if not resume_text.strip():

            st.warning(
                "Please upload a resume first."
            )

        elif not jd_text.strip():

            st.warning(
                "Please upload a job description first."
            )

        else:

            result = screen_candidate(
                resume_text,
                jd_text
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Skill Match",
                f"{result['skill_match_score']:.2f}%"
            )

            c2.metric(
                "Text Similarity",
                f"{result['text_similarity_score']:.2f}%"
            )

            c3.metric(
                "Final Match",
                f"{result['final_match_score']:.2f}%"
            )

            st.success(
                f"Recommendation: "
                f"{result['recommendation']}"
            )

            st.write(
                "**Matched Skills:**",
                result["matched_skills"]
            )

            st.write(
                "**Missing Skills:**",
                result["missing_skills"]
            )

            result_df = pd.DataFrame([result])

            st.download_button(
                "📥 Download Screening Result",
                result_df.to_csv(index=False),
                "screening_result.csv",
                "text/csv"
            )


# ============================================================
# ROLE ANALYTICS
# ============================================================

elif page == "📊 Role Analytics":

    st.title("📊 Role-Wise Analytics")

    if role_df.empty:

        st.warning(
            "role_wise_analytics.csv not available."
        )

    else:

        st.dataframe(
            role_df,
            width="stretch",
            hide_index=True
        )

        if (
            "Average_Score" in role_df.columns
            and "standard_role" in role_df.columns
        ):

            chart = (
                role_df
                .set_index("standard_role")
                ["Average_Score"]
                .sort_values(
                    ascending=False
                )
                .head(15)
            )

            st.subheader(
                "🏆 Top Roles by Average Score"
            )

            st.bar_chart(chart)

        if not top_role_df.empty:

            st.subheader(
                "🥇 Top Candidate for Each Role"
            )

            st.dataframe(
                top_role_df,
                width="stretch",
                hide_index=True
            )


# ============================================================
# SKILL GAP ANALYSIS
# ============================================================

elif page == "📈 Skill Gap Analysis":

    st.title("📈 Candidate Skill Gap Analysis")

    if skill_gap_df.empty:

        st.warning(
            "Skill gap analysis file not available."
        )

    else:

        c1, c2 = st.columns(2)

        c1.metric(
            "Unique Missing Skills",
            len(skill_gap_df)
        )

        if "Candidates_Missing" in skill_gap_df.columns:

            c2.metric(
                "Total Skill Gap Occurrences",
                int(
                    skill_gap_df[
                        "Candidates_Missing"
                    ].sum()
                )
            )

        if (
            "Skill" in skill_gap_df.columns
            and "Candidates_Missing" in skill_gap_df.columns
        ):

            top_gaps = (
                skill_gap_df
                .head(20)
                .set_index("Skill")
                ["Candidates_Missing"]
            )

            st.bar_chart(top_gaps)

        st.dataframe(
            skill_gap_df,
            width="stretch",
            hide_index=True
        )


# ============================================================
# EXPLAINABLE AI
# ============================================================

elif page == "🧠 Explainable AI":

    st.title(
        "🧠 Explainable AI — Candidate Decision"
    )

    if explainable_df.empty:

        st.warning(
            "Explainable candidate data unavailable."
        )

    else:

        names = sorted(
            explainable_df["Name"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_name = st.selectbox(
            "Select Candidate",
            names
        )

        candidate = explainable_df[
            explainable_df["Name"]
            .astype(str)
            == selected_name
        ]

        if not candidate.empty:

            row = candidate.iloc[0]

            score = safe_value(
                row.get(
                    "final_match_score",
                    "N/A"
                )
            )

            recommendation = safe_value(
                row.get(
                    "final_recommendation",
                    "N/A"
                )
            )

            st.metric(
                "Final Match Score",
                f"{score}%"
            )

            st.subheader(
                f"Recommendation: {recommendation}"
            )

            st.write(
                "**Matched Skills:**",
                safe_value(
                    row.get(
                        "matched_skills_text",
                        "None"
                    )
                )
            )

            st.write(
                "**Missing Skills:**",
                safe_value(
                    row.get(
                        "missing_skills_text",
                        "None"
                    )
                )
            )

            if "explanation_type" in row.index:

                st.write(
                    "**Evidence Level:**",
                    safe_value(
                        row["explanation_type"]
                    )
                )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "AI Resume Screening System"
)

st.sidebar.caption(
    "NLP • Machine Learning • Explainable AI"
)
