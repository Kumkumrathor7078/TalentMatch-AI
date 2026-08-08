# TalentMatch-AI 🤖

## AI-Powered Resume Screening & Candidate Matching System

TalentMatch-AI is an end-to-end **Machine Learning and NLP-based recruitment screening application** designed to help recruiters evaluate how well a candidate's resume matches a given job description.

The system analyzes the uploaded resume and job description, identifies relevant skills, calculates text similarity, generates an overall match score, and provides an AI-assisted recommendation.

The project combines **data processing, machine learning, NLP-based text analysis, candidate ranking, recruitment analytics, and Streamlit deployment** into a single practical application.

---

## 🚀 Live Demo

🔗 [**Launch TalentMatch-AI**](https://talentmatch-ai-nlfnsrrggg4rpznpporwa4.streamlit.app/)

> Try the live TalentMatch-AI application for AI-powered resume screening and candidate matching.

> Replace `YOUR_STREAMLIT_APP_URL` with the Streamlit URL after deployment.

---

## 🎯 Problem Statement

Recruiters often have to review a large number of resumes for a single job opening.

Manually comparing every resume with the corresponding job description can be:

* Time-consuming
* Difficult to scale
* Repetitive
* Challenging when many candidates have similar profiles

TalentMatch-AI aims to simplify the **initial resume screening process** by automatically comparing candidate resumes with job requirements and presenting structured matching insights.

---

# 💡 Solution

TalentMatch-AI follows an automated screening workflow:

```text
Resume
   +
Job Description
        ↓
Text Processing
        ↓
Skill Matching
        ↓
Text Similarity
        ↓
Match Score Calculation
        ↓
Candidate Recommendation
        ↓
Recruitment Insights
```

The system provides recruiters with a quick overview of candidate-job compatibility while keeping the final hiring decision with humans.

---

# ✨ Key Features

### 📄 Resume Upload

Upload a candidate resume in `.txt` format through the Streamlit interface.

### 💼 Job Description Upload

Upload the corresponding job description in `.txt` format.

### 🎯 Skill Matching

The application identifies skills present in both the resume and job description and calculates a **Skill Match Score**.

### 🔎 Text Similarity

The system compares the resume content with the job description and generates a **Text Similarity Score**.

### ⭐ Final Match Score

The application combines the matching information to generate an overall **Final Match Score**.

### 💡 Candidate Recommendation

Based on the generated matching score, the system provides an AI-assisted recommendation such as:

**Recommended**

### 🔗 Matched Skills

The application displays skills identified as relevant to both the candidate profile and job requirements.

Example:

```text
Data Science
Git
Machine Learning
NLP
```

### 📊 Candidate & Recruitment Analytics

The project also generates structured recruitment outputs for:

* Candidate ranking
* Candidate comparison
* Candidate profiles
* Explainable candidate profiles
* Recruiter search results
* Recruiter dashboard data
* Final candidate ranking

---

# 📊 Example Application Result

A sample run using a **Data Scientist resume** and **Data Scientist job description** produced:

| Metric          |           Score |
| --------------- | --------------: |
| Skill Match     |      **75.00%** |
| Text Similarity |      **52.79%** |
| Final Match     |      **66.11%** |
| Recommendation  | **Recommended** |

### Matched Skills

```text
Data Science
Git
Machine Learning
NLP
```

These results demonstrate how the application converts resume and job-description information into structured candidate matching insights.

---

# 🧠 Machine Learning & NLP Workflow

The project follows a practical Data Science workflow:

```text
Raw Resume Data
      ↓
Data Processing
      ↓
Text Preparation
      ↓
Skill Analysis
      ↓
Text Similarity
      ↓
Candidate Matching
      ↓
Scoring
      ↓
Recommendation
```

The machine learning development and experimentation are documented separately in the project's Jupyter Notebook.

---

# 📓 Model Development

The repository contains the Jupyter Notebook used during the development process:

```text
AI_Resume_Model.ipynb
```

The notebook provides the development-side view of the project, while:

```text
app.py
```

contains the Streamlit application used for the deployed interface.

This separation demonstrates the complete transition from **model development to application deployment**.

---

# 📁 Project Structure

```text
TalentMatch-AI/
│
├── app.py
├── AI_Resume_Model.ipynb
├── requirements.txt
├── README.md
│
├── label_encoder.pkl
│
├── candidate_comparison.csv
├── candidate_ranking.csv
├── explainable_candidate_profiles.csv
├── final_candidate_ranking.csv
├── recruiter_candidate_profiles.csv
├── recruiter_dashboard_data.csv
└── recruiter_search_results.csv
```

---

# 📊 Generated Recruitment Outputs

The application/project generates multiple structured CSV outputs.

### `candidate_comparison.csv`

Contains candidate comparison results.

### `candidate_ranking.csv`

Contains candidate ranking information.

### `explainable_candidate_profiles.csv`

Contains structured candidate information used for explainable recruitment insights.

### `final_candidate_ranking.csv`

Contains final candidate ranking results.

### `recruiter_candidate_profiles.csv`

Contains recruiter-oriented candidate profile information.

### `recruiter_dashboard_data.csv`

Contains structured data for recruitment dashboard analysis.

### `recruiter_search_results.csv`

Contains candidate search-related results.

---

# 🛠️ Technology Stack

| Technology           | Usage                                 |
| -------------------- | ------------------------------------- |
| **Python**           | Application and ML development        |
| **Pandas**           | Data processing                       |
| **NumPy**            | Numerical operations                  |
| **Scikit-learn**     | Machine Learning / text processing    |
| **Jupyter Notebook** | Model development and experimentation |
| **Streamlit**        | Interactive web application           |
| **Pickle**           | Saved model/encoding artifact         |
| **Git & GitHub**     | Version control and project hosting   |

---

# 💻 Run the Project Locally

## 1. Clone the repository

```bash
git clone https://github.com/Kumkumrathor7078/TalentMatch-AI.git
```

## 2. Open the project directory

```bash
cd TalentMatch-AI
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Start the Streamlit application

```bash
streamlit run app.py
```

The application will then open in your browser.

---

# 🌐 Streamlit Deployment

The application is designed to run as a **Streamlit web application** connected to the GitHub repository.

Deployment workflow:

```text
GitHub Repository
       ↓
requirements.txt
       ↓
app.py
       ↓
Streamlit
       ↓
Live Web Application
```

Users can interact with the deployed application by uploading:

1. Resume
2. Job Description

and receiving:

* Skill Match
* Text Similarity
* Final Match
* Recommendation
* Matched Skills

---

# 📈 End-to-End Project Workflow

TalentMatch-AI demonstrates the complete progression from data science development to deployment:

```text
Data
 ↓
Data Processing
 ↓
NLP / Text Analysis
 ↓
Machine Learning
 ↓
Candidate Matching
 ↓
Candidate Scoring
 ↓
Recruitment Analytics
 ↓
Streamlit Application
 ↓
GitHub
 ↓
Cloud Deployment
```

---

# 🎯 Project Objectives

The primary objectives of TalentMatch-AI are to:

* Automate the initial resume screening process
* Compare resumes with job descriptions
* Identify relevant candidate skills
* Generate measurable matching scores
* Provide candidate recommendations
* Produce structured recruitment analytics
* Demonstrate an end-to-end Machine Learning project
* Deploy the ML application as an interactive web application

---

# 🔮 Future Improvements

Possible future enhancements include:

* PDF resume support
* DOCX resume support
* Multiple resume batch processing
* More advanced semantic text matching
* Improved skill extraction
* Interactive recruiter dashboards
* Advanced candidate comparison
* Skill-gap analysis
* Database integration
* Authentication for recruiters

---

# ⚠️ Responsible Use

TalentMatch-AI is designed as an **AI-assisted screening and decision-support system**.

The generated scores and recommendations should not be treated as the sole basis for employment decisions.

Final recruitment decisions should always involve appropriate human evaluation and consideration of the candidate's complete qualifications and experience.

---

# 👩‍💻 Author

## Kumkum Rathor

**MSc Mathematics | NIT Warangal**

Aspiring **Data Scientist | Machine Learning & AI Enthusiast**

---

# ⭐ Project Highlights

| Area                     | Implementation |
| ------------------------ | -------------- |
| Resume Screening         | ✅              |
| Job Description Matching | ✅              |
| Skill Matching           | ✅              |
| Text Similarity          | ✅              |
| Final Match Score        | ✅              |
| Candidate Recommendation | ✅              |
| Candidate Ranking        | ✅              |
| Recruitment Analytics    | ✅              |
| Machine Learning         | ✅              |
| NLP / Text Analysis      | ✅              |
| Streamlit Application    | ✅              |
| GitHub Repository        | ✅              |
| Cloud Deployment         | ✅              |

---

## 📌 Final Takeaway

**TalentMatch-AI** demonstrates how Machine Learning and NLP-based text analysis can be transformed into a practical recruitment application.

The project covers the complete journey from **data processing and model development to candidate matching, recruitment analytics, interactive application development, and cloud deployment**.

> **From Resume Data → Intelligent Matching → Actionable Recruitment Insights.**


