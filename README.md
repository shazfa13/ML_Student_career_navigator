# AI Student Success & Career Navigator

A simple B.Tech CSE mini-project built with **Python + Streamlit**.

This repository is being built in stages.
- **Day 1:** Career Navigator module
- **Day 2:** Student Struggle Prediction module (added on top of Day 1, without changing it)
- **Day 3:** Integration into one app with a shared Dashboard and a Model Information page

## Navigation (Day 3)

The app opens with a landing page and a short student-profile onboarding
step. After selecting **Get Started** and completing the profile, it shows
a single, clear top-level navigation with 4 tabs:

- 🏠 **Dashboard** — a one-glance summary pulling from both modules
- 🎯 **Career Navigator** — Day 1 features, organized into 5 clear steps
- 📚 **Academic Risk Prediction** — Day 2 features, organized into 4 clear steps
- 📊 **Model Information** — plain-language explanation of both techniques, plus metrics

## Day 1 — Career Navigator (Complete)

Features:
1. Student profile (name, year, CGPA, current skills, target career)
2. Current skill selection (multiselect)
3. Career recommendation using **cosine similarity**
4. Skill-gap analysis for a chosen target career
5. Simple, rule-based learning roadmap

Careers covered: Data Analyst, Data Scientist, Machine Learning
Engineer, Full Stack Developer, Cloud Engineer, Cybersecurity Analyst.

> ⚠️ Note: The career recommendation is a **skill compatibility score**
> (based on how closely your skills match a career's requirements),
> **not** a real job-placement prediction. All career/skill data is
> hand-created for this project, not scraped from real job postings.

## Day 2 — Student Struggle Prediction (Complete)

A small ML module that predicts a student's **academic risk level**
using a **Random Forest Classifier**.

Features used:
- Attendance (%)
- Assignment Completion (%)
- Quiz Average (%)
- Previous Marks (%)
- Study Hours per Week

Target classes: `Low Risk`, `Medium Risk`, `High Risk`

The model is trained on a small **synthetic, hand-generated** academic
dataset (`student_data.csv`, 1,000 rows, no real student data). Training
happens once per app session (cached), on a stratified 75/25 train/test split, and
is evaluated with accuracy, precision, recall, F1-score, and a
confusion matrix — all shown in the app under "📊 Model Performance".

The prediction screen also shows the top 3 factors driving each
prediction, using the Random Forest's built-in `feature_importances_`
(no SHAP is used).

> ⚠️ Note: This is a learning-support estimate based on a small
> synthetic dataset, not an official academic evaluation.

## Day 3 — Integration & Dashboard (Complete)

Day 3 does not add any new ML logic. It reorganizes the existing Day 1
and Day 2 features into one coherent app:

- A single top-level 4-tab navigation (see "Navigation" above) replaces
  the earlier 6-tab layout.
- The **Dashboard** now shows a real cross-module summary: student
  profile, selected career + compatibility %, number of skill gaps,
  and the current academic risk level — all in one place.
- The **Career Navigator** tab keeps every Day 1 feature (skills,
  recommendations, target career, skill gaps, roadmap), just organized
  into 5 numbered steps on one page instead of separate tabs.
- The **Academic Risk Prediction** tab keeps every Day 2 feature
  (inputs, prediction, important factors, model performance),
  organized into 4 numbered steps.
- A new **Model Information** tab explains Cosine Similarity and the
  Random Forest Classifier in plain language and shows the Random
  Forest's evaluation metrics (the same metrics already computed in
  Day 2 — nothing was recalculated differently).

No ML algorithm was changed and no existing feature was removed. The
academic dataset and Random Forest training pipeline were improved without
changing the Career Navigator module.

## Project Structure

```
AI_Student_Career_Navigator/
├── app.py             # Streamlit app: 4-tab navigation (Dashboard,
│                       #   Career Navigator, Academic Risk Prediction,
│                       #   Model Information)
├── career_data.py      # Day 1: career -> skill -> importance dataset
├── model.py            # Day 2: Random Forest training/prediction logic
├── student_data.csv   # Day 2: synthetic academic dataset (1,000 rows)
├── requirements.txt   # Python dependencies
└── README.md
```

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn (cosine similarity for Day 1, Random Forest for Day 2)

No databases, APIs, Flask/FastAPI, React, LangChain, deep learning,
XGBoost, SHAP, NLP, or web scraping are used.

## Installation

1. (Optional but recommended) Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the App

```
streamlit run app.py
```

Then open the URL Streamlit prints in your terminal (usually
`http://localhost:8501`).

## How to Use

1. Select **Get Started** on the landing page and complete your student
  profile. This information is stored in session state and used across
  every tab.
2. Open the **🏠 Dashboard** tab any time for a one-glance summary of
   your profile, selected career, career compatibility, skill gaps,
   and current academic risk level.
3. Open the **🎯 Career Navigator** tab to walk through: your skills,
   career recommendations (sorted by compatibility), choosing a target
   career, its skill gaps, and your learning roadmap — all on one page.
4. Open the **📚 Academic Risk Prediction** tab, enter your academic
   details, click **Predict Academic Risk**, and see the prediction,
   the top contributing factors, and the model's performance metrics.
5. Open the **📊 Model Information** tab for a plain-language
   explanation of Cosine Similarity and the Random Forest Classifier,
   along with the Random Forest's evaluation results.

## Roadmap for Future Days

- Further polish/integration as needed (e.g. deeper links between
  career goals and academic risk insights).

## Known Limitations

**Day 1:**
- The skill and career dataset is small and hand-picked for demo
  purposes — it is not exhaustive.
- Selecting zero skills shows "N/A" for career compatibility and skill
  gaps on the Dashboard, and a warning in the Career Navigator tab,
  instead of an error.
- Profile data (name, year, CGPA) is currently for display purposes
  only and does not yet affect the career recommendation.

**Day 2:**
- The academic dataset contains 1,000 synthetic rows with feature noise,
  overlap, and a composite risk relationship. It is meant to demonstrate
  the ML pipeline, not to be a production-accurate predictor. Evaluation
  scores are reported from the held-out test set and can change when the
  generator or model settings change.
- Slider inputs are already restricted to valid ranges (0-100% / 0-40
  hours), so out-of-range values can't be entered from the UI.

**Day 3:**
- The Dashboard's "Academic Risk Snapshot" reflects whatever values are
  currently set on the sliders in the Academic Risk Prediction tab (or
  their defaults, if you haven't visited that tab yet) — it does not
  require clicking the "Predict Academic Risk" button first, so the
  Dashboard is always up to date.
- The Dashboard's "Career Snapshot" reflects the target career chosen
  in the Career Navigator tab's "Target Career" step (defaults to the
  first career in the list if none has been chosen yet).
