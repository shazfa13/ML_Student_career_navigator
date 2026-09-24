"""
career_data.py

This file holds a small, HAND-CREATED (not scraped) dataset that maps
each career to the skills it needs and how important each skill is.

Importance weights are on a 0.0 - 1.0 scale:
    1.0 = absolutely essential
    0.5 = somewhat useful
    0.0 = not needed (skill simply won't appear in that career's dict)

NOTE FOR DAY 2 INTEGRATION:
This module only deals with CAREER data. The student-struggle-prediction
module (Day 2) will use a different dataset (academic records), so it
will live in its own file (e.g. student_data.py) and will NOT need to
touch anything in here. Keeping this file career-only avoids conflicts.
"""

# ---------------------------------------------------------------------
# CAREER DATASET
# Each career maps to a dictionary of {skill_name: importance_weight}
# ---------------------------------------------------------------------
CAREER_DATA = {
    "Data Analyst": {
        "Excel": 0.9,
        "SQL": 0.9,
        "Statistics": 0.8,
        "Python": 0.7,
        "Data Visualization": 0.8,
        "Power BI": 0.7,
        "Communication": 0.6,
    },
    "Data Scientist": {
        "Python": 0.9,
        "Statistics": 0.9,
        "Machine Learning": 0.9,
        "SQL": 0.7,
        "Data Visualization": 0.7,
        "Pandas": 0.8,
        "Deep Learning": 0.5,
        "Communication": 0.5,
    },
    "Machine Learning Engineer": {
        "Python": 0.9,
        "Machine Learning": 0.9,
        "Deep Learning": 0.8,
        "Statistics": 0.7,
        "Data Structures": 0.7,
        "Model Deployment": 0.7,
        "SQL": 0.5,
        "Cloud Computing": 0.6,
    },
    "Full Stack Developer": {
        "HTML/CSS": 0.9,
        "JavaScript": 0.9,
        "React": 0.8,
        "Node.js": 0.7,
        "SQL": 0.6,
        "Git": 0.7,
        "Data Structures": 0.6,
        "REST APIs": 0.7,
    },
    "Cloud Engineer": {
        "Cloud Computing": 0.9,
        "Linux": 0.8,
        "Networking": 0.7,
        "Docker": 0.7,
        "Kubernetes": 0.6,
        "Python": 0.5,
        "Security Basics": 0.6,
        "Git": 0.5,
    },
    "Cybersecurity Analyst": {
        "Networking": 0.9,
        "Security Basics": 0.9,
        "Linux": 0.7,
        "Cryptography": 0.6,
        "Python": 0.5,
        "Ethical Hacking": 0.7,
        "Risk Assessment": 0.6,
    },
}


def get_all_skills():
    """
    Returns a sorted list of every unique skill mentioned across all
    careers. This list is used to build the skill multiselect in the
    Streamlit sidebar and to build vectors for cosine similarity.
    """
    all_skills = set()
    for skill_dict in CAREER_DATA.values():
        all_skills.update(skill_dict.keys())
    return sorted(all_skills)


def get_career_names():
    """Returns a list of all career names in the dataset."""
    return list(CAREER_DATA.keys())
