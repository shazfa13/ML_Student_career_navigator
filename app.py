"""
app.py

AI Student Success & Career Navigator - DAY 1 + DAY 2 + DAY 3
Modules:
    - Career Navigator (Day 1)
    - Student Struggle Prediction (Day 2)
    - Integration & Dashboard (Day 3)

DAY 1 features (unchanged):
    1. Student profile (sidebar form)
    2. Current skill selection (multiselect)
    3. Career recommendation (cosine similarity)
    4. Skill-gap analysis
    5. Simple learning roadmap (rule-based, no AI API used)

DAY 2 features (unchanged, logic lives in model.py):
    6. Academic risk prediction using a RandomForestClassifier
       (trained on student_data.csv)

DAY 3 features (new, added below without changing Day 1/Day 2 logic):
    7. A single top-level navigation with 4 clear sections:
       Dashboard, Career Navigator, Academic Risk Prediction, Model Info
    8. A home Dashboard that pulls a quick summary from BOTH modules
    9. A Model Information page explaining Cosine Similarity and
       Random Forest in plain language, with evaluation metrics

This file is intentionally kept simple (no classes, no extra
architecture) so it is easy to read, demo, and extend in later days.
"""

import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

from career_data import CAREER_DATA, get_all_skills, get_career_names
from project_data import PROJECT_DATA
import model as struggle_model  # DAY 2: Student Struggle Prediction logic

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="ML-Based Student Success & Career Navigator",
    page_icon="🎓",
    layout="wide",
)

ALL_SKILLS = get_all_skills()
CAREER_NAMES = get_career_names()

st.markdown(
    """
    <style>
    :root { --navy: #17233f; --ink: #26324d; --muted: #596780; --blue: #405bd6; --purple: #6552c7; --line: #d9e1ee; --surface: #ffffff; --page: #eef2f8; --sidebar: #e4ebf7; }
    .stApp { background: var(--page); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(238, 242, 248, 0.96); }
    [data-testid="stSidebar"] { background: var(--sidebar); border-right: 1px solid #c9d5e7; }
    [data-testid="stSidebar"] * { color: var(--navy); }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { font-size: 0.95rem; }
    [data-testid="stSidebar"] hr { border-color: #c3d0e3; }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 { color: var(--navy); }
    .stApp p, .stApp label, .stApp [data-testid="stCaptionContainer"] { color: var(--muted); }
    .stApp [data-testid="stWidgetLabel"] p, .stApp [data-testid="stWidgetLabel"] label { color: var(--navy); font-weight: 600; }
    .eyebrow { color: var(--purple); font-size: 0.82rem; font-weight: 800; letter-spacing: 0.1em; text-transform: uppercase; }
    .hero { padding: 3.7rem 0 2.3rem; margin: 0 auto; max-width: 900px; text-align: center; }
    .hero h1 { color: var(--navy); font-size: clamp(2.65rem, 5.5vw, 4.6rem); line-height: 1.04; margin: 0.6rem 0 1.25rem; letter-spacing: 0; }
    .hero p { color: var(--muted); font-size: clamp(1.15rem, 2vw, 1.4rem); line-height: 1.55; margin: 0 auto; max-width: 760px; }
    .intro-card { background: var(--surface); border: 1px solid var(--line); border-radius: 16px; padding: 1.45rem 1.5rem; min-height: 150px; box-shadow: 0 10px 26px rgba(31, 55, 94, 0.08); }
    .intro-card h3 { margin: 0 0 0.7rem; color: var(--navy); font-size: 1.12rem; }
    .intro-card p { color: var(--muted); line-height: 1.65; margin: 0; font-size: 1rem; }
    .feature-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; max-width: 1050px; margin: 0.7rem auto 2rem; }
    .feature-card { background: var(--surface); border: 1px solid var(--line); border-radius: 16px; padding: 1.35rem 1.4rem; min-height: 145px; box-shadow: 0 10px 26px rgba(31, 55, 94, 0.08); }
    .feature-card h3 { color: var(--navy); font-size: 1.08rem; margin: 0 0 0.65rem; }
    .feature-card p { color: var(--muted); font-size: 0.97rem; line-height: 1.55; margin: 0; }
    .landing-action { text-align: center; margin: 0.3rem 0 3rem; }
    .metric-card { background: var(--surface); border: 1px solid var(--line); border-radius: 14px; padding: 1rem 1.1rem; min-height: 92px; box-shadow: 0 5px 18px rgba(33, 53, 86, 0.06); }
    .metric-label { color: var(--muted); font-size: 0.78rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.04em; }
    .metric-value { color: var(--navy); font-size: 1.35rem; font-weight: 800; margin-top: 0.45rem; }
    .risk-card { border-radius: 14px; padding: 1.15rem 1.35rem; margin: 0.6rem 0 1rem; font-size: 1.35rem; font-weight: 700; }
    .risk-low { background: #eaf8f0; border: 1px solid #b9e6ca; color: #187342; }
    .risk-medium { background: #fff7df; border: 1px solid #f0d98d; color: #87630b; }
    .risk-high { background: #fff0f0; border: 1px solid #efbcbc; color: #a42f2f; }
    div[data-testid="stMetric"] { background: white; border: 1px solid #e0e7f2; border-radius: 14px; padding: 0.75rem 1rem; }
    .stButton > button, [data-testid="stFormSubmitButton"] button { min-height: 50px; padding: 0.8rem 1.6rem; border-radius: 9px; font-size: 1.05rem; line-height: 1.2; }
    .stButton > button[kind="primary"] { background: linear-gradient(135deg, var(--blue), var(--purple)); border: 0; border-radius: 10px; color: #ffffff; font-weight: 700; padding: 0.8rem 1.6rem; }
    .stButton > button[kind="primary"] p { color: #ffffff; }
    [data-baseweb="select"], [data-baseweb="input"] { min-height: 48px; background: var(--surface); border: 2px solid var(--line); border-radius: 9px; }
    [data-baseweb="select"] *, [data-baseweb="input"] input { color: var(--navy) !important; }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div, [data-testid="stMultiSelect"] [data-baseweb="select"] > div { background: #ffffff !important; color: var(--navy) !important; }
    [data-testid="stSelectbox"] [data-baseweb="select"] svg, [data-testid="stMultiSelect"] [data-baseweb="select"] svg { fill: #6c5ce7 !important; }
    [data-baseweb="popover"], [role="listbox"] { background: #ffffff !important; color: var(--navy) !important; border: 1px solid #dfe3f1 !important; }
    [role="option"] { background: #ffffff !important; color: var(--navy) !important; }
    [role="option"]:hover, [aria-selected="true"] { background: #eef0ff !important; color: var(--navy) !important; }
    [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input { min-height: 46px; box-sizing: border-box; padding: 0.65rem 0.8rem; color: var(--navy); background: var(--surface); border: 2px solid var(--line); border-radius: 9px; font-size: 1rem; }
    [data-testid="stSelectbox"] [data-baseweb="select"], [data-testid="stMultiSelect"] [data-baseweb="select"] { min-height: 48px; }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div, [data-testid="stMultiSelect"] [data-baseweb="select"] > div { padding: 0.45rem 0.8rem; font-size: 1rem; }
    [data-testid="stSelectbox"] [data-baseweb="select"] { height: 48px !important; min-height: 48px !important; overflow: visible !important; }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div { height: 44px !important; min-height: 44px !important; padding: 0 0.8rem !important; display: flex !important; align-items: center !important; overflow: visible !important; }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div > div { min-height: 42px !important; display: flex !important; align-items: center !important; line-height: 1.35 !important; padding: 0 !important; overflow: visible !important; }
    [data-testid="stSlider"] [role="slider"] { min-height: 24px; }
    [data-testid="stMultiSelect"] [data-baseweb="tag"] { background: #dce4ff; color: var(--navy); }
    button[kind="primary"], [data-testid="stFormSubmitButton"] button { background: linear-gradient(135deg, #5d69ec, #bd54ce) !important; border: 0 !important; color: #ffffff !important; }
    button[kind="primary"] p, [data-testid="stFormSubmitButton"] button p { color: #ffffff !important; }
    [data-testid="stMetric"] { color: var(--navy); }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); }
    @media (max-width: 700px) { .feature-grid { grid-template-columns: 1fr; } .hero { padding-top: 2.3rem; } }

    /* Shared visual system */
    :root { --lavender: #f3f1ff; --periwinkle: #e9edff; --accent: #6c5ce7; --accent-blue: #5679f2; --pink: #df6ac8; }
    .block-container { max-width: 1240px; padding: 2.2rem 3.2rem 4rem; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #f4f3ff 0%, #edf2ff 100%); padding: 1rem 0.8rem; width: 220px !important; min-width: 220px !important; }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; width: 220px !important; }
    .brand { display: flex; align-items: center; gap: 0.55rem; color: #1d2b55; font-weight: 800; font-size: 1.08rem; margin: 0 0 0.2rem; }
    .brand-mark { display: inline-flex; align-items: center; justify-content: center; width: 1.8rem; height: 1.8rem; border-radius: 9px; background: linear-gradient(135deg, #836ef7, #526ee8); color: white; font-size: 0.95rem; }
    .brand-subtitle { color: #7782a2; font-size: 0.72rem; margin: 0 0 1.9rem 2.35rem; }
    .side-caption { color: #7b86a6; font-size: 0.72rem; line-height: 1.6; margin-top: 2rem; padding: 0.9rem 0.6rem; border-top: 1px solid #dfe3f5; }
    [data-testid="stSidebar"] .stButton > button { min-height: 42px; width: 100%; padding: 0.45rem 0.85rem; text-align: left; font-size: 0.86rem; border-radius: 9px; border: 1px solid transparent; background: transparent; color: #445477; box-shadow: none; }
    [data-testid="stSidebar"] .stButton > button > div { width: 100%; display: flex; justify-content: flex-start; align-items: center; }
    [data-testid="stSidebar"] .stButton > button p { width: 100%; margin: 0 !important; text-align: left !important; color: #445477 !important; line-height: 1.35; }
    [data-testid="stSidebar"] .stButton > button:hover { background: #e8e7fb; border-color: #deddf5; color: #413b9c; }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] { background: #e1defd !important; color: #4c43ae !important; border-color: #d4d0f7 !important; }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] p { color: #4c43ae !important; }
    .hero-shell { --hero-height: 560px; background: linear-gradient(120deg, #f8f7ff 0%, #eef3ff 100%); border: 1px solid #e0e4f5; border-radius: 22px; padding: 3rem 3.4rem; min-height: var(--hero-height); height: var(--hero-height); box-sizing: border-box; margin: 0.2rem 0 1.8rem; box-shadow: 0 18px 42px rgba(44, 65, 125, 0.10); overflow: hidden; }
    .hero-copy { padding: 1rem 0.5rem 0.4rem 0; }
    .hero-shell h1 { font-size: clamp(2.1rem, 4vw, 3.7rem); line-height: 1.06; letter-spacing: -0.02em; margin: 0.55rem 0 1.1rem; }
    .hero-shell h1 span { background: linear-gradient(90deg, #7358e8, #d764c8); -webkit-background-clip: text; background-clip: text; color: transparent; }
    .hero-shell .hero-lead { color: #556483; font-size: 1.12rem; line-height: 1.65; max-width: 560px; margin: 0 0 0.75rem; }
    .hero-shell .hero-support { color: #7b86a2; line-height: 1.65; max-width: 535px; margin: 0; }
    .hero-visual { min-height: 560px; height: 560px; border-radius: 18px; background: linear-gradient(145deg, rgba(255,255,255,.18), rgba(221,228,255,.26)), url('https://images.unsplash.com/photo-1523240795612-9a054b0db644?auto=format&fit=crop&w=900&q=80') center/cover; background-blend-mode: screen, normal; box-shadow: inset 0 0 0 1px rgba(255,255,255,.75); position: relative; filter: brightness(1.08) saturate(1.08); }
    [data-testid="stImage"] { height: 560px; min-height: 560px; border-radius: 18px; overflow: hidden; }
    [data-testid="stImage"] img { width: 100%; height: 100%; object-fit: cover; filter: brightness(1.08) saturate(1.08); }
    .hero-visual::after { content: 'Plan today. Grow tomorrow.'; position: absolute; right: 1rem; bottom: 1rem; max-width: 130px; color: #314276; font-size: 0.78rem; font-weight: 700; line-height: 1.35; background: rgba(255,255,255,.82); padding: 0.65rem; border-radius: 10px; }
    .section-kicker { color: #6d62d9; font-weight: 800; font-size: 0.78rem; letter-spacing: 0.1em; text-transform: uppercase; margin: 0 0 0.5rem; }
    .feature-grid { max-width: none; margin: 0 0 2rem; gap: 1.1rem; }
    .feature-card { border-radius: 14px; min-height: 150px; padding: 1.35rem 1.45rem; transition: transform .18s ease, box-shadow .18s ease; }
    .feature-card:hover { transform: translateY(-3px); box-shadow: 0 15px 30px rgba(45, 65, 120, 0.12); }
    .feature-icon { display: inline-flex; align-items: center; justify-content: center; width: 2.1rem; height: 2.1rem; border-radius: 50%; background: #ece9ff; color: #6b5dde; font-size: 1rem; margin-bottom: 0.65rem; }
    .form-shell { background: #fff; border: 1px solid #e1e5f2; border-radius: 18px; padding: 1.6rem 1.8rem 1.8rem; box-shadow: 0 15px 34px rgba(44, 65, 125, 0.08); margin-top: 1.2rem; }
    .stepper { display: flex; align-items: center; gap: 0.6rem; margin: 1rem 0 1.6rem; color: #687594; font-size: 0.78rem; font-weight: 700; }
    .step { display: flex; align-items: center; gap: 0.4rem; white-space: nowrap; }
    .step-dot { display: inline-flex; align-items: center; justify-content: center; width: 1.6rem; height: 1.6rem; border-radius: 50%; color: white; background: #6b5ce7; font-size: 0.72rem; }
    .step-muted .step-dot { background: #e7e9f6; color: #7c86a5; }
    .step-line { height: 1px; background: #d9ddef; width: 80px; }
    .form-shell [data-testid="stForm"] { border: 0; padding: 0; }
    .app-header { display: flex; align-items: flex-end; justify-content: space-between; gap: 1rem; padding: 0.4rem 0 1.6rem; border-bottom: 1px solid #e0e4f1; margin-bottom: 1.25rem; }
    .app-header h1 { font-size: 1.55rem; margin: 0.25rem 0 0; letter-spacing: 0; }
    .app-header p { margin: 0; color: #71809e; font-size: 0.9rem; }
    .page-intro { padding: 0.5rem 0 1.2rem; }
    .page-intro h2 { font-size: clamp(1.65rem, 3vw, 2.35rem); margin: 0.25rem 0 0.45rem; }
    .page-intro p { max-width: 720px; margin: 0; color: #687694; line-height: 1.6; }
    .section-title { color: #24345f; font-size: 1.05rem; font-weight: 800; margin: 1.35rem 0 0.8rem; }
    .metric-card { min-height: 108px; padding: 1.1rem 1.2rem; border-radius: 14px; }
    .metric-value { overflow-wrap: anywhere; }
    .soft-card { background: #ffffff; border: 1px solid #e1e5f2; border-radius: 16px; padding: 1.25rem 1.35rem; box-shadow: 0 9px 24px rgba(44, 65, 125, 0.07); }
    .soft-card h3 { margin: 0 0 0.45rem; font-size: 1.05rem; color: #24345f; }
    .soft-card p { margin: 0; color: #71809e; line-height: 1.55; }
    .pill { display: inline-block; padding: 0.35rem 0.7rem; margin: 0.25rem 0.25rem 0 0; border-radius: 999px; background: #eef0ff; border: 1px solid #dfe1fb; color: #4e58a3; font-size: 0.78rem; font-weight: 700; }
    .career-card { background: white; border: 1px solid #e1e5f2; border-radius: 15px; padding: 1.1rem 1.25rem; margin: 0.65rem 0; box-shadow: 0 7px 20px rgba(44, 65, 125, 0.055); }
    .career-card-top { display: flex; justify-content: space-between; align-items: center; gap: 1rem; }
    .career-card-title { color: #24345f; font-weight: 800; font-size: 1rem; }
    .career-score { color: #6558d7; font-size: 1.05rem; font-weight: 800; white-space: nowrap; }
    .compat-track { height: 8px; background: #eceefd; border-radius: 999px; overflow: hidden; margin-top: 0.8rem; }
    .compat-fill { height: 100%; background: linear-gradient(90deg, #6674ed, #b45bd2); border-radius: inherit; }
    .risk-card { border-radius: 15px; padding: 1.25rem 1.4rem; margin: 0.6rem 0 1rem; }
    .risk-card strong { display: block; font-size: 0.75rem; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.25rem; }
    .risk-card span { font-size: 1.45rem; }
    .journey { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.65rem; align-items: stretch; }
    .journey-step { position: relative; background: white; border: 1px solid #e1e5f2; border-radius: 14px; padding: 1rem; min-height: 105px; }
    .journey-step:not(:last-child)::after { content: '↓'; position: absolute; right: -0.8rem; top: 1.9rem; color: #877be4; font-weight: 800; z-index: 2; }
    .journey-step small { display: block; color: #7d88a5; letter-spacing: 0.08em; font-weight: 800; font-size: 0.65rem; }
    .journey-step strong { display: block; color: #2a3a65; margin-top: 0.55rem; font-size: 0.9rem; }
    .roadmap-flow { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 1.15rem; align-items: stretch; margin: 1.35rem 0 1rem; }
    .roadmap-stage { position: relative; background: #ffffff; border: 1px solid #dfe4f2; border-radius: 16px; padding: 1.1rem; min-height: 185px; box-shadow: 0 9px 24px rgba(44, 65, 125, 0.07); }
    .roadmap-stage:not(:last-child)::after { content: '→'; position: absolute; right: -1rem; top: 50%; transform: translateY(-50%); color: #796fe0; font-size: 1.35rem; font-weight: 800; z-index: 2; }
    .roadmap-stage.current { background: linear-gradient(180deg, #ffffff, #f5f7ff); }
    .roadmap-stage.gap { background: linear-gradient(180deg, #ffffff, #fff8fc); }
    .roadmap-stage.recommended { background: linear-gradient(180deg, #ffffff, #f5f2ff); border-color: #d9d1f6; }
    .roadmap-stage.project { background: linear-gradient(180deg, #ffffff, #f3f9ff); }
    .roadmap-stage.goal { background: linear-gradient(180deg, #ffffff, #f2f4ff); }
    .roadmap-stage-label { color: #7d88a5; font-size: 0.64rem; font-weight: 800; letter-spacing: 0.09em; line-height: 1.35; }
    .roadmap-stage h3 { color: #263762; font-size: 0.98rem; margin: 0.45rem 0 0.8rem; }
    .roadmap-node { display: block; background: #f0f1ff; border: 1px solid #dfe1fa; border-radius: 9px; color: #4c5797; font-size: 0.78rem; font-weight: 700; line-height: 1.3; padding: 0.48rem 0.55rem; margin: 0.35rem 0; }
    .roadmap-stage.gap .roadmap-node { background: #fff0f8; border-color: #f2d8e8; color: #9a4f7f; }
    .roadmap-stage.recommended .roadmap-node { background: #ebe8ff; border-color: #d9d1f6; color: #5a4bb1; }
    .roadmap-stage.project .roadmap-node { background: #eaf5ff; border-color: #d4e8fb; color: #416b95; }
    .roadmap-stage.goal .roadmap-node { background: #e6ebff; border-color: #d1daf8; color: #3f56a1; }
    .roadmap-empty { color: #8993ad; font-size: 0.78rem; line-height: 1.45; margin-top: 0.6rem; }
    .roadmap-legend { display: flex; flex-wrap: wrap; gap: 0.55rem; margin: 0.8rem 0 1.35rem; }
    .legend-item { display: inline-flex; align-items: center; gap: 0.35rem; color: #71809e; font-size: 0.75rem; }
    .legend-swatch { width: 0.75rem; height: 0.75rem; border-radius: 3px; background: #f0f1ff; border: 1px solid #dfe1fa; }
    .legend-swatch.gap { background: #fff0f8; border-color: #f2d8e8; }
    .legend-swatch.recommended { background: #ebe8ff; border-color: #d9d1f6; }
    .legend-swatch.project { background: #eaf5ff; border-color: #d4e8fb; }
    .legend-swatch.goal { background: #e6ebff; border-color: #d1daf8; }
    .roadmap-detail { background: #ffffff; border: 1px solid #dddaf5; border-left: 4px solid #796fe0; border-radius: 13px; padding: 1rem 1.15rem; box-shadow: 0 8px 20px rgba(44, 65, 125, 0.06); }
    .roadmap-detail h3 { margin: 0 0 0.45rem; color: #293966; font-size: 1rem; }
    .roadmap-detail p { margin: 0.3rem 0; color: #71809e; line-height: 1.5; font-size: 0.86rem; }
    @media (max-width: 900px) { .roadmap-flow { grid-template-columns: 1fr; gap: 0.7rem; } .roadmap-stage { min-height: auto; } .roadmap-stage:not(:last-child)::after { content: '↓'; right: 50%; top: auto; bottom: -1.05rem; transform: translateX(50%); } }
    .workflow { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.5rem; margin: 1rem 0 1.5rem; }
    .workflow-step { text-align: center; position: relative; background: #fff; border: 1px solid #e1e5f2; border-radius: 13px; padding: 1rem 0.6rem; color: #2b3b68; font-size: 0.85rem; font-weight: 700; }
    .workflow-step:not(:last-child)::after { content: '→'; position: absolute; right: -1.25rem; top: 1rem; color: #786de0; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.4rem; padding: 0.35rem; background: #e9ecfa; border-radius: 12px; }
    .stTabs [data-baseweb="tab"] { height: 2.4rem; border-radius: 9px; color: #6b7897; font-weight: 700; padding: 0 1rem; }
    .stTabs [aria-selected="true"] { background: #ffffff; color: #4e4ab2; box-shadow: 0 3px 10px rgba(53, 63, 125, 0.1); }
    .stProgress > div > div > div { background: linear-gradient(90deg, #6674ed, #b45bd2); }
    [data-testid="stAlert"] { border-radius: 12px; border: 1px solid #dfe3f2; }
    [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
    @media (max-width: 760px) { .journey, .workflow { grid-template-columns: 1fr 1fr; } .journey-step:not(:last-child)::after, .workflow-step:not(:last-child)::after { display: none; } .app-header { align-items: flex-start; flex-direction: column; } }
    @media (max-width: 700px) { .block-container { padding: 1.2rem 1rem 3rem; } .hero-shell { padding: 1.4rem; min-height: 0; height: auto; } .hero-visual, [data-testid="stImage"] { min-height: 260px; height: 260px; margin-top: 1rem; } .step-line { width: 28px; } }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------


def get_priority(importance):
    """
    Converts a skill's importance weight into a priority label + emoji.

    Rule:
        importance >= 0.8  -> High   (🔴)
        0.7 <= importance < 0.8 -> Medium (🟡)
        importance < 0.7   -> Low    (🟢)
    """
    if importance >= 0.8:
        return "🔴 High"
    elif importance >= 0.7:
        return "🟡 Medium"
    else:
        return "🟢 Low"


def render_metric_card(label, value):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def render_app_header():
    st.markdown(
        '<div class="app-header"><div><h1>ML-Based Student Success &amp; Career Navigator</h1></div>'
        '<p>Your next step, made clearer.</p></div>',
        unsafe_allow_html=True,
    )


def render_page_intro(kicker, title, description):
    st.markdown(
        f'<div class="page-intro"><div class="section-kicker">{kicker}</div>'
        f'<h2>{title}</h2><p>{description}</p></div>',
        unsafe_allow_html=True,
    )


def render_skill_pills(skills):
    if not skills:
        return '<span class="pill">No skills selected yet</span>'
    return "".join(f'<span class="pill">{skill}</span>' for skill in skills)


def render_compatibility_card(career, score):
    score_value = max(0, min(float(score), 100))
    st.markdown(
        f'<div class="career-card"><div class="career-card-top">'
        f'<span class="career-card-title">{career}</span>'
        f'<span class="career-score">{score_value:.1f}%</span></div>'
        f'<div class="compat-track"><div class="compat-fill" style="width:{score_value}%"></div></div>'
        f'<p style="margin:.55rem 0 0;color:#7b86a2;font-size:.8rem;">Skill compatibility</p></div>',
        unsafe_allow_html=True,
    )


def render_journey_step(label, title):
    return f'<div class="journey-step"><small>{label}</small><strong>{title}</strong></div>'


def roadmap_stage_markup(label, title, items, variant, empty_message):
    nodes = "".join(f'<span class="roadmap-node">{item}</span>' for item in items)
    if not nodes:
        nodes = f'<div class="roadmap-empty">{empty_message}</div>'
    return (
        f'<div class="roadmap-stage {variant}"><div class="roadmap-stage-label">{label}</div>'
        f'<h3>{title}</h3>{nodes}</div>'
    )


def recommended_topics(skill):
    topic_map = {
        "Python": "syntax, functions, data handling",
        "SQL": "queries, joins, aggregations",
        "Statistics": "probability, distributions, hypothesis testing",
        "Machine Learning": "features, model evaluation, validation",
        "Data Visualization": "charts, storytelling, dashboards",
        "JavaScript": "DOM, async code, browser APIs",
        "React": "components, state, data fetching",
        "Cloud Computing": "deployment, storage, monitoring",
        "Networking": "protocols, routing, troubleshooting",
        "Linux": "shell, permissions, processes",
    }
    return topic_map.get(skill, f"{skill} fundamentals, guided practice, real-world use")


def project_application(skill, target_career):
    return f"Build a small {target_career} project that applies {skill} to a real student or campus problem."


def render_risk_card(predicted_label):
    risk_class = {
        "Low Risk": "risk-low",
        "Medium Risk": "risk-medium",
        "High Risk": "risk-high",
    }.get(predicted_label, "risk-medium")
    st.markdown(
        f'<div class="risk-card {risk_class}"><strong>Academic risk</strong>'
        f'<span>{predicted_label}</span></div>',
        unsafe_allow_html=True,
    )


def build_student_vector(selected_skills):
    """
    Builds a binary vector (1 = has skill, 0 = doesn't) over ALL_SKILLS,
    in the same order as ALL_SKILLS, so it can be compared to career
    vectors using cosine similarity.
    """
    return np.array([1 if skill in selected_skills else 0 for skill in ALL_SKILLS])


def build_career_vector(career_name):
    """
    Builds a weighted vector for a given career over ALL_SKILLS.
    Skills not required by the career get a weight of 0.
    """
    career_skills = CAREER_DATA[career_name]
    return np.array([career_skills.get(skill, 0.0) for skill in ALL_SKILLS])


def calculate_compatibility(selected_skills):
    """
    Calculates a skill-compatibility percentage for every career using
    cosine similarity between the student's skill vector and each
    career's weighted requirement vector.

    Returns a DataFrame sorted from highest to lowest compatibility.
    """
    student_vector = build_student_vector(selected_skills)

    # Handle the "no skills selected" case to avoid a divide-by-zero
    # inside cosine_similarity (a zero vector has no direction).
    if student_vector.sum() == 0:
        results = [{"Career": career, "Compatibility (%)": 0.0} for career in CAREER_NAMES]
        return pd.DataFrame(results).sort_values(
            "Compatibility (%)", ascending=False
        ).reset_index(drop=True)

    results = []
    for career in CAREER_NAMES:
        career_vector = build_career_vector(career)
        similarity = cosine_similarity(
            student_vector.reshape(1, -1), career_vector.reshape(1, -1)
        )[0][0]
        results.append({"Career": career, "Compatibility (%)": round(similarity * 100, 1)})

    df = pd.DataFrame(results).sort_values(
        "Compatibility (%)", ascending=False
    ).reset_index(drop=True)
    return df


def build_skill_gap_table(target_career, selected_skills):
    """
    Builds a DataFrame showing, for the target career:
        - Required Skill
        - Importance
        - Whether the student already has it
        - Priority (only meaningful when the skill is missing)
    """
    rows = []
    for skill, importance in CAREER_DATA[target_career].items():
        has_skill = skill in selected_skills
        priority = "-" if has_skill else get_priority(importance)
        rows.append(
            {
                "Required Skill": skill,
                "Importance": importance,
                "Have It?": "✅ Yes" if has_skill else "❌ No",
                "Priority if Missing": priority,
            }
        )
    df = pd.DataFrame(rows).sort_values("Importance", ascending=False).reset_index(drop=True)
    return df


def build_learning_roadmap(target_career, selected_skills):
    """
    Builds a simple, rule-based, ORDERED roadmap of missing skills.

    Ordering rule:
        1. High-priority skills first
        2. Medium-priority skills next
        3. Low-priority skills last
        (within the same priority, higher importance comes first)

    This is plain Python logic - no AI/LLM API is used here.
    """
    priority_rank = {"🔴 High": 0, "🟡 Medium": 1, "🟢 Low": 2}

    missing_skills = []
    for skill, importance in CAREER_DATA[target_career].items():
        if skill not in selected_skills:
            priority = get_priority(importance)
            missing_skills.append(
                {"skill": skill, "importance": importance, "priority": priority}
            )

    # Sort by priority rank first, then by importance (descending)
    missing_skills.sort(key=lambda x: (priority_rank[x["priority"]], -x["importance"]))

    return missing_skills


def recommend_projects(target_career, selected_skills):
    """Rank projects using the target career, current skills, and career gaps."""
    selected_skill_set = set(selected_skills)
    gap_table = build_skill_gap_table(target_career, selected_skills)
    career_gap_set = set(gap_table.loc[gap_table["Have It?"] == "❌ No", "Required Skill"])
    recommendations = []

    for project in PROJECT_DATA:
        required_skills = set(project["required_skills"])
        gained_skills = set(project["skills_gained"])
        existing_matches = required_skills & selected_skill_set
        gap_coverage = career_gap_set & (required_skills | gained_skills)
        career_match = target_career in project["target_careers"]
        existing_score = len(existing_matches) / len(required_skills) if required_skills else 0
        gap_score = len(gap_coverage) / len(career_gap_set) if career_gap_set else 0
        relevance = round((50 if career_match else 0) + (30 * existing_score) + (20 * gap_score))

        recommendations.append(
            {
                **project,
                "relevance": relevance,
                "gap_coverage": gap_coverage,
            }
        )

    return sorted(recommendations, key=lambda item: (-item["relevance"], item["project_name"]))


def render_project_recommendations(target_career, selected_skills):
    """Render personalized project cards without adding another profile flow."""
    render_page_intro(
        "Project Recommendations",
        "Build your next career proof point.",
        "Build projects that strengthen your skills and move you closer to your target career.",
    )

    compatibility_df = calculate_compatibility(selected_skills)
    compatibility_row = compatibility_df.loc[compatibility_df["Career"] == target_career]
    compatibility_score = float(compatibility_row["Compatibility (%)"].iloc[0]) if not compatibility_row.empty else 0
    summary_cols = st.columns(2)
    with summary_cols[0]:
        render_metric_card("Target Career", target_career)
    with summary_cols[1]:
        render_metric_card("Career Compatibility", f"{compatibility_score:.1f}%")

    recommendations = recommend_projects(target_career, selected_skills)[:5]
    st.markdown('<div class="section-title">Recommended for You</div>', unsafe_allow_html=True)
    st.caption("Recommendations combine career relevance, skills you already have, and skills you can build next.")

    for project in recommendations:
        existing_skills = [skill for skill in project["required_skills"] if skill in selected_skills]
        why_parts = [f"matches your {target_career} goal"]
        if existing_skills:
            why_parts.append(f"builds on {', '.join(existing_skills[:3])}")
        if project["gap_coverage"]:
            why_parts.append(f"helps develop {', '.join(sorted(project['gap_coverage'])[:3])}")

        skill_markup = "".join(
            f'<span class="pill">{"✅" if skill in existing_skills else "⚠️"} {skill}</span>'
            for skill in project["required_skills"]
        )
        gained_markup = "".join(f'<span class="pill">{skill}</span>' for skill in project["skills_gained"])
        st.markdown(
            f'<div class="soft-card">'
            f'<div class="career-card-top"><h3>{project["project_name"]}</h3>'
            f'<span class="career-score">{project["relevance"]}%</span></div>'
            f'<p>{project["description"]}</p>'
            f'<p><strong>Difficulty:</strong> {project["difficulty"]} &nbsp; '
            f'<strong>Relevance:</strong> {project["relevance"]}%</p>'
            f'<p><strong>Why this project?</strong> This project is recommended because it '
            f'{"; ".join(why_parts)}.</p>'
            f'<p><strong>Required Skills</strong><br>{skill_markup}</p>'
            f'<p><strong>Skills You\'ll Gain</strong><br>{gained_markup}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if not recommendations:
        st.info("No projects are available for this career yet. Choose another target career to explore projects.")


# ----------------------------------------------------------------------
# DAY 2 HELPER: cache the trained Random Forest so it's only trained
# once per app session instead of retraining on every rerun/interaction.
# This does not touch any Day 1 (career) logic above.
# ----------------------------------------------------------------------
@st.cache_resource
def get_trained_struggle_model():
    return struggle_model.train_and_evaluate_model()


# ----------------------------------------------------------------------
# LANDING PAGE AND FIRST-RUN PROFILE FLOW
# ----------------------------------------------------------------------
def render_workspace_sidebar():
    st.sidebar.markdown(
        '<div class="brand"><span class="brand-mark">✦</span>Student Workspace</div>',
        unsafe_allow_html=True,
    )
    pages = [
        ("⌂", "Home", "home"),
        ("◎", "Career Navigator", "career"),
        ("💡", "Project Recommendations", "projects"),
        ("▥", "Academic Risk", "academic"),
        ("▤", "Skill Roadmap", "roadmap"),
        ("⚙", "Model Information", "model"),
    ]
    active_page = st.session_state.get("active_page", "home")
    for icon, label, page_key in pages:
        button_label = f"{icon}  {label}"
        if st.sidebar.button(
            button_label,
            key=f"nav_{page_key}",
            use_container_width=True,
            type="primary" if active_page == page_key else "secondary",
        ):
            st.session_state["active_page"] = page_key
            st.rerun()
    if st.session_state.get("profile_complete", False):
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**{st.session_state['student_name']}**")
        st.sidebar.caption(
            f"{st.session_state['student_year']} · CGPA {st.session_state['student_cgpa']:.1f}"
        )
        if st.sidebar.button("✎  Edit Profile", key="edit_profile", use_container_width=True):
            st.session_state["profile_complete"] = False
            st.rerun()
    st.sidebar.markdown(
        '<div class="side-caption">Small steps today.<br><strong>Better opportunities tomorrow.</strong></div>',
        unsafe_allow_html=True,
    )


if not st.session_state.get("app_started", False):
    hero_left, hero_right = st.columns([1.08, 0.92], gap="large")
    with hero_left:
        st.markdown(
            '<div class="hero-shell"><div class="hero-copy">'
            '<div class="eyebrow">Student success workspace</div>'
            '<h1>ML-Based Student Success &amp;<br><span>Career Navigator</span></h1>'
            '<p class="hero-lead">Understand your academic risk. Discover your career path. Build the skills to get there.</p>'
            '<p class="hero-support">One thoughtful workspace for academic-risk prediction, career guidance, and skill-gap analysis, designed to help you make your next step clearer.</p>'
            '</div></div>',
            unsafe_allow_html=True,
        )
        if st.button("Get Started →", type="primary"):
            st.session_state["app_started"] = True
            st.rerun()
    with hero_right:
        hero_image = Path("assets/student_hero.jpg")
        if hero_image.exists():
            st.image(str(hero_image), use_container_width=True)
        else:
            st.markdown('<div class="hero-visual" aria-label="Student studying at a desk"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-kicker">A clearer way forward</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="feature-grid">'
        '<div class="feature-card"><div class="feature-icon">⌁</div><h3>Career Navigator</h3>'
        '<p>Find careers that align with your current technical skills.</p></div>'
        '<div class="feature-card"><div class="feature-icon">◌</div><h3>Academic Risk</h3>'
        '<p>Understand your academic risk using machine learning.</p></div>'
        '<div class="feature-card"><div class="feature-icon">↗</div><h3>Skill Roadmap</h3>'
        '<p>Identify important skill gaps and follow a simple learning path.</p></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()

render_workspace_sidebar()

if not st.session_state.get("profile_complete", False):
    st.header("Let's personalize your experience")
    st.markdown("Tell us a little about yourself to get started.")
    st.markdown(
        '<div class="stepper"><div class="step"><span class="step-dot">1</span> Basic details</div>'
        '<div class="step-line"></div><div class="step step-muted"><span class="step-dot">2</span> Results</div></div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        with st.form("student_profile_form", border=False):
            profile_left, profile_right = st.columns(2, gap="large")
            with profile_left:
                profile_name = st.text_input("Student Name", value=st.session_state.get("student_name", ""), placeholder="Enter your full name")
                profile_year = st.selectbox(
                    "Year of Study",
                    ["1st Year", "2nd Year", "3rd Year", "4th Year"],
                    index=["1st Year", "2nd Year", "3rd Year", "4th Year"].index(
                        st.session_state.get("student_year", "1st Year")
                    ),
                )
                profile_cgpa = st.number_input(
                    "CGPA", min_value=0.0, max_value=10.0,
                    value=st.session_state.get("student_cgpa", 7.5), step=0.1,
                )
            with profile_right:
                profile_skills = st.multiselect(
                    "Current Technical Skills",
                    options=ALL_SKILLS,
                    default=st.session_state.get("student_skills", []),
                    placeholder="Python, Java, SQL, React...",
                    help="Select the skills you currently have.",
                )
                profile_career = st.selectbox(
                    "Target Career",
                    options=CAREER_NAMES,
                    index=CAREER_NAMES.index(st.session_state.get("target_career", CAREER_NAMES[0])),
                )

            submitted = st.form_submit_button("Continue →", type="primary", use_container_width=True)

    if submitted:
        if not profile_name.strip():
            st.warning("Please enter your name to continue.")
        else:
            st.session_state["student_name"] = profile_name.strip()
            st.session_state["student_year"] = profile_year
            st.session_state["student_cgpa"] = profile_cgpa
            st.session_state["student_skills"] = profile_skills
            st.session_state["target_career"] = profile_career
            st.session_state["profile_complete"] = True
            st.rerun()
    st.stop()

# Convenience variables used throughout the app
student_name = st.session_state["student_name"]
student_skills = st.session_state["student_skills"]

# ----------------------------------------------------------------------
# DAY 3 HELPER: read the current academic-input values from session
# state (with sensible defaults) so both the Academic Risk page and the
# Dashboard can show the SAME live prediction without duplicating code.
# This does not change any Day 1 or Day 2 logic - it only reads values
# that the sliders below already store via their `key=` arguments.
# ----------------------------------------------------------------------
def get_current_academic_risk():
    """
    Returns (predicted_label, probabilities, trained_result) using
    whatever academic input values currently live in session_state
    (or reasonable defaults on first run, before the sliders exist).
    """
    trained_result = get_trained_struggle_model()
    trained_model = trained_result["model"]

    attendance_val = st.session_state.get("academic_attendance", 75)
    assignment_val = st.session_state.get("academic_assignment", 75)
    quiz_val = st.session_state.get("academic_quiz", 70)
    previous_marks_val = st.session_state.get("academic_previous_marks", 70)
    study_hours_val = st.session_state.get("academic_study_hours", 10)

    predicted_label, probabilities = struggle_model.predict_risk(
        trained_model, attendance_val, assignment_val, quiz_val, previous_marks_val, study_hours_val
    )
    return predicted_label, probabilities, trained_result


def render_internal_application():
    render_app_header()
    active_page = st.session_state.get("active_page", "home")
    target_career = st.session_state.get("target_career", CAREER_NAMES[0])

    if active_page == "home":
        render_page_intro(
            "Home",
            f"Welcome back, {student_name} 👋",
            "Here's a quick look at your academic progress and career journey.",
        )
        predicted_label, _, _ = get_current_academic_risk()
        compat_df = calculate_compatibility(student_skills)
        target_score = float(compat_df.loc[compat_df["Career"] == target_career, "Compatibility (%)"].iloc[0])
        gap_df = build_skill_gap_table(target_career, student_skills)
        missing_count = int((gap_df["Have It?"] == "❌ No").sum())
        st.markdown('<div class="section-title">Your Academic Snapshot</div>', unsafe_allow_html=True)
        academic_cols = st.columns(3)
        with academic_cols[0]:
            render_metric_card("CGPA", f"{st.session_state['student_cgpa']:.1f}")
        with academic_cols[1]:
            render_metric_card("Year of Study", st.session_state["student_year"])
        with academic_cols[2]:
            render_metric_card("Academic Risk", predicted_label)
        st.markdown('<div class="section-title">Career Snapshot</div>', unsafe_allow_html=True)
        career_cols = st.columns(3)
        with career_cols[0]:
            render_metric_card("Target Career", target_career)
        with career_cols[1]:
            render_metric_card("Career Compatibility", f"{target_score:.1f}%")
        with career_cols[2]:
            render_metric_card("Skill Gaps", f"{missing_count} of {len(gap_df)}")
        st.markdown('<div class="section-title">Your Progress</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="soft-card"><h3>Your journey at a glance</h3>'
            '<p>Keep building one useful skill at a time. Your roadmap updates from your selected career goal.</p></div>',
            unsafe_allow_html=True,
        )
        progress_cols = st.columns(3)
        with progress_cols[0]:
            render_metric_card("Academic Risk", predicted_label)
        with progress_cols[1]:
            render_metric_card("Career Match", f"{target_score:.1f}%")
        with progress_cols[2]:
            render_metric_card("Skills Progress", f"{len(gap_df) - missing_count}/{len(gap_df)}")
        st.markdown('<div class="section-title">Quick Career Compatibility Overview</div>', unsafe_allow_html=True)
        for _, row in compat_df.iterrows():
            render_compatibility_card(row["Career"], row["Compatibility (%)"])

    elif active_page == "career":
        render_page_intro(
            "Career Navigator",
            "Explore career paths based on the skills you already have.",
            "Compare your current strengths with hand-curated career requirements using the existing cosine-similarity calculation.",
        )
        profile_cols = st.columns([1, 1.35])
        with profile_cols[0]:
            st.markdown(
                f'<div class="soft-card"><h3>Your Skill Profile</h3>'
                f'<p><strong>{student_name}</strong><br>{st.session_state["student_year"]} · CGPA {st.session_state["student_cgpa"]:.1f}<br>'
                f'{len(student_skills)} technical skills selected</p></div>',
                unsafe_allow_html=True,
            )
        with profile_cols[1]:
            st.markdown(
                f'<div class="soft-card"><h3>Current skills</h3><div>{render_skill_pills(student_skills)}</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown('<div class="section-title">Career Recommendations</div>', unsafe_allow_html=True)
        st.caption("Compatibility is a skill comparison, not a job-placement prediction.")
        compatibility_df = calculate_compatibility(student_skills)
        for _, row in compatibility_df.iterrows():
            render_compatibility_card(row["Career"], row["Compatibility (%)"])
        st.markdown('<div class="section-title">Choose a target career</div>', unsafe_allow_html=True)
        st.selectbox("Target career", options=CAREER_NAMES, key="target_career")
        gap_df = build_skill_gap_table(st.session_state["target_career"], student_skills)
        st.markdown('<div class="section-title">Skill gap analysis</div>', unsafe_allow_html=True)
        st.dataframe(gap_df, use_container_width=True, hide_index=True)

    elif active_page == "projects":
        render_project_recommendations(target_career, student_skills)

    elif active_page == "roadmap":
        render_page_intro(
            "Skill Roadmap",
            "Turn your career goal into a practical learning path.",
            "Your roadmap is generated from the current skills and target career you selected during onboarding.",
        )
        roadmap = build_learning_roadmap(target_career, student_skills)
        gap_names = [item["skill"] for item in roadmap[:4]]
        recommended_names = [item["skill"] for item in roadmap[:4]]
        project_names = [f"{item['skill']} project" for item in roadmap[:3]]
        st.markdown(
            '<div class="roadmap-flow">'
            + roadmap_stage_markup("CURRENT SKILLS", "What you already have", student_skills[:5], "current", "Add skills during onboarding to see your starting point.")
            + roadmap_stage_markup("IDENTIFIED SKILL GAPS", "What to strengthen", gap_names, "gap", "You have no identified gaps for this target.")
            + roadmap_stage_markup("RECOMMENDED LEARNING", "What to learn next", recommended_names, "recommended", "Your target career has no missing skills.")
            + roadmap_stage_markup("PRACTICAL PROJECTS", "How to apply it", project_names, "project", "Projects will appear as soon as there are learning steps.")
            + roadmap_stage_markup("TARGET CAREER", "Where you are heading", [target_career], "goal", "Choose a target career to begin.")
            + '</div>'
            '<div class="roadmap-legend">'
            '<span class="legend-item"><i class="legend-swatch"></i>Current skill</span>'
            '<span class="legend-item"><i class="legend-swatch gap"></i>Skill gap</span>'
            '<span class="legend-item"><i class="legend-swatch recommended"></i>Recommended</span>'
            '<span class="legend-item"><i class="legend-swatch project"></i>Project</span>'
            '<span class="legend-item"><i class="legend-swatch goal"></i>Career goal</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-title">Explore a recommended skill</div>', unsafe_allow_html=True)
        if roadmap:
            selected_skill = st.selectbox(
                "Recommended skill",
                options=[item["skill"] for item in roadmap],
                format_func=lambda skill: f"{skill} · {next(item['priority'] for item in roadmap if item['skill'] == skill)}",
            )
            selected_item = next(item for item in roadmap if item["skill"] == selected_skill)
            st.markdown(
                f'<div class="roadmap-detail"><h3>{selected_skill}</h3>'
                f'<p><strong>Why it is recommended:</strong> It is missing from your profile and is required for {target_career}.</p>'
                f'<p><strong>Priority:</strong> {selected_item["priority"]} · <strong>Importance:</strong> {selected_item["importance"]:.1f}</p>'
                f'<p><strong>Suggested topics:</strong> {recommended_topics(selected_skill)}</p>'
                f'<p><strong>Suggested application:</strong> {project_application(selected_skill, target_career)}</p></div>',
                unsafe_allow_html=True,
            )
        else:
            st.success("You already have all the required skills for this career.")
        st.markdown('<div class="section-title">Learning sequence</div>', unsafe_allow_html=True)
        if roadmap:
            for index, item in enumerate(roadmap, start=1):
                st.markdown(
                    f'<span class="pill">{index}. {item["skill"]}</span>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No additional learning steps are needed for the selected career.")

    elif active_page == "academic":
        render_page_intro(
            "Academic Risk",
            "Understand your current academic risk.",
            "Adjust the indicators below to identify areas that may need attention. This remains a learning-support estimate, not an official academic judgment.",
        )
        result = get_trained_struggle_model()
        trained_model = result["model"]
        input_cols = st.columns(2)
        with input_cols[0]:
            attendance_input = st.slider("Attendance (%)", 0, 100, 75, key="academic_attendance")
            assignment_input = st.slider("Assignment Completion (%)", 0, 100, 75, key="academic_assignment")
            quiz_input = st.slider("Quiz Average (%)", 0, 100, 70, key="academic_quiz")
        with input_cols[1]:
            previous_marks_input = st.slider("Previous Marks (%)", 0, 100, 70, key="academic_previous_marks")
            study_hours_input = st.slider("Study Hours per Week", 0, 40, 10, key="academic_study_hours")
        if st.button("Predict Academic Risk", type="primary"):
            predicted_label, probabilities = struggle_model.predict_risk(
                trained_model, attendance_input, assignment_input, quiz_input, previous_marks_input, study_hours_input
            )
            render_risk_card(predicted_label)
            st.markdown('<div class="section-title">Prediction confidence</div>', unsafe_allow_html=True)
            confidence_df = pd.DataFrame({"Risk Level": list(probabilities.keys()), "Probability": [round(value * 100, 1) for value in probabilities.values()]})
            for _, row in confidence_df.sort_values("Probability", ascending=False).iterrows():
                render_compatibility_card(row["Risk Level"], row["Probability"])
        st.markdown('<div class="section-title">Why this matters</div>', unsafe_allow_html=True)
        factor_cols = st.columns(3)
        for column, (feature_name, importance) in zip(factor_cols, struggle_model.get_top_factors(result["feature_importances"], top_n=3)):
            with column:
                render_metric_card(feature_name.replace("_", " "), f"{importance:.2f}")
        st.markdown('<div class="section-title">Model performance</div>', unsafe_allow_html=True)
        metric_cols = st.columns(4)
        for column, label, value in zip(metric_cols, ["Accuracy", "Precision", "Recall", "F1 Score"], [result["accuracy"], result["precision"], result["recall"], result["f1"]]):
            with column:
                render_metric_card(label, f"{value * 100:.1f}%")
        with st.expander("View detailed model tables"):
            st.dataframe(result["class_metrics"].round(3), use_container_width=True)
            st.dataframe(result["confusion_matrix"], use_container_width=True)

    elif active_page == "model":
        render_page_intro(
            "About the Model",
            "How the system works",
            "Two explainable techniques connect your student profile with career guidance and academic-risk insights.",
        )
        st.markdown(
            '<div class="workflow"><div class="workflow-step">Student Data</div><div class="workflow-step">Preprocessing</div>'
            '<div class="workflow-step">Machine Learning Model</div><div class="workflow-step">Academic Risk Prediction</div></div>',
            unsafe_allow_html=True,
        )
        info_cols = st.columns(2)
        with info_cols[0]:
            st.markdown('<div class="soft-card"><h3>Career matching</h3><p>Cosine Similarity compares your selected skill vector with weighted career requirements and returns a compatibility percentage.</p></div>', unsafe_allow_html=True)
        with info_cols[1]:
            st.markdown('<div class="soft-card"><h3>Academic risk</h3><p>A Random Forest Classifier uses attendance, assignment completion, quiz average, previous marks, and study hours to classify risk.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Evaluation results</div>', unsafe_allow_html=True)
        result = get_trained_struggle_model()
        metric_cols = st.columns(4)
        for column, label, value in zip(metric_cols, ["Accuracy", "Precision", "Recall", "F1 Score"], [result["accuracy"], result["precision"], result["recall"], result["f1"]]):
            with column:
                render_metric_card(label, f"{value * 100:.1f}%")
        with st.expander("View feature importance and confusion matrix"):
            st.dataframe(result["confusion_matrix"], use_container_width=True)
            importance_df = pd.DataFrame({"Feature": list(result["feature_importances"].keys()), "Importance": list(result["feature_importances"].values())}).sort_values("Importance", ascending=False)
            st.bar_chart(importance_df.set_index("Feature")["Importance"])


render_internal_application()
st.stop()


# ----------------------------------------------------------------------
# MAIN AREA - TOP-LEVEL NAVIGATION (Day 3: one integrated application)
# ----------------------------------------------------------------------
st.title("🎓 ML-Based Student Success & Career Navigator")
st.caption("An integrated Career Navigator + Academic Risk Prediction tool")

tab_dashboard, tab_career, tab_academic, tab_model_info = st.tabs(
    [
        "🏠 Dashboard",
        "🎯 Career Navigator",
        "📚 Academic Risk Prediction",
        "📊 Model Information",
    ]
)

# =========================================================================
# 🏠 DASHBOARD - one-glance summary pulling from BOTH modules
# =========================================================================
with tab_dashboard:
    st.subheader("Welcome" + (f", {student_name}!" if student_name else "!"))
    st.write(
        "This dashboard gives you a one-glance summary. Use the tabs above "
        "to explore career recommendations, skill gaps, your learning "
        "roadmap, and your academic risk prediction in detail."
    )

    st.markdown("#### Student Profile")
    p1, p2, p3 = st.columns(3)
    with p1:
        render_metric_card("Student Name", student_name)
    with p2:
        render_metric_card("Year", st.session_state["student_year"])
    with p3:
        render_metric_card("CGPA", f"{st.session_state['student_cgpa']:.1f}")

    st.markdown("#### Career Snapshot")
    target_career = st.session_state.get("target_career", CAREER_NAMES[0])

    if len(student_skills) == 0:
        c1, c2, c3 = st.columns(3)
        with c1:
            render_metric_card("Selected Career", target_career)
        with c2:
            render_metric_card("Career Compatibility", "N/A")
        with c3:
            render_metric_card("Skill Gaps", "N/A")
        st.info("Select at least one current skill in your profile to see career compatibility.")
    else:
        compat_df = calculate_compatibility(student_skills)
        target_row = compat_df[compat_df["Career"] == target_career].iloc[0]
        gap_df = build_skill_gap_table(target_career, student_skills)
        missing_count = int((gap_df["Have It?"] == "❌ No").sum())

        c1, c2, c3 = st.columns(3)
        with c1:
            render_metric_card("Selected Career", target_career)
        with c2:
            render_metric_card("Career Compatibility", f"{target_row['Compatibility (%)']}%")
        with c3:
            render_metric_card("Skill Gaps", f"{missing_count} of {len(gap_df)}")

        st.caption(
            "💡 Change the target career in the **🎯 Career Navigator** tab "
            "(Skill Gap Analysis section) to update this snapshot."
        )

    st.markdown("#### Academic Risk Snapshot")
    predicted_label, probabilities, _ = get_current_academic_risk()
    risk_display = struggle_model.RISK_EMOJI.get(predicted_label, predicted_label)

    render_risk_card(predicted_label)
    st.caption(f"Current model label: {risk_display}")

    st.caption(
        "💡 This uses the current values from the **📚 Academic Risk "
        "Prediction** tab (or defaults, if you haven't entered any yet)."
    )

    if len(student_skills) > 0:
        st.markdown("#### Quick Career Compatibility Overview")
        st.bar_chart(compat_df.set_index("Career")["Compatibility (%)"])

# =========================================================================
# 🎯 CAREER NAVIGATOR - Day 1 functionality, organized into clear steps
# =========================================================================
with tab_career:
    st.header("🎯 Career Navigator")

    # ---- 1. Student Skills ----------------------------------------------
    st.subheader("1. Student Skills")
    st.write("Here is a summary of the profile and skills you entered during onboarding:")

    profile_col1, profile_col2 = st.columns(2)
    with profile_col1:
        st.markdown(f"**Name:** {student_name if student_name else '_Not entered_'}")
        st.markdown(f"**Year of Study:** {st.session_state['student_year']}")
        st.markdown(f"**CGPA:** {st.session_state['student_cgpa']}")
    with profile_col2:
        st.markdown(f"**Total Skills Selected:** {len(student_skills)}")
        if student_skills:
            st.markdown("**Skills:**")
            st.write(", ".join(student_skills))
        else:
            st.markdown("_No skills selected yet._")

    st.markdown("---")

    # ---- 2. Career Recommendations ---------------------------------------
    st.subheader("2. Career Recommendations")
    st.caption(
        "⚠️ This is a **skill compatibility score** based on cosine "
        "similarity between your skills and each career's required "
        "skills. It is **not** a job-placement prediction."
    )

    if len(student_skills) == 0:
        st.warning("Please select at least one current skill in your profile to see recommendations.")
    else:
        compatibility_df = calculate_compatibility(student_skills)

        # Display sorted list with a simple progress bar per career
        for _, row in compatibility_df.iterrows():
            st.write(f"**{row['Career']}** — {row['Compatibility (%)']}%")
            st.progress(min(int(row["Compatibility (%)"]), 100))

        with st.expander("See raw compatibility table"):
            st.dataframe(compatibility_df, use_container_width=True)

    st.markdown("---")

    # ---- 3. Target Career -------------------------------------------------
    st.subheader("3. Target Career")
    st.write("Choose the career you want to work towards. This selection also drives the Dashboard, Skill Gap Analysis, and Learning Roadmap below.")

    target_career = st.selectbox(
        "Select a target career to analyze:",
        options=CAREER_NAMES,
        key="target_career",
    )

    st.markdown("---")

    # ---- 4. Skill Gaps -----------------------------------------------------
    st.subheader("4. Skill Gaps")
    st.write(f"Comparing your skills against the requirements for **{target_career}**:")

    gap_df = build_skill_gap_table(target_career, student_skills)
    st.dataframe(gap_df, use_container_width=True, hide_index=True)

    missing_count = (gap_df["Have It?"] == "❌ No").sum()
    st.info(f"You are missing **{missing_count}** out of **{len(gap_df)}** required skills for this career.")

    missing_df = gap_df[gap_df["Have It?"] == "❌ No"]
    if not missing_df.empty:
        with st.expander("See missing-skill importance chart"):
            st.bar_chart(missing_df.set_index("Required Skill")["Importance"])

    st.markdown("---")

    # ---- 5. Learning Roadmap ------------------------------------------------
    st.subheader("5. Learning Roadmap")
    st.write(
        f"Suggested learning order to become a **{target_career}**, "
        "based on missing skills (High priority first):"
    )

    roadmap = build_learning_roadmap(target_career, student_skills)

    if len(roadmap) == 0:
        st.success("🎉 You already have all the required skills for this career!")
    else:
        for i, item in enumerate(roadmap, start=1):
            st.write(f"{i}. **{item['skill']}** — {item['priority']} (Importance: {item['importance']})")

    st.caption(
        "This roadmap is generated using simple rule-based logic "
        "(sorted by priority, then importance) — no external AI API is used."
    )

# =========================================================================
# 📚 ACADEMIC RISK PREDICTION - Day 2 functionality, organized into steps
# =========================================================================
with tab_academic:
    st.header("📚 Academic Risk Prediction")
    st.caption(
        "⚠️ This predicts an **academic risk level** using a Random "
        "Forest model trained on a synthetic dataset. It is a "
        "learning-support estimate, not an official academic judgment."
    )

    # Train (or fetch cached) model + evaluation results
    result = get_trained_struggle_model()
    trained_model = result["model"]

    # ---- 1. Academic Inputs ------------------------------------------------
    st.subheader("1. Academic Inputs")

    col_a, col_b = st.columns(2)
    with col_a:
        attendance_input = st.slider(
            "Attendance (%)", min_value=0, max_value=100, value=75, key="academic_attendance"
        )
        assignment_input = st.slider(
            "Assignment Completion (%)", min_value=0, max_value=100, value=75, key="academic_assignment"
        )
        quiz_input = st.slider(
            "Quiz Average (%)", min_value=0, max_value=100, value=70, key="academic_quiz"
        )
    with col_b:
        previous_marks_input = st.slider(
            "Previous Marks (%)", min_value=0, max_value=100, value=70, key="academic_previous_marks"
        )
        study_hours_input = st.slider(
            "Study Hours per Week", min_value=0, max_value=40, value=10, key="academic_study_hours"
        )

    predict_clicked = st.button("🔮 Predict Academic Risk", type="primary")

    st.markdown("---")

    # ---- 2. Prediction -------------------------------------------------------
    st.subheader("2. Prediction")

    if predict_clicked:
        predicted_label, probabilities = struggle_model.predict_risk(
            trained_model,
            attendance_input,
            assignment_input,
            quiz_input,
            previous_marks_input,
            study_hours_input,
        )

        risk_display = struggle_model.RISK_EMOJI.get(predicted_label, predicted_label)
        render_risk_card(predicted_label)
        st.caption(f"Model output: {risk_display}")

        # Show model confidence for each class
        st.markdown("**Prediction Confidence**")
        proba_df = pd.DataFrame(
            {
                "Risk Level": list(probabilities.keys()),
                "Probability": [round(p * 100, 1) for p in probabilities.values()],
            }
        ).sort_values("Probability", ascending=False).reset_index(drop=True)
        st.bar_chart(proba_df.set_index("Risk Level")["Probability"])
        st.dataframe(proba_df, use_container_width=True, hide_index=True)
    else:
        st.info("👆 Enter the details above and click **Predict Academic Risk**.")

    st.markdown("---")

    # ---- 3. Important Factors --------------------------------------------------
    st.subheader("3. Important Factors")
    st.write(
        "These are the factors the Random Forest model relies on most, "
        "overall, based on its `feature_importances_`:"
    )
    top_factors = struggle_model.get_top_factors(result["feature_importances"], top_n=3)
    for i, (feature_name, importance) in enumerate(top_factors, start=1):
        readable_name = feature_name.replace("_", " ")
        st.write(f"{i}. **{readable_name}** (importance: {round(importance, 2)})")

    st.markdown("---")

    # ---- 4. Model Performance ------------------------------------------------
    st.subheader("4. Model Performance")
    st.write(
        "The Random Forest model is trained on a synthetic "
        "academic dataset (`student_data.csv`) containing 1,000 records and using a 75/25 "
        "train/test split. Scores below are measured on the test "
        "portion the model did not see during training."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{result['accuracy'] * 100:.1f}%")
    m2.metric("Precision", f"{result['precision'] * 100:.1f}%")
    m3.metric("Recall", f"{result['recall'] * 100:.1f}%")
    m4.metric("F1-Score", f"{result['f1'] * 100:.1f}%")

    sample_col, distribution_col = st.columns(2)
    with sample_col:
        st.markdown("**Training and testing samples**")
        st.write(f"Training samples: {result['train_samples']}")
        st.write(f"Testing samples: {result['test_samples']}")
    with distribution_col:
        st.markdown("**Class distribution**")
        distribution_df = pd.DataFrame(
            {
                "Risk Level": list(result["class_distribution"].keys()),
                "Records": list(result["class_distribution"].values()),
            }
        )
        st.dataframe(distribution_df, use_container_width=True, hide_index=True)

    st.markdown("**Per-class metrics**")
    st.dataframe(result["class_metrics"].round(3), use_container_width=True)

    st.markdown("**Confusion Matrix**")
    st.dataframe(result["confusion_matrix"], use_container_width=True)

    st.markdown("**Overall Feature Importance (from the Random Forest)**")
    importance_df = pd.DataFrame(
        {
            "Feature": list(result["feature_importances"].keys()),
            "Importance": list(result["feature_importances"].values()),
        }
    ).sort_values("Importance", ascending=False).reset_index(drop=True)
    st.bar_chart(importance_df.set_index("Feature")["Importance"])

# =========================================================================
# 📊 MODEL INFORMATION - plain-language explanation of both techniques
# =========================================================================
with tab_model_info:
    st.header("📊 Model Information")
    st.write(
        "This page explains, in simple terms, the two techniques used "
        "in this project and shows their evaluation results."
    )

    st.subheader("🎯 Career Matching — Cosine Similarity")
    st.write(
        "Cosine Similarity compares the student's skill vector with the "
        "skill requirements of different careers. Your selected skills "
        "become a list of 1s and 0s, while each career's requirements "
        "become a list of importance weights from 0 to 1. The closer "
        "the two vectors point in the same direction, the higher the "
        "compatibility percentage. This is a mathematical comparison, "
        "not a trained model, so it does not need training data."
    )

    st.markdown("---")

    st.subheader("📚 Academic Risk Prediction — Random Forest Classifier")
    st.write(
        "Random Forest is used to classify students into Low Risk, "
        "Medium Risk, or High Risk. It is made up of many decision trees. "
        "Each tree examines the academic inputs and votes for a risk "
        "level; the forest combines those votes into the final prediction."
    )
    st.markdown("**The five academic inputs are:**")
    st.markdown(
        "- Attendance (%)\n"
        "- Assignment Completion (%)\n"
        "- Quiz Average (%)\n"
        "- Previous Marks (%)\n"
        "- Study Hours per Week"
    )
    st.write(
        "Feature importance shows how much each input contributed to the "
        "forest's decisions overall. It helps explain the model, but it "
        "does not prove that one factor directly caused a student's risk."
    )

    st.markdown("**Evaluation results (on held-out test data):**")
    academic_result = get_trained_struggle_model()
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Accuracy", f"{academic_result['accuracy'] * 100:.1f}%")
    e2.metric("Precision", f"{academic_result['precision'] * 100:.1f}%")
    e3.metric("Recall", f"{academic_result['recall'] * 100:.1f}%")
    e4.metric("F1-Score", f"{academic_result['f1'] * 100:.1f}%")

    with st.expander("See confusion matrix"):
        st.dataframe(academic_result["confusion_matrix"], use_container_width=True)

    with st.expander("See feature importance chart"):
        importance_df = pd.DataFrame(
            {
                "Feature": list(academic_result["feature_importances"].keys()),
                "Importance": list(academic_result["feature_importances"].values()),
            }
        ).sort_values("Importance", ascending=False).reset_index(drop=True)
        st.bar_chart(importance_df.set_index("Feature")["Importance"])

    st.caption(
        "Both techniques are intentionally simple and explainable, "
        "matching the scope of this B.Tech project — no deep learning, "
        "SHAP, or external AI APIs are used anywhere."
    )
