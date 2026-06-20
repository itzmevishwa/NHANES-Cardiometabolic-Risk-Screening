"""
═══════════════════════════════════════════════════════════════════════════
  CARDIO·SCAN — Cardiometabolic Risk Screening
  A premium screening tool powered by leakage-free ML models (NHANES 2021-23)
═══════════════════════════════════════════════════════════════════════════
  Run with:  streamlit run app.py
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="CardioScan — Risk Screening",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── variables ── */
:root {
    --teal:   #5EEAD4;
    --indigo: #818CF8;
    --purple: #C084FC;
    --green:  #22C55E;
    --amber:  #F59E0B;
    --red:    #EF4444;
}

/* ── keyframes ── */
@keyframes fadeInUp {
    from { opacity:0; transform:translateY(28px); }
    to   { opacity:1; transform:translateY(0); }
}
@keyframes fadeInScale {
    from { opacity:0; transform:scale(0.93); }
    to   { opacity:1; transform:scale(1); }
}
@keyframes slideInLeft {
    from { opacity:0; transform:translateX(-22px); }
    to   { opacity:1; transform:translateX(0); }
}
@keyframes shimmerText {
    0%   { background-position:-400% center; }
    100% { background-position: 400% center; }
}
@keyframes heartbeat {
    0%,100% { transform:scale(1); }
    14%     { transform:scale(1.20); }
    28%     { transform:scale(1); }
    42%     { transform:scale(1.12); }
    70%     { transform:scale(1); }
}
@keyframes scanLine {
    0%   { transform:translateY(-100%); opacity:.7; }
    100% { transform:translateY(3000%); opacity:0; }
}
@keyframes pillFloat {
    0%,100% { transform:translateY(0); }
    50%     { transform:translateY(-5px); }
}
@keyframes gradientShift {
    0%,100% { background-position:0% 50%; }
    50%     { background-position:100% 50%; }
}
@keyframes resultCardIn {
    from { opacity:0; transform:translateY(22px) scale(.97); }
    to   { opacity:1; transform:translateY(0) scale(1); }
}
@keyframes badgePop {
    0%  { transform:scale(.65); opacity:0; }
    70% { transform:scale(1.10); }
    100%{ transform:scale(1);   opacity:1; }
}
@keyframes scaleInX {
    from { transform:scaleX(0); }
    to   { transform:scaleX(1); }
}
@keyframes glimmerSlide {
    0%   { left:-100%; }
    100% { left: 200%; }
}
@keyframes pulseDot {
    0%,100% { opacity:1;  transform:scale(1); }
    50%     { opacity:.4; transform:scale(1.6); }
}

/* ── base ── */
.stApp {
    background:
        radial-gradient(ellipse 900px 500px at 5% -5%,  rgba(94,234,212,.09),  transparent 55%),
        radial-gradient(ellipse 800px 600px at 95% 5%,  rgba(99,102,241,.11),  transparent 55%),
        radial-gradient(ellipse 600px 400px at 50% 95%, rgba(192,132,252,.07), transparent 50%),
        linear-gradient(175deg, #060A13 0%, #080D1A 50%, #0A0F1C 100%);
    color: #E6EDF5;
    font-family: 'Space Grotesk', sans-serif;
}
#MainMenu, footer, header { visibility:hidden; }
.block-container { padding-top:1rem; padding-bottom:3rem; max-width:1180px; }

/* ── hero ── */
.hero-wrapper {
    text-align:center;
    padding:2.4rem 0 1.5rem;
    position:relative;
    overflow:hidden;
    animation:fadeInUp .9s ease both;
}
.hero-scan-line {
    position:absolute; left:0; right:0; top:0;
    height:2px;
    background:linear-gradient(90deg, transparent 0%, rgba(94,234,212,.65) 40%, rgba(129,140,248,.65) 60%, transparent 100%);
    animation:scanLine 5s ease-in-out 1s infinite;
}
.hero-icon {
    font-size:3.4rem;
    display:block;
    margin-bottom:.5rem;
    animation:heartbeat 1s ease-in-out infinite;
    filter:drop-shadow(0 0 18px rgba(94,234,212,.6));
}
.hero-title {
    font-family:'Sora',sans-serif;
    font-size:3.9rem;
    font-weight:800;
    letter-spacing:-0.04em;
    margin:0;
    background:linear-gradient(135deg, #5EEAD4 0%, #818CF8 40%, #C084FC 70%, #5EEAD4 100%);
    background-size:300% auto;
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    background-clip:text;
    animation:shimmerText 7s linear infinite;
}
.hero-sub {
    color:#8A98AD;
    font-size:1.05rem;
    margin-top:.65rem;
    animation:fadeInUp 1s ease .3s both;
}
.hero-pills {
    margin-top:1.1rem;
    display:flex;
    justify-content:center;
    gap:.65rem;
    flex-wrap:wrap;
    animation:fadeInUp 1s ease .55s both;
}
.pill {
    display:inline-flex; align-items:center; gap:.35rem;
    padding:.38rem 1rem;
    border:1px solid rgba(94,234,212,.3);
    border-radius:999px;
    background:rgba(94,234,212,.07);
    color:#5EEAD4;
    font-size:.76rem; letter-spacing:.08em; text-transform:uppercase; font-weight:600;
    animation:pillFloat 3.5s ease-in-out infinite;
}
.pill-indigo { border-color:rgba(129,140,248,.3); background:rgba(129,140,248,.07); color:#818CF8; animation-delay:.6s; }
.pill-purple { border-color:rgba(192,132,252,.3); background:rgba(192,132,252,.07); color:#C084FC; animation-delay:1.2s; }

/* ── section labels ── */
.section-label {
    font-family:'Sora',sans-serif;
    font-size:.74rem; font-weight:700;
    letter-spacing:.2em; text-transform:uppercase;
    color:#5EEAD4;
    margin:2rem 0 .9rem;
    display:flex; align-items:center; gap:.75rem;
    animation:slideInLeft .6s ease both;
}
.step-num {
    display:inline-flex; align-items:center; justify-content:center;
    width:1.9rem; height:1.9rem; border-radius:50%;
    background:linear-gradient(135deg, rgba(94,234,212,.2), rgba(99,102,241,.2));
    border:1px solid rgba(94,234,212,.38);
    font-size:.7rem; font-weight:800; color:#5EEAD4; flex-shrink:0;
}
.section-label::after {
    content:""; flex:1; height:1px;
    background:linear-gradient(90deg, rgba(94,234,212,.35), transparent);
}

/* ── glass card ── */
.glass {
    background:rgba(255,255,255,.035);
    border:1px solid rgba(255,255,255,.09);
    border-radius:20px; padding:1.6rem 1.8rem;
    backdrop-filter:blur(14px);
    transition:border-color .3s, box-shadow .3s;
    animation:fadeInScale .7s ease both;
}
.glass:hover { border-color:rgba(94,234,212,.18); box-shadow:0 8px 40px rgba(94,234,212,.06); }

/* ── input group headers ── */
.input-group-label {
    font-family:'Sora',sans-serif;
    font-size:.78rem; font-weight:700;
    letter-spacing:.12em; text-transform:uppercase;
    color:#8A98AD;
    padding:.6rem 0 .35rem;
    display:flex; align-items:center; gap:.5rem;
    border-bottom:1px solid rgba(255,255,255,.05);
    margin-bottom:.4rem;
}

/* ── inputs ── */
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div {
    background:rgba(255,255,255,.05) !important;
    border:1px solid rgba(255,255,255,.11) !important;
    border-radius:12px !important;
    color:#E6EDF5 !important;
    transition:border-color .25s, box-shadow .25s !important;
}
.stNumberInput input:focus {
    border-color:rgba(94,234,212,.5) !important;
    box-shadow:0 0 0 3px rgba(94,234,212,.08) !important;
}
.stSlider label, .stNumberInput label, .stSelectbox label {
    color:#B5C0D0 !important; font-size:.84rem !important; font-weight:500 !important;
}

/* ── bmi chip ── */
.bmi-chip {
    display:inline-flex; align-items:center; gap:.5rem;
    padding:.42rem 1.1rem;
    background:linear-gradient(135deg, rgba(94,234,212,.13), rgba(99,102,241,.13));
    border:1px solid rgba(94,234,212,.27);
    border-radius:999px;
    font-size:.88rem; font-weight:600; color:#5EEAD4;
    font-family:'Sora',sans-serif;
    animation:fadeInScale .4s ease both;
    margin-top:.35rem;
}

/* ── checkboxes ── */
.stCheckbox {
    background:rgba(255,255,255,.03);
    border:1px solid rgba(255,255,255,.08);
    border-radius:14px; padding:.75rem 1rem;
    transition:all .25s ease;
    animation:fadeInUp .5s ease both;
}
.stCheckbox:hover {
    border-color:rgba(94,234,212,.42);
    background:rgba(94,234,212,.04);
    transform:translateY(-2px);
    box-shadow:0 6px 22px rgba(94,234,212,.09);
}

/* ── button ── */
.stButton > button {
    width:100%;
    background:linear-gradient(135deg, #5EEAD4 0%, #6366F1 50%, #A855F7 100%);
    background-size:200% auto;
    color:#05111A;
    font-family:'Sora',sans-serif;
    font-weight:700; font-size:1.08rem; letter-spacing:.04em;
    border:none; border-radius:16px; padding:.98rem 1rem;
    transition:transform .3s ease, box-shadow .3s ease, filter .3s ease;
    box-shadow:0 8px 32px rgba(94,234,212,.22), 0 2px 8px rgba(0,0,0,.3);
    animation:gradientShift 5s ease infinite;
}
.stButton > button:hover {
    transform:translateY(-3px);
    box-shadow:0 18px 50px rgba(99,102,241,.42), 0 4px 14px rgba(0,0,0,.4);
    filter:brightness(1.08);
    color:#05111A;
}
.stButton > button:active { transform:translateY(-1px); }

/* ── result card ── */
.result-card {
    border-radius:22px; padding:1.5rem 1.6rem;
    border:1px solid rgba(255,255,255,.08);
    background:rgba(255,255,255,.035);
    backdrop-filter:blur(16px);
    position:relative; overflow:hidden;
    animation:resultCardIn .65s cubic-bezier(.34,1.56,.64,1) both;
    animation-delay:var(--delay, 0s);
}
.rc-accent {
    position:absolute; top:0; left:0; right:0; height:2px;
    border-radius:22px 22px 0 0;
}
.rc-glimmer {
    position:absolute; top:0; bottom:0; left:-100%; width:55%;
    background:linear-gradient(105deg, transparent, rgba(255,255,255,.045), transparent);
    animation:glimmerSlide 2.4s ease .8s both;
    pointer-events:none;
}
.risk-name { font-family:'Sora',sans-serif; font-size:1.28rem; font-weight:700; margin:0; }
.risk-sub  { color:#8A98AD; font-size:.85rem; margin:.12rem 0 0; }
.risk-score {
    font-family:'Sora',sans-serif; font-size:3rem; font-weight:800; line-height:1;
    margin-top:.7rem;
    animation:fadeInScale .5s cubic-bezier(.34,1.56,.64,1) both;
    animation-delay:calc(var(--delay, 0s) + .15s);
}
.risk-bar-wrap {
    margin-top:.65rem;
    background:rgba(255,255,255,.06);
    border-radius:999px; height:5px; overflow:hidden;
}
.risk-bar {
    height:100%; border-radius:999px;
    transform-origin:left center;
    animation:scaleInX 1.3s cubic-bezier(.22,1,.36,1) both;
    animation-delay:calc(var(--delay, 0s) + .35s);
}
.badge {
    display:inline-flex; align-items:center; gap:.35rem;
    padding:.32rem .88rem; border-radius:999px;
    font-size:.77rem; font-weight:700; letter-spacing:.06em;
    animation:badgePop .5s cubic-bezier(.34,1.56,.64,1) both;
    animation-delay:calc(var(--delay, 0s) + .1s);
}
.badge-dot {
    width:6px; height:6px; border-radius:50%;
    animation:pulseDot 1.8s ease-in-out infinite;
}
.advice {
    margin-top:1rem; padding:.85rem 1.1rem;
    border-radius:14px; background:rgba(255,255,255,.04);
    border-left:3px solid;
    font-size:.87rem; color:#C4CFDD; line-height:1.55;
    animation:slideInLeft .5s ease both;
    animation-delay:calc(var(--delay, 0s) + .4s);
}

/* ── limitations ── */
.lim-glass {
    background:rgba(255,255,255,.028);
    border:1px solid rgba(255,255,255,.08);
    border-radius:20px; padding:1.6rem 1.8rem;
    animation:fadeInUp .6s ease both;
}
.lim-glass li {
    padding:.48rem 0;
    border-bottom:1px solid rgba(255,255,255,.04);
    font-size:.9rem; color:#B5C0D0; line-height:1.6;
}
.lim-glass li:last-child { border-bottom:none; }

/* ── footer ── */
.footer-wrap { padding:2rem 0 .5rem; text-align:center; animation:fadeInUp .7s ease both; }
.footer-line {
    height:1px;
    background:linear-gradient(90deg, transparent, rgba(94,234,212,.4), rgba(129,140,248,.4), transparent);
    margin-bottom:1.6rem;
}
.footer-name {
    font-family:'Sora',sans-serif; font-size:1.1rem; font-weight:700;
    background:linear-gradient(135deg, #5EEAD4, #818CF8);
    -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
}
.footer-role { color:#8A98AD; font-size:.85rem; margin-top:.3rem; }
.footer-links { margin-top:.9rem; display:flex; justify-content:center; gap:1rem; }
.footer-links a {
    color:#5EEAD4; text-decoration:none; font-size:.84rem; font-weight:500;
    padding:.32rem .9rem;
    border:1px solid rgba(94,234,212,.22); border-radius:999px;
    transition:all .25s ease;
}
.footer-links a:hover {
    background:rgba(94,234,212,.09);
    border-color:rgba(94,234,212,.5);
    transform:translateY(-2px);
    box-shadow:0 4px 14px rgba(94,234,212,.12);
}
.disclaimer {
    text-align:center; color:#3E4A5C; font-size:.78rem;
    margin-top:1.5rem; line-height:1.7; padding:0 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────
DISEASES = {
    "diabetes":     {"name": "Diabetes",            "icon": "🩸", "test": "HbA1c blood test"},
    "cvd":          {"name": "Heart Disease (CVD)",  "icon": "🫀", "test": "lipid panel + ECG"},
    "hypertension": {"name": "Hypertension",         "icon": "💓", "test": "clinical BP monitoring"},
    "metabolic":    {"name": "Metabolic Syndrome",   "icon": "⚡", "test": "fasting glucose + lipids"},
}

@st.cache_resource
def load_models():
    models = {}
    for key in DISEASES:
        try:
            models[key] = {
                "model":    joblib.load(f"models/{key}_model.pkl"),
                "features": joblib.load(f"models/{key}_features.pkl"),
            }
            try:
                models[key]["scaler"] = joblib.load(f"models/{key}_scaler.pkl")
            except Exception:
                models[key]["scaler"] = None
        except Exception:
            models[key] = None
    return models

models = load_models()

# ─────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-scan-line"></div>
    <span class="hero-icon">❤️</span>
    <h1 class="hero-title">CardioScan</h1>
    <div class="hero-sub">Early cardiometabolic risk screening — from simple, low-cost measurements</div>
    <div class="hero-pills">
        <span class="pill">⚡ Leakage-free ML</span>
        <span class="pill pill-indigo">📊 NHANES 2021–2023</span>
        <span class="pill pill-purple">🩺 4 Conditions</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# STEP 1 — SELECT CONDITIONS
# ─────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label"><span class="step-num">01</span>Choose conditions to screen for</div>',
            unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
selected = {}
with c1: selected["diabetes"]     = st.checkbox("🩸  Diabetes", value=True)
with c2: selected["cvd"]          = st.checkbox("🫀  Heart Disease")
with c3: selected["hypertension"] = st.checkbox("💓  Hypertension")
with c4: selected["metabolic"]    = st.checkbox("⚡  Metabolic Syndrome")

chosen = [k for k, v in selected.items() if v]

# ─────────────────────────────────────────────────────────────────────────
# STEP 2 — INPUTS
# ─────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label"><span class="step-num">02</span>Enter patient information</div>',
            unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="input-group-label"><span>👤</span> Demographics</div>', unsafe_allow_html=True)
    d1, d2, d3, d4, d5 = st.columns(5)
    age    = d1.number_input("Age (years)", 18, 100, 45)
    gender = d2.selectbox("Gender", ["Male", "Female"])
    race   = d3.selectbox("Race/Ethnicity", ["Mexican American","Other Hispanic","White","Black","Asian","Other"])
    income = d4.number_input("Income ratio (0–5)", 0.0, 5.0, 2.5, 0.1)
    educ   = d5.selectbox("Education", ["< 9th grade","9-11th grade","High school","Some college","College grad"])

    st.markdown('<div class="input-group-label" style="margin-top:.9rem;"><span>📏</span> Body Measurements</div>',
                unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    weight = b1.number_input("Weight (kg)", 30.0, 250.0, 75.0, 0.5)
    height = b2.number_input("Height (cm)", 120.0, 220.0, 168.0, 0.5)
    waist  = b3.number_input("Waist (cm)", 50.0, 200.0, 92.0, 0.5)
    bmi = round(weight / ((height / 100) ** 2), 1)
    st.markdown(f'<div class="bmi-chip">⚖️ &nbsp;Calculated BMI &nbsp;<b style="font-size:1rem">{bmi}</b></div>',
                unsafe_allow_html=True)

    st.markdown('<div class="input-group-label" style="margin-top:.9rem;"><span>💓</span> Blood Pressure</div>',
                unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    systolic  = p1.number_input("Systolic (mmHg)", 80, 220, 122)
    diastolic = p2.number_input("Diastolic (mmHg)", 40, 140, 78)
    pulse     = p3.number_input("Pulse (bpm)", 40, 140, 72)

    st.markdown('<div class="input-group-label" style="margin-top:.9rem;"><span>🌿</span> Lifestyle</div>',
                unsafe_allow_html=True)
    l1, l2, l3, l4, l5 = st.columns(5)
    sleep      = l1.number_input("Sleep (hrs)", 2.0, 14.0, 7.0, 0.5)
    activity   = l2.number_input("Activity (min/day)", 0, 600, 30)
    smoke      = l3.selectbox("Smoking", ["Never","Former","Current"])
    alcohol    = l4.number_input("Alcohol (drinks/day)", 0, 15, 1)
    depression = l5.number_input("Depression score (0–27)", 0, 27, 2)

# map categorical → numeric
gender_code = 1 if gender == "Male" else 2
race_code   = {"Mexican American":1,"Other Hispanic":2,"White":3,"Black":4,"Asian":6,"Other":7}[race]
educ_code   = {"< 9th grade":1,"9-11th grade":2,"High school":3,"Some college":4,"College grad":5}[educ]
smoke_code  = {"Never":0,"Former":1,"Current":2}[smoke]

all_inputs = {
    "RIDAGEYR": age,        "RIAGENDR": gender_code, "RIDRETH3": race_code,
    "INDFMPIR": income,     "DMDEDUC2": educ_code,
    "BMXWT": weight,        "BMXHT": height,         "BMXWAIST": waist,  "BMXBMI": bmi,
    "systolic_avg": systolic, "diastolic_avg": diastolic, "pulse_avg": pulse,
    "avg_sleep_hours": sleep, "depression_score": depression,
    "total_activity": activity, "smoking_status": smoke_code, "alcohol_drinks": alcohol,
}

# ─────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────
RISK_THRESHOLDS = {
    "diabetes":     {"moderate": 0.25, "high": 0.45},
    "cvd":          {"moderate": 0.15, "high": 0.30},
    "hypertension": {"moderate": 0.50, "high": 0.70},
    "metabolic":    {"moderate": 0.30, "high": 0.50},
}

def risk_level(p, model_name):
    t = RISK_THRESHOLDS[model_name]

    if p < t["moderate"]:
        return "LOW", "#22C55E", "rgba(34,197,94,0.12)"

    if p < t["high"]:
        return "MODERATE", "#F59E0B", "rgba(245,158,11,0.12)"

    return "HIGH", "#EF4444", "rgba(239,68,68,0.12)"

def gauge(p, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=p * 100,
        number={"suffix": "%", "font": {"size": 30, "color": "#E6EDF5", "family": "Sora"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#3A4555", "tickfont": {"color": "#5C6677", "size": 9}},
            "bar":  {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  33], "color": "rgba(34,197,94,0.10)"},
                {"range": [33, 66], "color": "rgba(245,158,11,0.10)"},
                {"range": [66,100], "color": "rgba(239,68,68,0.10)"},
            ],
        },
    ))
    fig.update_layout(
        height=200, margin=dict(l=15, r=15, t=20, b=5),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#E6EDF5"},
    )
    return fig

def predict(key):
    m = models[key]
    feats = m["features"]
    row = pd.DataFrame([{f: all_inputs.get(f, 0) for f in feats}])
    if m.get("scaler") is not None:
        row_in = m["scaler"].transform(row)
        return float(m["model"].predict_proba(row_in)[0][1])
    # tree models: pass numpy to avoid feature-name mismatch warnings
    return float(m["model"].predict_proba(row.values)[0][1])

# ─────────────────────────────────────────────────────────────────────────
# STEP 3 — RUN
# ─────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label"><span class="step-num">03</span>Run screening</div>',
            unsafe_allow_html=True)

_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    run = st.button("⚡  Run Screening")

if run:
    if not chosen:
        st.warning("Please select at least one condition to screen for.")
    elif any(models[k] is None for k in chosen):
        st.error("Models not found. Ensure the 'models/' folder is next to app.py.")
    else:
        st.markdown('<div class="section-label">Results · Risk assessment</div>', unsafe_allow_html=True)

        with st.spinner("Analysing risk factors…"):
            probs = {k: predict(k) for k in chosen}

        # overview bar
        bar = go.Figure(go.Bar(
            x=[probs[k] * 100 for k in chosen],
            y=[DISEASES[k]["name"] for k in chosen],
            orientation="h",
            marker=dict(
               color=[risk_level(probs[k], k)[1] for k in chosen],
                line=dict(width=0),
            ),
            text=[f"{probs[k]*100:.0f}%" for k in chosen],
            textposition="outside",
            textfont=dict(color="#E6EDF5", family="Sora", size=14),
        ))
        bar.update_layout(
            height=90 + 60 * len(chosen),
            margin=dict(l=10, r=55, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                range=[0, 115], showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                tickfont=dict(color="#5C6677"),
                title=dict(text="Risk %", font=dict(color="#5C6677")),
            ),
            yaxis=dict(tickfont=dict(color="#E6EDF5", size=13)),
        )
        st.plotly_chart(bar, use_container_width=True, config={"displayModeBar": False})

        # detailed cards
        for i, k in enumerate(chosen):
            p = probs[k]
            level, color, bg = risk_level(p, k)
            info  = DISEASES[k]
            delay = f"{i * 0.12:.2f}s"

            advice_lead = "<b>Recommend confirmatory testing.</b>" if level != "LOW" else "<b>Low risk detected.</b>"
            advice_body = (
                f"This patient should be referred for a <b>{info['test']}</b> to confirm."
                if level != "LOW"
                else "Routine monitoring advised; no urgent testing indicated by this screen."
            )
            advice_icon = "⚠️" if level != "LOW" else "✅"

            colA, colB = st.columns([1, 1.4])
            with colA:
                st.plotly_chart(gauge(p, color), use_container_width=True,
                                config={"displayModeBar": False})
            with colB:
                st.markdown(f"""
                <div class="result-card" style="border-color:{color}30; --delay:{delay};">
                    <div class="rc-accent" style="background:linear-gradient(90deg,{color},{color}77);"></div>
                    <div class="rc-glimmer"></div>
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                        <div>
                            <p class="risk-name">{info['icon']}  {info['name']}</p>
                            <p class="risk-sub">Estimated screening risk</p>
                        </div>
                        <span class="badge" style="background:{bg};color:{color};">
                            <span class="badge-dot" style="background:{color};"></span>
                            {level} RISK
                        </span>
                    </div>
                    <div class="risk-score" style="color:{color};">{p*100:.1f}%</div>
                    <div class="risk-bar-wrap">
                        <div class="risk-bar" style="width:{p*100:.1f}%;background:linear-gradient(90deg,{color}bb,{color});"></div>
                    </div>
                    <div class="advice" style="border-color:{color}66;">
                        {advice_icon} {advice_lead} {advice_body}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# LIMITATIONS
# ─────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">About · Limitations &amp; honest notes</div>', unsafe_allow_html=True)

st.markdown("""
<div class="lim-glass">
    <p style="margin:0 0 .9rem;color:#E6EDF5;font-weight:600;font-family:'Sora',sans-serif;font-size:1rem;">
        This is a screening aid, not a diagnostic tool.
    </p>
    <ul style="margin:0;padding-left:1.1rem;">
        <li><b style="color:#E6EDF5;">Screening, not diagnosis.</b>
            The model flags who is <i>likely</i> at risk from low-cost inputs.
            A positive screen should be confirmed with the proper clinical lab test.</li>
        <li><b style="color:#E6EDF5;">Honest accuracy (AUC 0.80–0.82).</b>
            Realistic screening scores — not inflated. An earlier version had data leakage;
            this version was rebuilt leakage-free.</li>
        <li><b style="color:#E6EDF5;">No family medical history.</b>
            Family history was dropped from the NHANES 2021–2023 public release and
            could not be included.</li>
        <li><b style="color:#E6EDF5;">Single population &amp; cycle.</b>
            Trained only on U.S. NHANES 2021–2023 adults. Performance on other
            populations is not guaranteed.</li>
        <li><b style="color:#E6EDF5;">Self-reported data.</b>
            Diagnoses and lifestyle answers come from questionnaires and carry
            inherent reporting bias.</li>
        <li><b style="color:#E6EDF5;">Metabolic syndrome.</b>
            Trained only on fasting participants, so its sample is smaller than
            the other models.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="footer-wrap">
    <div class="footer-line"></div>
    <div class="footer-name">Designed, built &amp; trained by Vishwa A</div>
    <div class="footer-role">Data &amp; ML Engineer · Leakage-free ML pipeline · NHANES 2021–2023</div>
    <div class="footer-links">
        <a href="https://github.com/itzmevishwa">GitHub</a>
        <a href="https://linkedin.com/in/vishwa444">LinkedIn</a>
    </div>
</div>
<div class="disclaimer">
    CardioScan is an educational screening aid built on CDC NHANES 2021–2023 data
    using leakage-free machine learning.<br>
    It does not provide a diagnosis and is not a substitute for professional medical judgment.
</div>
""", unsafe_allow_html=True)
