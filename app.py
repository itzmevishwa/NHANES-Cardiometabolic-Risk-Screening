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

# ─────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioScan — Risk Screening",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────
# PREMIUM CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ---- base ---- */
.stApp {
    background:
        radial-gradient(1200px 600px at 10% -10%, rgba(45,212,191,0.10), transparent 60%),
        radial-gradient(1000px 700px at 110% 10%, rgba(99,102,241,0.12), transparent 55%),
        linear-gradient(180deg, #070B14 0%, #0A0F1C 100%);
    color: #E6EDF5;
    font-family: 'Space Grotesk', sans-serif;
}
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1180px;}

/* ---- hero ---- */
.hero {
    text-align: center;
    padding: 1.2rem 0 0.4rem 0;
}
.hero h1 {
    font-family: 'Sora', sans-serif;
    font-size: 3.4rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin: 0;
    background: linear-gradient(135deg, #5EEAD4 0%, #818CF8 55%, #C084FC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero .sub {
    color: #8A98AD;
    font-size: 1.05rem;
    margin-top: 0.4rem;
    letter-spacing: 0.02em;
}
.hero .pill {
    display: inline-block;
    margin-top: 0.9rem;
    padding: 0.35rem 0.9rem;
    border: 1px solid rgba(94,234,212,0.3);
    border-radius: 999px;
    background: rgba(94,234,212,0.06);
    color: #5EEAD4;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ---- section labels ---- */
.section-label {
    font-family: 'Sora', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #5EEAD4;
    margin: 1.6rem 0 0.6rem 0;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-label::after {
    content: "";
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(94,234,212,0.4), transparent);
}

/* ---- cards ---- */
.glass {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 1.4rem 1.6rem;
    backdrop-filter: blur(12px);
}

/* ---- inputs ---- */
.stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 12px !important;
    color: #E6EDF5 !important;
}
.stSlider label, .stNumberInput label, .stSelectbox label {
    color: #B5C0D0 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* ---- buttons ---- */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #5EEAD4 0%, #6366F1 100%);
    color: #06121A;
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    letter-spacing: 0.02em;
    border: none;
    border-radius: 14px;
    padding: 0.85rem 1rem;
    transition: all 0.25s ease;
    box-shadow: 0 8px 30px rgba(94,234,212,0.25);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(99,102,241,0.45);
    color: #06121A;
}

/* ---- disease select tiles ---- */
.stCheckbox {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 0.7rem 0.9rem;
    transition: all 0.2s ease;
}
.stCheckbox:hover { border-color: rgba(94,234,212,0.4); }

/* ---- result cards ---- */
.result-card {
    border-radius: 20px;
    padding: 1.4rem 1.5rem;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.035);
    margin-bottom: 0.4rem;
}
.risk-name {
    font-family: 'Sora', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    margin: 0;
}
.risk-sub { color: #8A98AD; font-size: 0.85rem; margin: 0.1rem 0 0 0; }
.risk-score {
    font-family: 'Sora', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1;
}
.badge {
    display: inline-block;
    padding: 0.3rem 0.85rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.04em;
}
.advice {
    margin-top: 0.9rem;
    padding: 0.8rem 1rem;
    border-radius: 12px;
    background: rgba(255,255,255,0.04);
    font-size: 0.9rem;
    color: #C4CFDD;
    line-height: 1.5;
}
.disclaimer {
    text-align: center;
    color: #5C6677;
    font-size: 0.8rem;
    margin-top: 2.5rem;
    line-height: 1.6;
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
                "model": joblib.load(f"models/{key}_model.pkl"),
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
<div class="hero">
    <h1>CardioScan</h1>
    <div class="sub">Early cardiometabolic risk screening — from simple, low-cost measurements</div>
    <div class="pill">Leakage-free ML · NHANES 2021–2023</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# STEP 1 — SELECT DISEASES
# ─────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Step 1 · Choose what to screen for</div>', unsafe_allow_html=True)

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
st.markdown('<div class="section-label">Step 2 · Enter patient information</div>', unsafe_allow_html=True)

with st.container():
    st.markdown("**👤  Demographics**")
    d1, d2, d3, d4, d5 = st.columns(5)
    age    = d1.number_input("Age (years)", 18, 100, 45)
    gender = d2.selectbox("Gender", ["Male", "Female"])
    race   = d3.selectbox("Race/Ethnicity", ["Mexican American","Other Hispanic","White","Black","Asian","Other"])
    income = d4.number_input("Income ratio (0–5)", 0.0, 5.0, 2.5, 0.1)
    educ   = d5.selectbox("Education", ["< 9th grade","9-11th grade","High school","Some college","College grad"])

    st.markdown("**📏  Body Measurements**")
    b1, b2, b3 = st.columns(3)
    weight = b1.number_input("Weight (kg)", 30.0, 250.0, 75.0, 0.5)
    height = b2.number_input("Height (cm)", 120.0, 220.0, 168.0, 0.5)
    waist  = b3.number_input("Waist (cm)", 50.0, 200.0, 92.0, 0.5)
    bmi = round(weight / ((height/100) ** 2), 1)
    st.caption(f"Calculated BMI: **{bmi}**")

    st.markdown("**💓  Blood Pressure**")
    p1, p2, p3 = st.columns(3)
    systolic  = p1.number_input("Systolic (mmHg)", 80, 220, 122)
    diastolic = p2.number_input("Diastolic (mmHg)", 40, 140, 78)
    pulse     = p3.number_input("Pulse (bpm)", 40, 140, 72)

    st.markdown("**🌿  Lifestyle**")
    l1, l2, l3, l4, l5 = st.columns(5)
    sleep      = l1.number_input("Sleep (hrs)", 2.0, 14.0, 7.0, 0.5)
    activity   = l2.number_input("Activity (min/day)", 0, 600, 30)
    smoke      = l3.selectbox("Smoking", ["Never","Former","Current"])
    alcohol    = l4.number_input("Alcohol (drinks/day)", 0, 15, 1)
    depression = l5.number_input("Depression score (0–27)", 0, 27, 2)

# map categorical → numeric (matching NHANES coding)
gender_code = 1 if gender == "Male" else 2
race_code   = {"Mexican American":1,"Other Hispanic":2,"White":3,"Black":4,"Asian":6,"Other":7}[race]
educ_code   = {"< 9th grade":1,"9-11th grade":2,"High school":3,"Some college":4,"College grad":5}[educ]
smoke_code  = {"Never":0,"Former":1,"Current":2}[smoke]

# full feature dictionary (all possible inputs)
all_inputs = {
    "RIDAGEYR": age, "RIAGENDR": gender_code, "RIDRETH3": race_code,
    "INDFMPIR": income, "DMDEDUC2": educ_code,
    "BMXWT": weight, "BMXHT": height, "BMXWAIST": waist, "BMXBMI": bmi,
    "systolic_avg": systolic, "diastolic_avg": diastolic, "pulse_avg": pulse,
    "avg_sleep_hours": sleep, "depression_score": depression,
    "total_activity": activity, "smoking_status": smoke_code,
    "alcohol_drinks": alcohol,
}

# ─────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────
def risk_level(p):
    if p < 0.33:  return "LOW",      "#22C55E", "rgba(34,197,94,0.12)"
    if p < 0.66:  return "MODERATE", "#F59E0B", "rgba(245,158,11,0.12)"
    return "HIGH", "#EF4444", "rgba(239,68,68,0.12)"

def gauge(p, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=p*100,
        number={"suffix": "%", "font": {"size": 30, "color": "#E6EDF5", "family": "Sora"}},
        gauge={
            "axis": {"range": [0,100], "tickcolor": "#3A4555", "tickfont": {"color":"#5C6677","size":9}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,33],  "color": "rgba(34,197,94,0.10)"},
                {"range": [33,66], "color": "rgba(245,158,11,0.10)"},
                {"range": [66,100],"color": "rgba(239,68,68,0.10)"},
            ],
        },
    ))
    fig.update_layout(height=200, margin=dict(l=15,r=15,t=20,b=5),
                      paper_bgcolor="rgba(0,0,0,0)", font={"color":"#E6EDF5"})
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
st.markdown("<br>", unsafe_allow_html=True)
run = st.button("⚡  Run Screening")

if run:
    if not chosen:
        st.warning("Please select at least one condition to screen for.")
    elif any(models[k] is None for k in chosen):
        st.error("Models not found. Ensure the 'models/' folder is next to app.py.")
    else:
        st.markdown('<div class="section-label">Results · Risk assessment</div>', unsafe_allow_html=True)

        # summary chart across selected diseases
        probs = {k: predict(k) for k in chosen}

        # bar overview
        bar = go.Figure(go.Bar(
            x=[probs[k]*100 for k in chosen],
            y=[DISEASES[k]["name"] for k in chosen],
            orientation="h",
            marker=dict(color=[risk_level(probs[k])[1] for k in chosen]),
            text=[f"{probs[k]*100:.0f}%" for k in chosen],
            textposition="outside",
            textfont=dict(color="#E6EDF5", family="Sora", size=14),
        ))
        bar.update_layout(
            height=90 + 60*len(chosen),
            margin=dict(l=10,r=40,t=10,b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(range=[0,100], showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                       tickfont=dict(color="#5C6677"), title=dict(text="Risk %", font=dict(color="#5C6677"))),
            yaxis=dict(tickfont=dict(color="#E6EDF5", size=13)),
        )
        st.plotly_chart(bar, use_container_width=True, config={"displayModeBar": False})

        # detailed cards
        for k in chosen:
            p = probs[k]
            level, color, bg = risk_level(p)
            info = DISEASES[k]

            colA, colB = st.columns([1, 1.4])
            with colA:
                st.plotly_chart(gauge(p, color), use_container_width=True, config={"displayModeBar": False})
            with colB:
                st.markdown(f"""
                <div class="result-card" style="border-color:{color}40;">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                        <div>
                            <p class="risk-name">{info['icon']}  {info['name']}</p>
                            <p class="risk-sub">Estimated screening risk</p>
                        </div>
                        <span class="badge" style="background:{bg};color:{color};">{level} RISK</span>
                    </div>
                    <div class="risk-score" style="color:{color};margin-top:0.6rem;">{p*100:.1f}%</div>
                    <div class="advice">
                        {"⚠️ <b>Recommend confirmatory testing.</b> " if level!="LOW" else "✅ <b>Low risk.</b> "}
                        {"This patient should be referred for a " + info['test'] + " to confirm." if level!="LOW"
                         else "Routine monitoring advised; no urgent testing indicated by this screen."}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# LIMITATIONS
# ─────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">About · Limitations & honest notes</div>', unsafe_allow_html=True)

st.markdown("""
<div class="glass" style="line-height:1.7;">
    <p style="margin:0 0 0.8rem 0;color:#E6EDF5;font-weight:600;font-family:'Sora',sans-serif;">
        This is a screening aid, not a diagnostic tool.
    </p>
    <ul style="margin:0;padding-left:1.1rem;color:#B5C0D0;font-size:0.9rem;">
        <li style="margin-bottom:0.5rem;">
            <b style="color:#E6EDF5;">Screening, not diagnosis.</b>
            The model flags who is <i>likely</i> at risk from low-cost inputs.
            A positive screen should be confirmed with the proper clinical lab test.
        </li>
        <li style="margin-bottom:0.5rem;">
            <b style="color:#E6EDF5;">Honest accuracy (AUC 0.80–0.82).</b>
            These are realistic screening scores — not an inflated "99%".
            An earlier version had data leakage; this version was rebuilt leakage-free.
        </li>
        <li style="margin-bottom:0.5rem;">
            <b style="color:#E6EDF5;">No family medical history.</b>
            Family history of diabetes / heart disease (a known risk factor)
            was dropped from the NHANES 2021–2023 public release, so it could
            not be included.
        </li>
        <li style="margin-bottom:0.5rem;">
            <b style="color:#E6EDF5;">Single population &amp; cycle.</b>
            Trained only on U.S. NHANES 2021–2023 adults. Performance on other
            populations is not guaranteed.
        </li>
        <li style="margin-bottom:0.5rem;">
            <b style="color:#E6EDF5;">Self-reported data.</b>
            Diagnoses and lifestyle answers come from questionnaires and carry
            inherent reporting bias.
        </li>
        <li style="margin-bottom:0;">
            <b style="color:#E6EDF5;">Metabolic syndrome.</b>
            Trained only on fasting participants, so its sample is smaller than
            the other models.
        </li>
    </ul>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# CREDITS
# ─────────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding:1.6rem 0 0.5rem 0;">
    <div style="font-family:'Sora',sans-serif;font-size:1.05rem;font-weight:700;
                background:linear-gradient(135deg,#5EEAD4,#818CF8);
                -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;">
        Designed, built &amp; trained by Vishwa A
    </div>
    <div style="color:#8A98AD;font-size:0.85rem;margin-top:0.3rem;">
        Data &amp; ML Engineer · Leakage-free ML pipeline · NHANES 2021–2023
    </div>
    <div style="margin-top:0.7rem;">
        <a href="https://github.com/itzmevishwa" style="color:#5EEAD4;text-decoration:none;font-size:0.85rem;margin:0 0.6rem;">GitHub</a>
        <span style="color:#3A4555;">·</span>
        <a href="https://linkedin.com/in/vishwa444" style="color:#5EEAD4;text-decoration:none;font-size:0.85rem;margin:0 0.6rem;">LinkedIn</a>
    </div>
</div>

<div class="disclaimer">
    CardioScan is an educational screening aid built on CDC NHANES 2021–2023 data
    using leakage-free machine learning.<br>
    It does not provide a diagnosis and is not a substitute for professional medical judgment.
</div>
""", unsafe_allow_html=True)
