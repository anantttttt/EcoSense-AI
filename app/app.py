from dotenv import load_dotenv
import os

import joblib
import pandas as pd
import requests
import altair as alt
import streamlit as st


# ============================================================
# Project paths + environment
# ============================================================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=True)

MODEL_PATH = os.path.join(SRC_DIR, "sustainability_model.joblib")
EVALUATION_PATH = os.path.join(SRC_DIR, "model_evaluation.joblib")

# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="EcoSense AI",
    page_icon="🌱",
    layout="wide",
)

# ============================================================
# Load model
# ============================================================

try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
except Exception as e:
    model = None
    model_loaded = False
    model_error = str(e)

try:
    evaluation = joblib.load(EVALUATION_PATH)
except Exception:
    evaluation = None


# ============================================================
# Environment / IBM Granite configuration
# ============================================================

WATSONX_API_KEY = os.getenv("WATSONX_API_KEY", "").strip()
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "").strip()
WATSONX_URL = os.getenv(
    "WATSONX_URL",
    "https://us-south.ml.cloud.ibm.com",
).strip()

GRANITE_MODEL_ID = os.getenv(
    "GRANITE_MODEL_ID",
    "ibm/granite-4-h-small",
).strip()

IBM_CONFIGURED = bool(WATSONX_API_KEY and WATSONX_PROJECT_ID)


# ============================================================
# Session state
#
# Streamlit reruns the script whenever a widget is clicked.
# Everything important is therefore stored in session_state so
# clicking Granite or What-If does not erase the analysis.
# ============================================================

DEFAULT_STATE = {
    "analysis_done": False,
    "analysis": None,
    "granite_advice": None,
    "granite_error": None,
    "scenario_done": False,
    "scenario": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# Sustainability scoring
# ============================================================

TRANSPORT_MAPPING = {
    "Public Transport": 0,
    "Walking / Cycling": 0,
    "Electric Vehicle": 1,
    "Motorcycle / Scooter": 2,
    "Petrol / Diesel Car": 3,
}


def electricity_score(value):
    if value <= 100:
        return 100
    if value <= 150:
        return 90
    if value <= 200:
        return 80
    if value <= 300:
        return 65
    if value <= 400:
        return 45
    return 25


def transport_score(value):
    scores = {
        "Walking / Cycling": 100,
        "Public Transport": 95,
        "Electric Vehicle": 85,
        "Motorcycle / Scooter": 65,
        "Petrol / Diesel Car": 40,
    }
    return scores.get(value, 50)


def waste_score(value):
    if value <= 2:
        return 100
    if value <= 5:
        return 90
    if value <= 7:
        return 75
    if value <= 10:
        return 60
    if value <= 15:
        return 40
    return 20


def water_score(value):
    if value <= 100:
        return 100
    if value <= 150:
        return 90
    if value <= 200:
        return 75
    if value <= 250:
        return 60
    if value <= 300:
        return 40
    return 20


def get_status(score):
    if score >= 80:
        return "🌱 Good"
    if score >= 60:
        return "🟡 Moderate"
    return "🔴 Needs Improvement"


def calculate_sustainability(electricity, transport, waste, water):
    category_scores = {
        "Electricity": electricity_score(electricity),
        "Transport": transport_score(transport),
        "Waste": waste_score(waste),
        "Water": water_score(water),
    }

    overall_score = round(sum(category_scores.values()) / 4, 1)
    biggest_impact = min(category_scores, key=category_scores.get)

    return {
        "score": overall_score,
        "status": get_status(overall_score),
        "category_scores": category_scores,
        "biggest_impact": biggest_impact,
    }


def build_recommendations(electricity, transport, waste, water, category_scores):
    recommendations = []

    if electricity > 300:
        recommendations.append(
            "⚡ Reduce unnecessary electricity use and consider LED lighting, "
            "efficient appliances, and switching off unused devices."
        )
    elif electricity > 200:
        recommendations.append(
            "💡 Look for energy-efficiency opportunities and reduce avoidable "
            "electricity consumption."
        )
    else:
        recommendations.append(
            "✅ Electricity usage is currently in a relatively lower range."
        )

    if transport == "Petrol / Diesel Car":
        recommendations.append(
            "🚗 When practical, consider public transport, carpooling, cycling, "
            "or walking for some journeys."
        )
    elif transport == "Motorcycle / Scooter":
        recommendations.append(
            "🛵 Combining trips and using public transport when possible could "
            "further reduce transport impact."
        )
    elif transport == "Electric Vehicle":
        recommendations.append(
            "🔋 Your selected transport option has a relatively strong score "
            "in this prototype's scoring framework."
        )
    elif transport in ["Public Transport", "Walking / Cycling"]:
        recommendations.append(
            "🚌 Your selected transport option scores well in this prototype."
        )

    if waste > 10:
        recommendations.append(
            "♻️ Reduce single-use products, reuse materials, and improve waste "
            "segregation where possible."
        )
    elif waste > 5:
        recommendations.append(
            "♻️ Try reducing avoidable packaging and improving waste segregation."
        )
    else:
        recommendations.append(
            "✅ Weekly waste is currently in a relatively lower range."
        )

    if water > 200:
        recommendations.append(
            "💧 Check for leaks, avoid unnecessary running water, and reuse water "
            "where appropriate."
        )
    elif water > 150:
        recommendations.append(
            "💧 There are opportunities to reduce water consumption through "
            "shorter usage periods and fixing leaks."
        )
    else:
        recommendations.append(
            "✅ Daily water usage is currently in a relatively lower range."
        )

    # Add a focused recommendation for the weakest category.
    weakest = min(category_scores, key=category_scores.get)

    focused = {
        "Electricity": (
            "🎯 Main opportunity: electricity has the lowest category score, "
            "so energy efficiency is the first area to focus on."
        ),
        "Transport": (
            "🎯 Main opportunity: transport has the lowest category score, "
            "so lower-impact travel choices are the first area to focus on."
        ),
        "Waste": (
            "🎯 Main opportunity: waste has the lowest category score, "
            "so reducing and segregating waste is the first area to focus on."
        ),
        "Water": (
            "🎯 Main opportunity: water has the lowest category score, "
            "so reducing avoidable water use is the first area to focus on."
        ),
    }

    recommendations.append(focused[weakest])
    return recommendations


def get_decision_factors(electricity, transport_value, waste, water):
    factors = []

    if electricity > 300:
        factors.append("⚡ High electricity consumption")
    elif electricity > 200:
        factors.append("💡 Elevated electricity consumption")

    if transport_value >= 2:
        factors.append("🚗 Higher-impact transport selection")

    if waste > 10:
        factors.append("♻️ High weekly waste generation")
    elif waste > 5:
        factors.append("♻️ Moderate-to-high weekly waste generation")

    if water > 200:
        factors.append("💧 High water consumption")
    elif water > 150:
        factors.append("💧 Elevated water consumption")

    if not factors:
        factors.append("🌱 No major high-impact factor detected by the rule-based checks.")

    return factors


def get_impact_areas(electricity, transport_value, waste, water):
    impact_areas = []

    if electricity > 300:
        impact_areas.append(
            "Energy: reducing unnecessary electricity use may lower energy demand "
            "and associated emissions."
        )

    if transport_value >= 2:
        impact_areas.append(
            "Transport: reducing fossil-fuel vehicle use may lower transport-related "
            "emissions."
        )

    if waste > 10:
        impact_areas.append(
            "Waste: reducing and segregating waste can support resource recovery "
            "and reduce avoidable waste."
        )

    if water > 200:
        impact_areas.append(
            "Water: reducing unnecessary water consumption can support more "
            "efficient use of freshwater resources."
        )

    if not impact_areas:
        impact_areas.append(
            "The current inputs do not indicate a major high-impact area in the "
            "prototype's rule-based checks."
        )

    return impact_areas


# ============================================================
# IBM Granite helpers
# ============================================================

def get_iam_token():
    response = requests.post(
        "https://iam.cloud.ibm.com/identity/token",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": WATSONX_API_KEY,
        },
        timeout=30,
    )

    if response.status_code != 200:
        try:
            details = response.json()
        except Exception:
            details = response.text

        raise RuntimeError(
            f"IAM authentication failed ({response.status_code}): {details}"
        )

    data = response.json()
    token = data.get("access_token")

    if not token:
        raise RuntimeError("IBM IAM did not return an access token.")

    return token


def ask_granite(analysis):
    token = get_iam_token()

    category_scores = analysis["category_scores"]

    prompt = f"""
You are IBM Granite acting as the sustainability advisor inside the EcoSense AI
prototype.

Use ONLY the information supplied below.

User sustainability inputs:
- Monthly electricity: {analysis["electricity"]} kWh
- Primary transport: {analysis["transport"]}
- Weekly household waste: {analysis["waste"]} kg
- Daily water usage: {analysis["water"]} litres

EcoSense prototype scoring:
- Overall score: {analysis["score"]}/100
- Overall status: {analysis["status"]}
- Electricity score: {category_scores["Electricity"]}/100
- Transport score: {category_scores["Transport"]}/100
- Waste score: {category_scores["Waste"]}/100
- Water score: {category_scores["Water"]}/100
- Main opportunity area: {analysis["biggest_impact"]}

Provide:
1. A short explanation of the user's overall sustainability profile.
2. Identify the main opportunity area and explain why it matters within this
   prototype's scoring framework.
3. Give exactly 3 practical, affordable actions.
4. Briefly explain why each action could help.

Rules:
- Do not invent carbon-emission quantities.
- Do not claim scientific certainty.
- Do not make medical, legal, or financial claims.
- Do not request sensitive personal information.
- Make clear that this is a prototype decision-support system.
- Keep the answer concise and easy to understand.
"""

    endpoint = (
        WATSONX_URL.rstrip("/")
        + "/ml/v1/text/chat?version=2025-10-25"
    )

    payload = {
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "project_id": WATSONX_PROJECT_ID,
        "model_id": GRANITE_MODEL_ID,
        "max_completion_tokens": 400,
        "temperature": 0.2,
    }

    response = requests.post(
        endpoint,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json=payload,
        timeout=90,
    )

    if response.status_code != 200:
        try:
            details = response.json()
        except Exception:
            details = response.text

        raise RuntimeError(
            f"watsonx request failed ({response.status_code}): {details}"
        )

    data = response.json()

    # Standard chat response.
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        pass

    # Fallback in case the service returns a slightly different structure.
    try:
        return data["choices"][0]["message"]["content"][0]["text"]
    except (KeyError, IndexError, TypeError):
        pass

    raise RuntimeError(
        "IBM Granite returned a response, but no readable assistant message "
        "was found."
    )


# ============================================================
# Green chart helpers
# ============================================================

def score_green(score):
    """Return a darker green as the sustainability score increases."""
    low = (132, 226, 174)   # lighter green for lower scores
    high = (7, 105, 62)     # darker green for higher scores
    ratio = max(0.0, min(1.0, float(score) / 100.0))
    rgb = tuple(round(low[i] + (high[i] - low[i]) * ratio) for i in range(3))
    return "#{:02x}{:02x}{:02x}".format(*rgb)


# ============================================================
# Phase 6: Dashboard / UI Upgrade
# ============================================================

st.markdown(
    """
    <style>
    /* =========================================================
       ECOSENSE AI | PREMIUM POLISH
       Presentation layer only.
       ========================================================= */

    .dashboard-shell {
        margin:0.35rem 0 1.2rem 0;
        padding:0.25rem 0;
    }

    .dashboard-status {
        display:flex;
        flex-wrap:wrap;
        gap:0.5rem;
        margin-bottom:0.65rem;
    }

    .status-pill {
        display:inline-flex;
        align-items:center;
        gap:0.35rem;
        padding:0.32rem 0.65rem;
        border-radius:999px;
        border:1px solid rgba(57,217,138,0.22);
        background:rgba(57,217,138,0.055);
        color:#b8fbd2;
        font-size:0.64rem;
        font-weight:800;
        letter-spacing:0.07em;
        text-transform:uppercase;
    }

    .status-pulse {
        width:6px;
        height:6px;
        border-radius:50%;
        background:#69f0ae;
        box-shadow:0 0 11px rgba(105,240,174,0.9);
    }

    .dashboard-title {
        font-family:var(--font-display);
        color:#f4fff9;
        font-size:1.68rem;
        font-weight:700;
        letter-spacing:-0.035em;
    }

    .dashboard-subtitle {
        color:rgba(232,255,244,0.48);
        font-size:0.83rem;
        margin-top:0.22rem;
    }

    .score-card.primary {
        min-height:245px !important;
        overflow:hidden;
    }

    .score-card.primary::after {
        content:"ECO INDEX";
        position:absolute;
        top:1rem;
        right:1rem;
        padding:0.25rem 0.48rem;
        border:1px solid rgba(57,217,138,0.16);
        border-radius:999px;
        color:rgba(105,240,174,0.60);
        font-size:0.53rem;
        font-weight:800;
        letter-spacing:0.12em;
    }

    .score-card:not(.primary) {
        min-height:245px !important;
        align-items:flex-start !important;
        text-align:left !important;
        justify-content:center !important;
    }

    .score-card:not(.primary)::before {
        content:"";
        position:absolute;
        left:1.55rem;
        right:1.55rem;
        bottom:1.15rem;
        height:2px;
        background:linear-gradient(90deg,rgba(57,217,138,0.45),transparent);
    }

    .score-ring {
        width:132px !important;
        height:132px !important;
        box-shadow:
            0 0 0 7px rgba(57,217,138,0.035),
            0 0 42px rgba(57,217,138,0.20) !important;
    }

    .score-ring-inner {
        width:104px !important;
        height:104px !important;
    }

    .score-ring-value {
        font-size:2.05rem !important;
        color:#a9ffce !important;
    }

    .mini-card {
        min-height:128px !important;
        background:linear-gradient(145deg,rgba(17,35,27,0.68),rgba(7,14,11,0.82)) !important;
    }

    .chart-panel {
        position:relative;
        background:
            radial-gradient(circle at 90% 0%,rgba(57,217,138,0.055),transparent 30%),
            linear-gradient(160deg,rgba(14,28,22,0.68),rgba(5,11,8,0.84)) !important;
    }

    .chart-panel::after {
        content:"0 — 100";
        position:absolute;
        top:1.05rem;
        right:1.35rem;
        color:rgba(232,255,244,0.30);
        font-size:0.58rem;
        letter-spacing:0.08em;
    }

    .prediction-card {
        position:relative;
        overflow:hidden;
    }

    .prediction-card::after {
        content:"ML";
        position:absolute;
        right:1rem;
        top:50%;
        transform:translateY(-50%);
        font-family:var(--font-display);
        font-size:1.8rem;
        font-weight:800;
        color:rgba(105,240,174,0.08);
    }

    .decision-card,
    .impact-card,
    .recommendation-card {
        position:relative;
        overflow:hidden;
    }

    .decision-card::after,
    .impact-card::after {
        content:"";
        position:absolute;
        width:75px;
        height:75px;
        right:-34px;
        bottom:-44px;
        border-radius:50%;
        border:1px solid rgba(57,217,138,0.08);
    }

    div[data-testid="stTabs"] {
        margin-top:1.25rem;
    }

    div[data-testid="stTabs"] div[data-baseweb="tab-list"] {
        padding:0.3rem;
        border:1px solid rgba(57,217,138,0.12);
        border-radius:14px;
        background:rgba(7,15,11,0.64);
    }

    button[data-baseweb="tab"] {
        border-radius:10px !important;
        border:1px solid transparent !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        border-color:rgba(57,217,138,0.18) !important;
        background:rgba(57,217,138,0.07) !important;
    }

    .simulator-hero {
        padding:1.35rem 1.5rem;
        margin-bottom:1rem;
        border-radius:19px;
        border:1px solid rgba(57,217,138,0.18);
        background:
            radial-gradient(circle at 92% 0%,rgba(57,217,138,0.11),transparent 34%),
            linear-gradient(145deg,rgba(16,36,27,0.66),rgba(7,14,11,0.84));
    }

    .simulator-kicker {
        color:#69f0ae;
        font-size:0.64rem;
        font-weight:850;
        letter-spacing:0.14em;
        text-transform:uppercase;
    }

    .simulator-title {
        color:#f4fff9;
        font-family:var(--font-display);
        font-size:1.4rem;
        font-weight:700;
        margin-top:0.22rem;
    }

    .simulator-copy {
        color:rgba(232,255,244,0.52);
        font-size:0.82rem;
        margin-top:0.25rem;
    }

    .granite-panel {
        box-shadow:0 24px 60px rgba(0,0,0,0.32) !important;
    }

    .granite-response {
        border-left:3px solid #39d98a !important;
    }

    .ra-card {
        min-height:165px;
        background:linear-gradient(145deg,rgba(13,28,21,0.62),rgba(7,14,11,0.76)) !important;
    }

    .ra-card::after {
        content:"✓";
        position:absolute;
        right:1rem;
        top:0.9rem;
        color:rgba(105,240,174,0.15);
        font-size:1.3rem;
        font-weight:900;
    }

    .onboarding-hero {
        padding:1.35rem 1.45rem;
        border-radius:18px;
        border:1px solid rgba(57,217,138,0.16);
        background:linear-gradient(145deg,rgba(14,31,23,0.62),rgba(6,13,10,0.82));
        margin-bottom:1rem;
    }

    .onboarding-title {
        color:#f4fff9;
        font-family:var(--font-display);
        font-size:1.15rem;
        font-weight:700;
    }

    .onboarding-copy {
        color:rgba(232,255,244,0.52);
        font-size:0.80rem;
        margin-top:0.25rem;
    }

    /* No default blue alert surfaces. */
    div[data-testid="stAlert"] {
        background:rgba(57,217,138,0.06) !important;
        border:1px solid rgba(57,217,138,0.17) !important;
        border-radius:14px !important;
        color:#f4fff9 !important;
    }

    div[data-testid="stAlert"] svg {
        color:#69f0ae !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    /* =========================================================
       EcoSense AI — Climate Intelligence Platform
       Dark / near-black canvas · layered emerald accents
       ========================================================= */

    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700;800&display=swap');

    :root{
        --bg-void:#05070a;
        --bg-deep:#070c0a;
        --surface-1: rgba(16,32,25,0.55);
        --surface-2: rgba(10,22,17,0.72);
        --line-soft: rgba(57,217,138,0.14);
        --line-mid: rgba(57,217,138,0.24);
        --line-strong: rgba(105,240,174,0.55);
        --g-forest:#0c6d3f;
        --g-mid:#1ca866;
        --g-bright:#39d98a;
        --g-neon:#69f0ae;
        --g-mint:#a9ffce;
        --ink:#f4fff9;
        --ink-dim: rgba(232,255,244,0.62);
        --ink-faint: rgba(232,255,244,0.40);
        --font-display:'Space Grotesk', 'Inter', sans-serif;
        --font-body:'Inter', -apple-system, sans-serif;
    }

    html, body, [class*="css"]{
        font-family: var(--font-body);
    }

    h1,h2,h3,h4, .eco-title, .score-number{
        font-family: var(--font-display);
    }

    /* ---------------- App canvas ---------------- */

    .stApp {
        background:
            radial-gradient(circle at 10% -6%, rgba(57,217,138,0.14), transparent 32%),
            radial-gradient(circle at 96% 4%, rgba(105,240,174,0.08), transparent 28%),
            radial-gradient(circle at 50% 115%, rgba(28,168,102,0.10), transparent 40%),
            repeating-linear-gradient(0deg, rgba(255,255,255,0.012) 0px, rgba(255,255,255,0.012) 1px, transparent 1px, transparent 64px),
            repeating-linear-gradient(90deg, rgba(255,255,255,0.012) 0px, rgba(255,255,255,0.012) 1px, transparent 1px, transparent 64px),
            var(--bg-void);
    }

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 4rem;
        max-width: 1460px;
    }

    ::selection{ background: rgba(57,217,138,0.30); color:#fff; }

    /* Scrollbar */
    ::-webkit-scrollbar{ width:10px; height:10px; }
    ::-webkit-scrollbar-track{ background:transparent; }
    ::-webkit-scrollbar-thumb{
        background: linear-gradient(180deg, rgba(57,217,138,0.45), rgba(57,217,138,0.15));
        border-radius: 999px;
        border: 2px solid var(--bg-void);
    }

    /* Headings / body copy */
    h1,h2,h3{ color: var(--ink) !important; letter-spacing:-0.02em; }
    h2{ font-weight:700 !important; }
    p, li, .stMarkdown, span{ color: var(--ink-dim); }
    .stCaption, [data-testid="stCaptionContainer"]{ color: var(--ink-faint) !important; }

    hr, div[data-testid="stDivider"]{
        border-color: rgba(57,217,138,0.14) !important;
    }

    /* ---------------- Section framing helpers ---------------- */

    .eyebrow{
        display:inline-flex;
        align-items:center;
        gap:0.4rem;
        color: var(--g-neon);
        font-size:0.68rem;
        font-weight:800;
        letter-spacing:0.16em;
        text-transform:uppercase;
        margin-bottom:0.5rem;
    }

    .eyebrow::before{
        content:"";
        width:6px; height:6px;
        border-radius:50%;
        background: var(--g-neon);
        box-shadow: 0 0 10px rgba(105,240,174,0.9);
    }

    .section-heading{
        font-family: var(--font-display);
        font-size:1.5rem;
        font-weight:700;
        color: var(--ink);
        margin: 0 0 0.15rem 0;
        letter-spacing:-0.01em;
    }

    .section-sub{
        color: var(--ink-faint);
        font-size:0.86rem;
        margin-bottom:1rem;
    }

    .subtle-divider {
        height:1px;
        background:linear-gradient(90deg, rgba(57,217,138,0.30), rgba(57,217,138,0.02) 70%);
        margin:1.6rem 0;
        border:none;
    }

    /* ---------------- Glass panel primitive ---------------- */

    .glass-panel{
        position:relative;
        border-radius:20px;
        border:1px solid var(--line-soft);
        background: linear-gradient(160deg, var(--surface-1), var(--surface-2));
        box-shadow: 0 18px 45px rgba(0,0,0,0.28), inset 0 1px 0 rgba(255,255,255,0.03);
        backdrop-filter: blur(6px);
    }

    /* ==================== HERO ==================== */

    .eco-hero {
        position:relative;
        overflow:hidden;
        padding:2.5rem 2.6rem;
        border-radius:26px;
        border:1px solid rgba(57,217,138,0.28);
        background:
            radial-gradient(circle at 85% 0%, rgba(57,217,138,0.20), transparent 45%),
            radial-gradient(circle at 15% 120%, rgba(105,240,174,0.10), transparent 50%),
            linear-gradient(150deg, rgba(14,48,34,0.85) 0%, rgba(7,13,10,0.96) 65%);
        box-shadow: 0 30px 80px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.05);
        margin-bottom:1.4rem;
    }

    .eco-hero::before{
        content:"";
        position:absolute;
        width:420px; height:420px;
        right:-160px; top:-220px;
        border-radius:50%;
        border:1px solid rgba(57,217,138,0.16);
        box-shadow: 0 0 120px rgba(57,217,138,0.10);
        animation: pulse-ring 6s ease-in-out infinite;
    }

    .eco-hero::after{
        content:"";
        position:absolute;
        inset:0;
        background-image: radial-gradient(rgba(105,240,174,0.35) 0.6px, transparent 0.6px);
        background-size: 26px 26px;
        opacity:0.10;
        pointer-events:none;
    }

    @keyframes pulse-ring{
        0%,100%{ opacity:0.55; transform:scale(1); }
        50%{ opacity:1; transform:scale(1.035); }
    }

    .eco-kicker {
        position:relative; z-index:1;
        color: var(--g-neon);
        font-size:0.72rem;
        font-weight:800;
        letter-spacing:0.2em;
        text-transform:uppercase;
        margin-bottom:0.7rem;
    }

    .eco-title {
        position:relative; z-index:1;
        font-family: var(--font-display);
        font-size:3.4rem;
        font-weight:700;
        line-height:1.02;
        letter-spacing:-0.04em;
        color: var(--ink);
        margin-bottom:0.5rem;
    }

    .eco-title-accent {
        background: linear-gradient(120deg, var(--g-mint), var(--g-neon) 55%, var(--g-mid));
        -webkit-background-clip:text;
        background-clip:text;
        color:transparent;
        filter: drop-shadow(0 0 22px rgba(57,217,138,0.28));
    }

    .eco-subtitle {
        position:relative; z-index:1;
        font-size:1.08rem;
        color: rgba(232,255,244,0.72);
        max-width:640px;
        margin-bottom:1rem;
    }

    .eco-badge-row{ position:relative; z-index:1; display:flex; flex-wrap:wrap; gap:0.5rem; }

    .eco-badge {
        display:inline-flex;
        align-items:center;
        gap:0.35rem;
        padding:0.42rem 0.85rem;
        border-radius:999px;
        border:1px solid rgba(57,217,138,0.28);
        background: rgba(57,217,138,0.07);
        color: rgba(232,255,244,0.85);
        font-size:0.74rem;
        font-weight:600;
    }

    /* ==================== SCORE / BENTO ROW ==================== */

    .score-card {
        position:relative;
        padding:1.5rem 1.4rem;
        border-radius:20px;
        border:1px solid var(--line-soft);
        background: linear-gradient(150deg, rgba(20,42,32,0.75), rgba(7,14,11,0.92));
        text-align:center;
        min-height:190px;
        display:flex;
        flex-direction:column;
        align-items:center;
        justify-content:center;
        box-shadow: 0 16px 40px rgba(0,0,0,0.24);
        transition: transform .22s ease, border-color .22s ease, box-shadow .22s ease;
    }

    .score-card:hover{
        transform: translateY(-3px);
        border-color: rgba(57,217,138,0.36);
        box-shadow: 0 22px 50px rgba(0,0,0,0.32), 0 0 30px rgba(57,217,138,0.08);
    }

    .score-card.primary{
        background:
            radial-gradient(circle at 30% 0%, rgba(57,217,138,0.14), transparent 55%),
            linear-gradient(150deg, rgba(20,48,35,0.85), rgba(6,13,10,0.95));
        border-color: rgba(57,217,138,0.30);
    }

    .score-ring{
        width:104px; height:104px;
        border-radius:50%;
        display:flex; align-items:center; justify-content:center;
        margin:0.35rem 0 0.6rem 0;
        background: conic-gradient(var(--g-neon) calc(var(--pct,0)*1%), rgba(255,255,255,0.06) 0);
        box-shadow: 0 0 30px rgba(57,217,138,0.18);
    }

    .score-ring-inner{
        width:82px; height:82px;
        border-radius:50%;
        background: radial-gradient(circle at 35% 25%, #0d211a, #050b08);
        display:flex; flex-direction:column; align-items:center; justify-content:center;
        border: 1px solid rgba(57,217,138,0.20);
    }

    .score-ring-value{
        font-family: var(--font-display);
        font-size:1.7rem;
        font-weight:700;
        color: var(--g-neon);
        line-height:1;
        text-shadow: 0 0 18px rgba(57,217,138,0.35);
    }

    .score-ring-max{ font-size:0.60rem; color: var(--ink-faint); letter-spacing:0.05em; }

    .score-number {
        font-size:2.1rem;
        font-weight:700;
        line-height:1;
        margin:0.5rem 0;
        color: var(--g-neon);
        text-shadow: 0 0 22px rgba(57,217,138,0.22);
    }

    .score-label {
        font-size:0.70rem;
        color: var(--ink-faint);
        letter-spacing:0.09em;
        text-transform:uppercase;
        font-weight:700;
    }

    .score-value-line{
        font-family: var(--font-display);
        font-size:1.55rem;
        font-weight:700;
        color: var(--ink);
        margin:0.85rem 0 0.35rem 0;
    }

    .score-progress-wrap {
        margin: 1.1rem 0 1.6rem 0;
        padding: 0.75rem 0.9rem;
        border-radius: 14px;
        border: 1px solid var(--line-soft);
        background: rgba(9,20,15,0.55);
    }

    .score-progress-meta {
        display:flex;
        justify-content:space-between;
        align-items:center;
        margin-bottom:0.45rem;
        font-size:0.66rem;
        color: var(--ink-faint);
        letter-spacing:0.09em;
        text-transform:uppercase;
        font-weight:800;
    }

    .score-progress-track {
        height:9px;
        width:100%;
        border-radius:999px;
        background:#0d1712;
        overflow:hidden;
        border:1px solid rgba(57,217,138,0.10);
    }

    .score-progress-fill {
        height:100%;
        border-radius:999px;
        background:linear-gradient(90deg,var(--g-forest),var(--g-mid),var(--g-neon));
        box-shadow:0 0 18px rgba(57,217,138,0.30);
        transition: width .5s ease;
    }

    /* ==================== PROFILE SNAPSHOT ==================== */

    .mini-card {
        position:relative;
        padding:1.05rem 1.1rem;
        border-radius:16px;
        border:1px solid var(--line-soft);
        background: rgba(12,26,20,0.60);
        min-height:108px;
        transition: transform .2s ease, border-color .2s ease;
    }

    .mini-card:hover{ transform: translateY(-2px); border-color: rgba(57,217,138,0.32); }

    .mini-icon{
        width:32px; height:32px;
        border-radius:9px;
        display:flex; align-items:center; justify-content:center;
        background: rgba(57,217,138,0.12);
        border:1px solid rgba(57,217,138,0.22);
        font-size:0.95rem;
        margin-bottom:0.55rem;
    }

    .mini-title {
        font-size:0.68rem;
        color: var(--ink-faint);
        margin-bottom:0.25rem;
        text-transform:uppercase;
        letter-spacing:0.08em;
        font-weight:700;
    }

    .mini-value {
        font-size:1.12rem;
        font-weight:700;
        color: var(--ink);
    }

    /* ==================== CHART PANEL ==================== */

    .chart-panel{
        padding:1.4rem 1.5rem 0.6rem 1.5rem;
        border-radius:20px;
        border:1px solid var(--line-soft);
        background: linear-gradient(160deg, rgba(14,28,22,0.55), rgba(7,13,11,0.75));
        margin-bottom:0.4rem;
    }

    .chart-legend{
        display:flex; align-items:center; gap:0.5rem;
        font-size:0.68rem;
        color: var(--ink-faint);
        letter-spacing:0.04em;
    }

    .chart-legend-bar{
        width:110px; height:8px; border-radius:999px;
        background: linear-gradient(90deg, #9aefbd, #39d98a 55%, #075c36);
    }

    /* ==================== CARDS: prediction / decision / impact / recs ==================== */

    .prediction-card {
        padding:1rem 1.15rem;
        border-radius:16px;
        border:1px solid rgba(57,217,138,0.22);
        background: linear-gradient(135deg, rgba(57,217,138,0.10), rgba(7,15,11,0.78));
        display:flex;
        align-items:center;
        gap:0.85rem;
        margin-bottom:1.1rem;
        box-shadow: 0 12px 30px rgba(0,0,0,0.2);
    }

    .prediction-icon {
        width:38px; height:38px;
        border-radius:11px;
        display:flex; align-items:center; justify-content:center;
        background: rgba(57,217,138,0.14);
        border:1px solid rgba(57,217,138,0.26);
        font-size:1.1rem;
        flex-shrink:0;
    }

    .prediction-kicker {
        color: var(--g-neon);
        font-size:0.64rem;
        font-weight:800;
        letter-spacing:0.11em;
        text-transform:uppercase;
    }

    .prediction-value {
        color: var(--ink);
        font-size:1.08rem;
        font-weight:700;
        margin-top:0.15rem;
    }

    .decision-card {
        padding:1rem 1.1rem;
        border-radius:15px;
        border:1px solid var(--line-soft);
        border-left:3px solid var(--g-bright);
        background: linear-gradient(135deg, rgba(57,217,138,0.06), rgba(7,15,11,0.75));
        min-height:74px;
        margin-bottom:0.7rem;
        color: var(--ink);
        transition: border-color .2s ease, transform .2s ease;
    }

    .decision-card:hover{ transform: translateX(2px); border-left-color: var(--g-neon); }

    .decision-number {
        display:inline-flex;
        width:26px; height:26px;
        align-items:center; justify-content:center;
        border-radius:8px;
        margin-right:0.6rem;
        background: rgba(57,217,138,0.12);
        color: var(--g-neon);
        border:1px solid rgba(57,217,138,0.22);
        font-size:0.72rem;
        font-weight:800;
    }

    .impact-card {
        padding:1.05rem 1.1rem;
        border-radius:15px;
        border:1px solid var(--line-soft);
        background: rgba(57,217,138,0.05);
        margin-bottom:0.7rem;
        color: var(--ink);
        line-height:1.55;
        transition: transform .2s ease, border-color .2s ease;
    }

    .impact-card:hover{ transform: translateX(2px); border-color: rgba(57,217,138,0.30); }

    .impact-tag {
        display:inline-block;
        margin-bottom:0.4rem;
        padding:0.26rem 0.6rem;
        border-radius:999px;
        background: rgba(57,217,138,0.12);
        border:1px solid rgba(57,217,138,0.20);
        color: var(--g-neon);
        font-size:0.62rem;
        font-weight:800;
        letter-spacing:0.09em;
        text-transform:uppercase;
    }

    .recommendation-card {
        display:flex;
        gap:0.8rem;
        align-items:flex-start;
        padding:1.05rem 1.1rem;
        margin-bottom:0.7rem;
        border-radius:15px;
        border:1px solid var(--line-soft);
        background: linear-gradient(90deg, rgba(57,217,138,0.06), rgba(7,15,11,0.68));
        transition: transform .2s ease, border-color .2s ease;
    }

    .recommendation-card:hover{ transform: translateY(-2px); border-color: rgba(57,217,138,0.30); }

    .recommendation-number {
        flex:0 0 32px; height:32px;
        border-radius:10px;
        display:flex; align-items:center; justify-content:center;
        background: linear-gradient(145deg, var(--g-forest), var(--g-bright));
        color:#03140b;
        font-weight:900;
        font-size:0.8rem;
        box-shadow: 0 0 18px rgba(57,217,138,0.18);
    }

    .recommendation-text { color: var(--ink); line-height:1.55; padding-top:0.2rem; }

    .section-kicker {
        color: var(--g-neon);
        font-size:0.66rem;
        font-weight:800;
        letter-spacing:0.14em;
        text-transform:uppercase;
        margin-bottom:0.3rem;
    }

    .green-alert {
        padding:1rem 1.1rem;
        border-radius:15px;
        border:1px solid var(--line-soft);
        background: rgba(57,217,138,0.06);
        color: var(--ink);
        margin-bottom:1rem;
    }

    /* ==================== GRANITE / AI PANEL ==================== */

    .granite-panel{
        position:relative;
        overflow:hidden;
        padding:1.5rem 1.6rem;
        border-radius:22px;
        border:1px solid rgba(57,217,138,0.26);
        background:
            radial-gradient(circle at 90% -10%, rgba(57,217,138,0.16), transparent 45%),
            linear-gradient(150deg, rgba(15,32,25,0.85), rgba(6,12,10,0.96));
        box-shadow: 0 20px 55px rgba(0,0,0,0.3);
        margin-bottom:1.2rem;
    }

    .granite-chip{
        display:inline-flex; align-items:center; gap:0.4rem;
        padding:0.32rem 0.75rem;
        border-radius:999px;
        border:1px solid rgba(57,217,138,0.30);
        background: rgba(57,217,138,0.08);
        color: var(--g-neon);
        font-size:0.65rem;
        font-weight:800;
        letter-spacing:0.10em;
        text-transform:uppercase;
        margin-bottom:0.8rem;
    }

    .granite-response{
        margin-top:1.1rem;
        padding:1.3rem 1.4rem;
        border-radius:16px;
        border:1px solid rgba(57,217,138,0.22);
        background: linear-gradient(160deg, rgba(9,20,15,0.9), rgba(6,11,9,0.95));
        color: var(--ink);
        line-height:1.7;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
    }

    .granite-response-header{
        display:flex; align-items:center; gap:0.6rem;
        margin-bottom:0.9rem;
        color: var(--g-neon);
        font-weight:800;
        font-size:0.78rem;
        letter-spacing:0.08em;
        text-transform:uppercase;
    }

    .granite-avatar{
        width:28px; height:28px; border-radius:8px;
        background: linear-gradient(145deg, var(--g-forest), var(--g-bright));
        display:flex; align-items:center; justify-content:center;
        font-size:0.85rem;
        color:#03140b;
    }

    /* ==================== RESPONSIBLE AI ==================== */

    .ra-card{
        padding:1.05rem 1.15rem;
        border-radius:15px;
        border:1px solid var(--line-soft);
        background: rgba(12,25,20,0.55);
        margin-bottom:0.75rem;
        transition: border-color .2s ease, transform .2s ease;
    }

    .ra-card:hover{ transform: translateY(-2px); border-color: rgba(57,217,138,0.28); }

    .ra-icon{
        width:34px; height:34px; border-radius:10px;
        display:flex; align-items:center; justify-content:center;
        background: rgba(57,217,138,0.12);
        border:1px solid rgba(57,217,138,0.20);
        margin-bottom:0.6rem;
        font-size:1rem;
    }

    .ra-title{ font-weight:700; color: var(--ink); margin-bottom:0.3rem; font-size:0.95rem; }
    .ra-text{ color: var(--ink-faint); font-size:0.84rem; line-height:1.55; }

    /* ==================== HOW IT WORKS FLOW ==================== */

    .section-card {
        padding:1.25rem 1.35rem;
        border-radius:18px;
        border:1px solid var(--line-soft);
        background: rgba(12,25,20,0.5);
        min-height:135px;
        transition: transform .2s ease, border-color .2s ease;
    }

    .section-card:hover{ transform: translateY(-3px); border-color: rgba(57,217,138,0.30); }

    .small-note { font-size:0.86rem; color: var(--ink-faint); }

    /* ==================== METRICS ==================== */

    div[data-testid="stMetric"] {
        padding:0.75rem 0.9rem;
        border-radius:13px;
        border:1px solid var(--line-soft);
        background: rgba(57,217,138,0.05);
        transition: border-color .2s ease;
    }

    div[data-testid="stMetric"]:hover{ border-color: rgba(57,217,138,0.26); }

    div[data-testid="stMetricValue"] { color: var(--g-neon) !important; font-family: var(--font-display); }
    div[data-testid="stMetricLabel"] { color: var(--ink-dim) !important; }

    /* ==================== ALERTS ==================== */

    div[data-testid="stAlert"] {
        background: rgba(57,217,138,0.06) !important;
        border: 1px solid var(--line-soft) !important;
        border-radius: 14px !important;
        color: var(--ink) !important;
    }
    div[data-testid="stAlert"] svg { color: var(--g-neon) !important; }

    a { color: var(--g-neon) !important; }

    div[data-baseweb="popover"] { border:1px solid var(--line-soft) !important; }
    div[data-baseweb="menu"] { background:#09120e !important; }
    div[data-baseweb="option"][aria-selected="true"] {
        background: rgba(57,217,138,0.12) !important;
        color: var(--g-neon) !important;
    }

    /* ==================== PROGRESS BAR / TABS ==================== */

    div[data-testid="stProgressBar"] > div > div > div {
        background: linear-gradient(90deg, var(--g-forest), var(--g-bright), var(--g-neon)) !important;
        box-shadow: 0 0 16px rgba(57,217,138,0.24);
    }

    div[data-testid="stTabs"] div[data-baseweb="tab-list"]{
        gap:0.35rem;
        border-bottom: 1px solid rgba(57,217,138,0.14);
    }

    button[data-baseweb="tab"] {
        color: var(--ink-faint) !important;
        font-weight:650 !important;
        border-radius: 10px 10px 0 0 !important;
        padding:0.55rem 1rem !important;
    }

    button[data-baseweb="tab"]:hover{
        color: rgba(232,255,244,0.85) !important;
        background: rgba(57,217,138,0.05) !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--g-neon) !important;
        background: rgba(57,217,138,0.08) !important;
    }

    div[data-baseweb="tab-highlight"] { background-color: var(--g-bright) !important; height:2.5px !important; }

    /* ==================== BUTTONS ==================== */

    .stButton > button {
        border-radius:12px;
        border:1px solid rgba(57,217,138,0.26) !important;
        background: linear-gradient(180deg, #121815, #080d0b) !important;
        color: var(--ink) !important;
        font-weight:700;
        letter-spacing:0.01em;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: rgba(57,217,138,0.65) !important;
        color: var(--g-neon) !important;
        background: linear-gradient(180deg, #16241d, #0a120e) !important;
        box-shadow: 0 0 22px rgba(57,217,138,0.12);
        transform: translateY(-1px);
    }

    .stButton button[kind="secondary"] {
        background:#08100c !important;
        color:#ffffff !important;
        border:1px solid rgba(57,217,138,0.40) !important;
        font-weight:750 !important;
    }

    .stButton button[kind="secondary"]:hover {
        background:#0e1a13 !important;
        color: var(--g-neon) !important;
        border-color: var(--g-bright) !important;
        box-shadow: 0 0 24px rgba(57,217,138,0.16) !important;
    }

    /* ==================== SIDEBAR ==================== */

    section[data-testid="stSidebar"] {
        background:
            radial-gradient(circle at 50% 0%, rgba(57,217,138,0.16), transparent 32%),
            linear-gradient(180deg, #091712 0%, #07100d 55%, #050a08 100%);
        border-right: 1px solid rgba(57,217,138,0.22);
        box-shadow: 12px 0 45px rgba(0,0,0,0.32);
    }

    section[data-testid="stSidebar"] > div { background: transparent; }

    div[data-testid="stSidebarContent"] { padding: 1.1rem 1rem 1.6rem 1rem; }

    .sidebar-brand {
        display:flex;
        align-items:center;
        gap:0.8rem;
        padding:0.2rem 0.15rem 1.15rem 0.15rem;
        border-bottom: 1px solid rgba(57,217,138,0.12);
        margin-bottom:1rem;
    }

    .brand-orbit {
        width:46px; height:46px;
        display:flex; align-items:center; justify-content:center;
        border-radius:14px;
        background: linear-gradient(145deg, rgba(57,217,138,0.22), rgba(57,217,138,0.04));
        border:1px solid rgba(57,217,138,0.40);
        box-shadow: 0 0 28px rgba(57,217,138,0.16), inset 0 1px 0 rgba(255,255,255,0.06);
        font-size:1.5rem;
    }

    .brand-name {
        font-family: var(--font-display);
        font-size:1.16rem;
        font-weight:700;
        letter-spacing:-0.02em;
        color: var(--ink);
        line-height:1.05;
    }

    .brand-version {
        margin-top:0.28rem;
        color: var(--g-neon);
        font-size:0.60rem;
        font-weight:800;
        letter-spacing:0.14em;
    }

    .sidebar-panel {
        padding:0.85rem 0.9rem;
        border:1px solid var(--line-soft);
        border-radius:15px;
        background: rgba(57,217,138,0.05);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.025);
        margin-bottom:0.9rem;
    }

    .sidebar-kicker {
        color: var(--g-neon);
        font-size:0.66rem;
        font-weight:800;
        letter-spacing:0.13em;
        text-transform:uppercase;
    }

    .sidebar-description {
        color: var(--ink-faint);
        font-size:0.70rem;
        line-height:1.5;
        margin-top:0.28rem;
    }

    .telemetry-label {
        display:flex;
        justify-content:space-between;
        align-items:center;
        margin:0.9rem 0 0.32rem 0;
        color: rgba(232,255,244,0.75);
        font-size:0.66rem;
        font-weight:800;
        letter-spacing:0.08em;
    }

    .telemetry-dot {
        width:7px; height:7px;
        border-radius:50%;
        background: var(--g-bright);
        box-shadow: 0 0 11px rgba(57,217,138,0.9);
        animation: dot-pulse 2.4s ease-in-out infinite;
    }

    @keyframes dot-pulse{
        0%,100%{ opacity:1; }
        50%{ opacity:0.35; }
    }

    section[data-testid="stSidebar"] div[data-baseweb="input"],
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background: rgba(2,8,5,0.88) !important;
        border:1px solid rgba(57,217,138,0.20) !important;
        border-radius:11px !important;
        box-shadow: inset 0 0 18px rgba(57,217,138,0.03);
    }

    section[data-testid="stSidebar"] div[data-baseweb="input"]:focus-within,
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div:focus-within {
        border-color: rgba(57,217,138,0.65) !important;
        box-shadow: 0 0 0 1px rgba(57,217,138,0.14), 0 0 20px rgba(57,217,138,0.09) !important;
    }

    section[data-testid="stSidebar"] input { color:#effff7 !important; }
    section[data-testid="stSidebar"] label { color: rgba(232,255,244,0.85) !important; }

    section[data-testid="stSidebar"] button[kind="primary"] {
        background: linear-gradient(135deg, #15985d, #39d98a) !important;
        border:1px solid rgba(105,240,174,0.75) !important;
        color:#03140b !important;
        font-weight:800 !important;
        border-radius:12px !important;
        min-height:3.2rem !important;
        box-shadow: 0 12px 30px rgba(57,217,138,0.20), inset 0 1px 0 rgba(255,255,255,0.18) !important;
        transition: all 0.2s ease;
        letter-spacing:0.02em;
    }

    section[data-testid="stSidebar"] button[kind="primary"]:hover {
        background: linear-gradient(135deg, #39d98a, #69f0ae) !important;
        transform: translateY(-1px);
        box-shadow: 0 14px 34px rgba(57,217,138,0.30), 0 0 24px rgba(57,217,138,0.12) !important;
    }

    .system-card {
        padding:0.7rem 0.8rem;
        border-radius:13px;
        border:1px solid var(--line-soft);
        background: rgba(2,8,5,0.62);
    }

    .system-line {
        display:flex;
        align-items:center;
        gap:0.55rem;
        font-size:0.74rem;
        color: rgba(232,255,244,0.80);
        padding:0.3rem 0;
    }

    .status-live {
        margin-left:auto;
        color: var(--g-neon);
        font-weight:800;
        font-size:0.63rem;
        letter-spacing:0.05em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Sidebar: green-tech control console
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-orbit">🌱</div>
            <div>
                <div class="brand-name">EcoSense AI</div>
                <div class="brand-version">SUSTAINABILITY INTELLIGENCE</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-panel">
            <div class="sidebar-kicker">◉ Input Telemetry</div>
            <div class="sidebar-description">
                Feed the decision engine with your current sustainability profile.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="telemetry-label"><span>⚡ ENERGY LOAD</span><span class="telemetry-dot"></span></div>',
        unsafe_allow_html=True,
    )

    electricity = st.number_input(
        "Monthly electricity (kWh)",
        min_value=0.0,
        value=150.0,
        step=10.0,
        key="input_electricity",
        help="Estimated household electricity consumption per month.",
    )

    st.markdown(
        '<div class="telemetry-label"><span>🚗 MOBILITY PROFILE</span><span class="telemetry-dot"></span></div>',
        unsafe_allow_html=True,
    )

    transport = st.selectbox(
        "Primary transportation",
        [
            "Public Transport",
            "Walking / Cycling",
            "Petrol / Diesel Car",
            "Motorcycle / Scooter",
            "Electric Vehicle",
        ],
        key="input_transport",
    )

    st.markdown(
        '<div class="telemetry-label"><span>♻️ WASTE STREAM</span><span class="telemetry-dot"></span></div>',
        unsafe_allow_html=True,
    )

    waste = st.number_input(
        "Weekly waste (kg)",
        min_value=0.0,
        value=5.0,
        step=0.5,
        key="input_waste",
        help="Estimated household waste generated per week.",
    )

    st.markdown(
        '<div class="telemetry-label"><span>💧 WATER DEMAND</span><span class="telemetry-dot"></span></div>',
        unsafe_allow_html=True,
    )

    water = st.number_input(
        "Daily water usage (litres)",
        min_value=0.0,
        value=150.0,
        step=10.0,
        key="input_water",
        help="Estimated daily household water usage.",
    )

    st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)

    analyse_clicked = st.button(
        "🌱  RUN SUSTAINABILITY ANALYSIS",
        type="primary",
        use_container_width=True,
    )

    st.markdown(
        """
        <div style="
            text-align:center;
            color:rgba(232,255,244,0.42);
            font-size:0.62rem;
            letter-spacing:0.10em;
            margin-top:0.6rem;
            text-transform:uppercase;">
            ● Decision engine ready
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="sidebar-panel">
            <div class="sidebar-kicker">⚙ System Telemetry</div>
            <div class="sidebar-description">
                Runtime components and AI services.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="system-card">', unsafe_allow_html=True)

    if model_loaded:
        st.markdown(
            '<div class="system-line">🟢 <span>Decision Tree ML</span><span class="status-live">READY</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="system-line">🔴 <span>Decision Tree ML</span><span style="margin-left:auto;color:#ff7777;">OFFLINE</span></div>',
            unsafe_allow_html=True,
        )
        st.caption(model_error)

    if IBM_CONFIGURED:
        st.markdown(
            '<div class="system-line">🟢 <span>IBM Granite</span><span class="status-live">CONNECTED</span></div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Model: {GRANITE_MODEL_ID}")
    else:
        st.markdown(
            '<div class="system-line">🟡 <span>IBM Granite</span><span style="margin-left:auto;color:#ffd166;">CONFIGURE</span></div>',
            unsafe_allow_html=True,
        )
        st.caption("Add watsonx credentials to .env.")

    st.markdown(
        '<div class="system-line">🟢 <span>Rule Engine</span><span class="status-live">ONLINE</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="system-line">🟢 <span>What-If Engine</span><span class="status-live">ONLINE</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="
            margin-top:1rem;
            padding:0.8rem 0.85rem;
            border-radius:13px;
            background:rgba(57,217,138,0.04);
            border:1px solid rgba(57,217,138,0.12);
            font-size:0.66rem;
            color:rgba(232,255,244,0.45);
            line-height:1.55;">
            <span style="color:#69f0ae;">●</span>
            ECOSENSE AI · PROTOTYPE BUILD<br>
            Climate Intelligence for Sustainable Living
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# Hero
# ============================================================

st.markdown(
    """
    <div class="eco-hero">
        <div class="eco-kicker">// SUSTAINABILITY INTELLIGENCE SYSTEM</div>
        <div class="eco-title">EcoSense <span class="eco-title-accent">AI</span> 🌱</div>
        <div class="eco-subtitle">
            Understand your sustainability profile, explore alternative choices, and get
            practical AI-assisted guidance — all from a single command center.
        </div>
        <div class="eco-badge-row">
            <span class="eco-badge">🎯 Decision Support</span>
            <span class="eco-badge">🤖 ML Classification</span>
            <span class="eco-badge">🧠 IBM Granite</span>
            <span class="eco-badge">🌍 SDG 12</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Run analysis
# ============================================================

if analyse_clicked:
    transport_value = TRANSPORT_MAPPING[transport]

    score_result = calculate_sustainability(
        electricity, transport, waste, water
    )

    recommendations = build_recommendations(
        electricity,
        transport,
        waste,
        water,
        score_result["category_scores"],
    )

    decision_factors = get_decision_factors(
        electricity,
        transport_value,
        waste,
        water,
    )

    impact_areas = get_impact_areas(
        electricity,
        transport_value,
        waste,
        water,
    )

    prediction = None

    if model_loaded:
        prediction_data = pd.DataFrame(
            [[electricity, transport_value, waste, water]],
            columns=["electricity", "transport", "waste", "water"],
        )

        try:
            prediction = model.predict(prediction_data)[0]
        except Exception as e:
            prediction = f"Prediction unavailable: {e}"

    st.session_state.analysis = {
        "electricity": electricity,
        "transport": transport,
        "transport_value": transport_value,
        "waste": waste,
        "water": water,
        "score": score_result["score"],
        "status": score_result["status"],
        "category_scores": score_result["category_scores"],
        "biggest_impact": score_result["biggest_impact"],
        "recommendations": recommendations,
        "decision_factors": decision_factors,
        "impact_areas": impact_areas,
        "prediction": prediction,
    }

    st.session_state.analysis_done = True
    st.session_state.granite_advice = None
    st.session_state.granite_error = None
    st.session_state.scenario_done = False
    st.session_state.scenario = None

# ============================================================
# Main dashboard
# ============================================================

if st.session_state.analysis_done and st.session_state.analysis is not None:
    analysis = st.session_state.analysis
    category_scores = analysis["category_scores"]

    st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="dashboard-shell">
            <div class="dashboard-status">
                <span class="status-pill"><span class="status-pulse"></span> Live Assessment</span>
                <span class="status-pill">Decision Engine Online</span>
                <span class="status-pill">4 Sustainability Signals</span>
            </div>
            <div class="dashboard-title">Sustainability Intelligence Dashboard</div>
            <div class="dashboard-subtitle">
                Your current profile, model result, impact signals, and practical next steps.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    score_col, status_col, opportunity_col = st.columns([1.15, 1, 1])

    score_value = max(0.0, min(100.0, float(analysis["score"])))

    with score_col:
        st.markdown(
            f'''<div class="score-card primary">
                <div class="score-label">OVERALL SUSTAINABILITY SCORE</div>
                <div class="score-ring" style="--pct:{score_value:.1f}">
                    <div class="score-ring-inner">
                        <div class="score-ring-value">{analysis["score"]}</div>
                        <div class="score-ring-max">/ 100</div>
                    </div>
                </div>
            </div>''',
            unsafe_allow_html=True,
        )

    with status_col:
        st.markdown(
            f'''<div class="score-card">
                <div class="score-label">CURRENT STATUS</div>
                <div class="score-value-line">{analysis["status"]}</div>
                <div class="score-label">Prototype score interpretation</div>
            </div>''',
            unsafe_allow_html=True,
        )

    with opportunity_col:
        st.markdown(
            f'''<div class="score-card">
                <div class="score-label">MAIN OPPORTUNITY</div>
                <div class="score-value-line">{analysis["biggest_impact"]}</div>
                <div class="score-label">Lowest category score</div>
            </div>''',
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="score-progress-wrap">
            <div class="score-progress-meta">
                <span>Prototype sustainability index</span>
                <span>{score_value:.1f}%</span>
            </div>
            <div class="score-progress-track">
                <div class="score-progress-fill" style="width:{score_value:.1f}%"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:0.75rem;">
            <div>
                <div class="eyebrow">Inputs Recap</div>
                <div class="section-heading" style="font-size:1.2rem;">🧭 Profile Snapshot</div>
            </div>
            <div style="font-size:0.62rem;color:rgba(232,255,244,0.34);text-transform:uppercase;letter-spacing:0.08em;">
                Current telemetry
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    snap_cols = st.columns(4)
    snapshots = [
        ("⚡", "Electricity", f'{analysis["electricity"]} kWh/mo'),
        ("🚗", "Transport", analysis["transport"]),
        ("♻️", "Waste", f'{analysis["waste"]} kg/week'),
        ("💧", "Water", f'{analysis["water"]} L/day'),
    ]

    for col, (icon, title, value) in zip(snap_cols, snapshots):
        with col:
            st.markdown(
                f'''<div class="mini-card">
                    <div class="mini-icon">{icon}</div>
                    <div class="mini-title">{title}</div>
                    <div class="mini-value">{value}</div>
                </div>''',
                unsafe_allow_html=True,
            )

    dashboard_tab, simulator_tab, granite_tab, responsible_tab = st.tabs(
        [
            "📈 Overview",
            "🔮 What-If Simulator",
            "🧠 IBM Granite",
            "🛡️ Responsible AI",
        ]
    )

    # ========================================================
    # Overview
    # ========================================================

    with dashboard_tab:
        st.markdown('<div class="section-kicker">// MODEL OUTPUT</div>', unsafe_allow_html=True)
        st.markdown("### 🤖 AI Sustainability Classification")

        if analysis["prediction"] is not None:
            st.markdown(
                f"""
                <div class="prediction-card">
                    <div class="prediction-icon">🧠</div>
                    <div>
                        <div class="prediction-kicker">Decision Tree Classification</div>
                        <div class="prediction-value">Model prediction: {analysis["prediction"]}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="green-alert">
                    <strong>ML prediction unavailable.</strong><br>
                    <span class="small-note">The rule-based sustainability analysis is still available.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-kicker">// CATEGORY BREAKDOWN</div>', unsafe_allow_html=True)
        st.markdown("### 📈 Category Performance")

        score_cols = st.columns(4)

        for col, category in zip(
            score_cols,
            ["Electricity", "Transport", "Waste", "Water"],
        ):
            with col:
                st.metric(
                    category,
                    f'{category_scores[category]}/100',
                )

        chart_data = pd.DataFrame(
            {
                "Category": list(category_scores.keys()),
                "Score": list(category_scores.values()),
            }
        )

        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                <span class="section-kicker" style="margin-bottom:0;">SCORE BY CATEGORY</span>
                <span class="chart-legend">Lower<span class="chart-legend-bar"></span>Higher</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Continuous green gradient: lower scores are lighter, higher scores are darker.
        green_chart = (
            alt.Chart(chart_data)
            .mark_bar(
                cornerRadiusTopLeft=10,
                cornerRadiusTopRight=10,
                size=64,
            )
            .encode(
                x=alt.X(
                    "Category:N",
                    sort=None,
                    title=None,
                    axis=alt.Axis(
                        labelColor="#dff9ea",
                        labelFont="Inter",
                        labelFontSize=12,
                        labelAngle=0,
                        domain=False,
                        ticks=False,
                    ),
                ),
                y=alt.Y(
                    "Score:Q",
                    title="Score (0–100)",
                    scale=alt.Scale(domain=[0, 100]),
                    axis=alt.Axis(
                        labelColor="#9fc9b4",
                        titleColor="#9fc9b4",
                        labelFont="Inter",
                        titleFont="Inter",
                        gridColor="#15251c",
                        domain=False,
                        tickCount=5,
                    ),
                ),
                color=alt.Color(
                    "Score:Q",
                    scale=alt.Scale(
                        domain=[0, 100],
                        range=["#9aefbd", "#075c36"],
                    ),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("Category:N", title="Category"),
                    alt.Tooltip("Score:Q", title="Score", format=".1f"),
                ],
            )
            .properties(
                height=380,
                background="transparent",
            )
            .configure_view(strokeWidth=0)
        )

        st.altair_chart(
            green_chart,
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)

        left, right = st.columns(2, gap="large")

        with left:
            st.markdown('<div class="section-kicker">// MODEL EXPLANATION</div>', unsafe_allow_html=True)
            st.markdown("### 🔎 AI Decision Factors")

            if analysis["decision_factors"]:
                for index, factor in enumerate(analysis["decision_factors"], start=1):
                    st.markdown(
                        f"""
                        <div class="decision-card">
                            <span class="decision-number">{index:02d}</span>
                            {factor}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '<div class="green-alert">No additional decision factors were triggered by the current inputs.</div>',
                    unsafe_allow_html=True,
                )

        with right:
            st.markdown('<div class="section-kicker">// IMPACT SCAN</div>', unsafe_allow_html=True)
            st.markdown("### 🌍 Potential Impact Areas")

            if analysis["impact_areas"]:
                for impact in analysis["impact_areas"]:
                    st.markdown(
                        f"""
                        <div class="impact-card">
                            <span class="impact-tag">Potential impact</span><br>
                            {impact}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    """
                    <div class="impact-card">
                        <span class="impact-tag">Impact scan</span><br>
                        No major high-impact area was identified by the prototype's current rule-based checks.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-kicker">// ACTION ENGINE</div>', unsafe_allow_html=True)
        st.markdown("### 💡 Recommended Actions")

        for index, recommendation in enumerate(
            analysis["recommendations"], start=1
        ):
            st.markdown(
                f"""
                <div class="recommendation-card">
                    <div class="recommendation-number">{index}</div>
                    <div class="recommendation-text">{recommendation}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("🧮 How is the sustainability score calculated?"):
            st.write(
                "EcoSense AI converts each input into a 0–100 category score using "
                "transparent prototype thresholds, then calculates the overall score "
                "as the equal average of Electricity, Transport, Waste, and Water."
            )
            st.markdown(
                """
                **Overall score**

                `Overall Score = (Electricity + Transport + Waste + Water) / 4`

                **Interpretation**

                - 80–100 → Good
                - 60–79 → Moderate
                - Below 60 → Needs Improvement
                """
            )

    # ========================================================
    # What-If Simulator
    # ========================================================

    with simulator_tab:
        st.markdown(
            """
            <div class="simulator-hero">
                <div class="simulator-kicker">// SCENARIO SIMULATION LAB</div>
                <div class="simulator-title">🔮 What-If Sustainability Simulator</div>
                <div class="simulator-copy">
                    Change one or more inputs and observe how the prototype score,
                    category performance, and ML classification respond.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        scenario_col1, scenario_col2 = st.columns(2)

        with scenario_col1:
            scenario_electricity = st.number_input(
                "Scenario electricity (kWh/month)",
                min_value=0.0,
                value=float(analysis["electricity"]),
                step=10.0,
                key="scenario_electricity",
            )

            scenario_transport = st.selectbox(
                "Scenario transport",
                list(TRANSPORT_MAPPING.keys()),
                index=list(TRANSPORT_MAPPING.keys()).index(
                    analysis["transport"]
                ),
                key="scenario_transport",
            )

        with scenario_col2:
            scenario_waste = st.number_input(
                "Scenario waste (kg/week)",
                min_value=0.0,
                value=float(analysis["waste"]),
                step=0.5,
                key="scenario_waste",
            )

            scenario_water = st.number_input(
                "Scenario water (litres/day)",
                min_value=0.0,
                value=float(analysis["water"]),
                step=10.0,
                key="scenario_water",
            )

        if st.button(
            "🔄 Analyse What-If Scenario",
            use_container_width=True,
        ):
            scenario_result = calculate_sustainability(
                scenario_electricity,
                scenario_transport,
                scenario_waste,
                scenario_water,
            )

            scenario_prediction = None

            if model_loaded:
                scenario_transport_value = TRANSPORT_MAPPING[
                    scenario_transport
                ]

                scenario_prediction_data = pd.DataFrame(
                    [[
                        scenario_electricity,
                        scenario_transport_value,
                        scenario_waste,
                        scenario_water,
                    ]],
                    columns=[
                        "electricity",
                        "transport",
                        "waste",
                        "water",
                    ],
                )

                try:
                    scenario_prediction = model.predict(
                        scenario_prediction_data
                    )[0]
                except Exception as e:
                    scenario_prediction = f"Unavailable: {e}"

            score_change = round(
                scenario_result["score"] - analysis["score"],
                1,
            )

            current_scores = analysis["category_scores"]
            new_scores = scenario_result["category_scores"]

            improved = [
                category
                for category in current_scores
                if new_scores[category] > current_scores[category]
            ]

            declined = [
                category
                for category in current_scores
                if new_scores[category] < current_scores[category]
            ]

            st.session_state.scenario = {
                "electricity": scenario_electricity,
                "transport": scenario_transport,
                "waste": scenario_waste,
                "water": scenario_water,
                "score": scenario_result["score"],
                "status": scenario_result["status"],
                "category_scores": new_scores,
                "prediction": scenario_prediction,
                "score_change": score_change,
                "improved": improved,
                "declined": declined,
            }
            st.session_state.scenario_done = True

        if st.session_state.scenario_done and st.session_state.scenario:
            scenario = st.session_state.scenario
            current_scores = analysis["category_scores"]

            st.markdown('<div class="section-kicker">// SIMULATION OUTPUT</div>', unsafe_allow_html=True)
            st.markdown("### 📊 Scenario Result")

            sc1, sc2, sc3 = st.columns(3)

            with sc1:
                st.metric(
                    "Scenario Score",
                    f'{scenario["score"]}/100',
                    delta=f'{scenario["score_change"]:+.1f}',
                )

            with sc2:
                st.metric("Scenario Status", scenario["status"])

            with sc3:
                st.metric(
                    "Scenario ML Prediction",
                    scenario["prediction"] or "Unavailable",
                )

            if scenario["score_change"] > 0:
                st.markdown(
                    f'<div class="green-alert"><strong>🌱 Scenario improvement detected</strong><br><span class="small-note">Prototype score increases by {scenario["score_change"]} points.</span></div>',
                    unsafe_allow_html=True,
                )
            elif scenario["score_change"] < 0:
                st.markdown(
                    f'<div class="green-alert"><strong>Scenario trade-off detected</strong><br><span class="small-note">Prototype score decreases by {abs(scenario["score_change"])} points.</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                '<div class="green-alert"><strong>Scenario unchanged.</strong><br><span class="small-note">This scenario produces the same prototype score.</span></div>',
                unsafe_allow_html=True,
            )

            comparison = pd.DataFrame(
                {
                    "Category": [
                        "Electricity",
                        "Transport",
                        "Waste",
                        "Water",
                    ],
                    "Current": [
                        current_scores["Electricity"],
                        current_scores["Transport"],
                        current_scores["Waste"],
                        current_scores["Water"],
                    ],
                    "Scenario": [
                        scenario["category_scores"]["Electricity"],
                        scenario["category_scores"]["Transport"],
                        scenario["category_scores"]["Waste"],
                        scenario["category_scores"]["Water"],
                    ],
                }
            )

            st.dataframe(
                comparison,
                hide_index=True,
                use_container_width=True,
            )

            if scenario["improved"]:
                st.markdown(
                    f'<div class="green-alert"><strong>🟢 Improved areas</strong><br><span class="small-note">{", ".join(scenario["improved"])}</span></div>',
                    unsafe_allow_html=True,
                )

            if scenario["declined"]:
                st.markdown(
                    f'<div class="green-alert"><strong>Areas requiring attention</strong><br><span class="small-note">{", ".join(scenario["declined"])}</span></div>',
                    unsafe_allow_html=True,
                )

            st.caption(
                "The What-If simulator shows changes according to the prototype's "
                "deterministic scoring rules and ML model. It is not a measured "
                "environmental impact calculator."
            )

    # ========================================================
    # IBM Granite
    # ========================================================

    with granite_tab:
        st.markdown(
            """
            <div class="granite-panel">
                <div class="granite-chip">✦ Generative AI Layer</div>
                <div class="section-heading" style="font-size:1.35rem;">🧠 IBM Granite Sustainability Advisor</div>
                <div class="section-sub" style="margin-bottom:0;">
                    Granite adds a generative-AI explanation layer that turns the assessment
                    into concise, practical sustainability guidance.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if IBM_CONFIGURED:
            st.markdown(
                f"""
                <div class="prediction-card">
                    <div class="prediction-icon">✦</div>
                    <div>
                        <div class="prediction-kicker">IBM Granite Runtime</div>
                        <div class="prediction-value">Connected to {GRANITE_MODEL_ID}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button(
                "🤖 Ask IBM Granite for Advice",
                type="secondary",
                use_container_width=True,
            ):
                with st.spinner(
                    "IBM Granite is analysing your profile..."
                ):
                    try:
                        advice = ask_granite(analysis)
                        st.session_state.granite_advice = advice
                        st.session_state.granite_error = None
                    except Exception as e:
                        st.session_state.granite_advice = None
                        st.session_state.granite_error = str(e)

            if st.session_state.granite_advice:
                st.markdown(
                    """
                    <div class="granite-response">
                        <div class="granite-response-header">
                            <span class="granite-avatar">✦</span>
                            Granite Advice
                        </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(st.session_state.granite_advice)
                st.markdown("</div>", unsafe_allow_html=True)

            if st.session_state.granite_error:
                st.markdown(
                    """
                    <div class="green-alert">
                        <strong>✦ Granite is temporarily unavailable.</strong><br>
                        <span class="small-note">
                            The core EcoSense assessment remains available. Try the AI advisor again later.
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:
            st.markdown(
                """
                <div class="green-alert">
                    <strong>✦ IBM Granite is in demo mode.</strong><br>
                    <span class="small-note">
                        Configure WATSONX_API_KEY and WATSONX_PROJECT_ID in the project's .env file
                        and restart Streamlit.
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ========================================================
    # Responsible AI
    # ========================================================

    with responsible_tab:
        st.markdown('<div class="section-kicker">// GOVERNANCE</div>', unsafe_allow_html=True)
        st.markdown("### 🛡️ Responsible AI")

        ra1, ra2 = st.columns(2)

        ra_items_left = [
            ("⚖️", "Fairness",
             "EcoSense AI does not use sensitive personal characteristics such "
             "as religion, ethnicity, gender, or income to classify sustainability "
             "profiles. Recommendations are based on the sustainability inputs."),
            ("🔎", "Transparency",
             "The system exposes category scores, the main opportunity area, "
             "decision factors, and the scoring methodology."),
            ("🧭", "Ethics",
             "EcoSense AI provides sustainability guidance rather than claiming "
             "to make scientifically definitive judgments about environmental impact."),
        ]

        ra_items_right = [
            ("🔐", "Privacy",
             "The prototype does not require personally identifiable or sensitive "
             "information to produce its assessment."),
            ("⚠️", "ML Limitation",
             "The machine-learning model was trained using a synthetic dataset "
             "created for prototype development. Its predictions should therefore "
             "not be treated as scientifically validated environmental measurements."),
        ]

        with ra1:
            for icon, title, text in ra_items_left:
                st.markdown(
                    f"""
                    <div class="ra-card">
                        <div class="ra-icon">{icon}</div>
                        <div class="ra-title">{title}</div>
                        <div class="ra-text">{text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with ra2:
            for icon, title, text in ra_items_right:
                st.markdown(
                    f"""
                    <div class="ra-card">
                        <div class="ra-icon">{icon}</div>
                        <div class="ra-title">{title}</div>
                        <div class="ra-text">{text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with st.expander("ℹ️ Model Evaluation"):
            if evaluation:
                metric_rows = []

                for key in [
                    "accuracy",
                    "precision",
                    "recall",
                    "f1_score",
                ]:
                    if key in evaluation:
                        metric_rows.append(
                            {
                                "Metric": key.replace("_", " ").title(),
                                "Value": round(float(evaluation[key]), 4),
                            }
                        )

                if metric_rows:
                    st.dataframe(
                        pd.DataFrame(metric_rows),
                        hide_index=True,
                        use_container_width=True,
                    )
            else:
                st.markdown(
                    '<div class="green-alert"><strong>Model evaluation data unavailable.</strong></div>',
                    unsafe_allow_html=True,
                )

        st.caption(
            "Prototype limitation: model predictions are not environmental "
            "measurements or scientific certifications."
        )

else:
    st.markdown(
        """
        <div class="onboarding-hero">
            <div class="eyebrow">System Ready</div>
            <div class="onboarding-title">Build your sustainability intelligence profile</div>
            <div class="onboarding-copy">
                Enter your four sustainability signals in the control panel, then run the
                analysis to activate the dashboard, ML classification, scenario lab, and AI advisor.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🧩 How EcoSense AI works")

    flow_cols = st.columns(4)
    flow = [
        ("01 // INPUT", "Provide electricity, transport, waste, and water data."),
        ("02 // SCORE", "Transparent rules convert inputs into category scores."),
        ("03 // PREDICT", "A Decision Tree provides an ML classification."),
        ("04 // ACT", "Explore What-If scenarios and Granite guidance."),
    ]

    for col, (title, description) in zip(flow_cols, flow):
        with col:
            st.markdown(
                f'''<div class="section-card">
                    <strong>{title}</strong><br>
                    <span class="small-note">{description}</span>
                </div>''',
                unsafe_allow_html=True,
            )

st.divider()
st.caption(
    "EcoSense AI is a sustainability decision-support prototype. Results depend "
    "on the supplied inputs, prototype scoring rules, training data, and model."
)