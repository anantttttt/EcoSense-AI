import streamlit as st
import os
import joblib
import pandas as pd
# -----------------------------
# Load ML Model
# -----------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "src",
    "sustainability_model.joblib"
)

model = joblib.load(MODEL_PATH)

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="EcoSense AI",
    page_icon="🌱",
    layout="wide"
)

# -----------------------------
# Custom Styling
# -----------------------------

st.markdown(
    """
    <style>
    .main {
        padding-top: 2rem;
    }

    .eco-card {
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        margin-bottom: 1rem;
    }

    .eco-title {
        font-size: 2.8rem;
        font-weight: 700;
    }

    .eco-subtitle {
        font-size: 1.2rem;
        opacity: 0.8;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="eco-title">🌱 EcoSense AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="eco-subtitle">'
    'AI-Powered Sustainability Decision Support'
    '</div>',
    unsafe_allow_html=True
)
st.write(
    "EcoSense AI helps users understand the sustainability impact "
    "of everyday choices and provides practical recommendations."
)
st.caption(
    "🌍 Primary SDG: SDG 12 — Responsible Consumption and Production"
)

st.divider()

# -----------------------------
# User Input
# -----------------------------
st.header("🔍 Sustainability Assessment")

col1, col2 = st.columns(2)

with col1:
    electricity = st.number_input(
        "Monthly electricity consumption (kWh)",
        min_value=0.0,
        value=150.0,
        step=10.0
    )

    transport = st.selectbox(
        "Primary mode of transportation",
        [
            "Public Transport",
            "Walking / Cycling",
            "Petrol / Diesel Car",
            "Motorcycle / Scooter",
            "Electric Vehicle"
        ]
    )

with col2:
    waste = st.number_input(
        "Estimated household waste per week (kg)",
        min_value=0.0,
        value=5.0,
        step=0.5
    )

    water = st.number_input(
        "Estimated daily water usage (litres)",
        min_value=0.0,
        value=150.0,
        step=10.0
    )

st.divider()

# -----------------------------
# Sustainability Analysis
# -----------------------------
if st.button("🌱 Analyse My Sustainability", type="primary"):

    # -----------------------------
    # ML Prediction
    # -----------------------------

    transport_mapping = {
        "Public Transport": 0,
        "Walking / Cycling": 0,
        "Petrol / Diesel Car": 3,
        "Motorcycle / Scooter": 2,
        "Electric Vehicle": 1
    }

    transport_value = transport_mapping[transport]

    prediction_data = pd.DataFrame(
    [[
        electricity,
        transport_value,
        waste,
        water
    ]],
    columns=[
        "electricity",
        "transport",
        "waste",
        "water"
    ]
)

    prediction = model.predict(prediction_data)[0]

    st.subheader("🤖 AI Sustainability Classification")
    st.success(f"EcoSense AI predicts: **{prediction}**")


    st.subheader("🔎 AI Decision Factors")

    factors = []

    if electricity > 300:
        factors.append("⚡ High electricity consumption")

    if transport_value >= 2:
        factors.append("🚗 High-emission transport")

    if waste > 10:
        factors.append("♻️ High weekly waste generation")

    if water > 200:
        factors.append("💧 High water consumption")

    if not factors:
        factors.append("🌱 No major high-impact factor detected")

    for factor in factors:
        st.write(f"- {factor}")
            # -----------------------------
    # Impact Analysis
    # -----------------------------

    st.subheader("🌍 Potential Impact Areas")

    impact_areas = []

    if electricity > 300:
        impact_areas.append(
            "Energy: Reducing unnecessary electricity use may "
            "lower energy demand and associated emissions."
        )

    if transport_value >= 2:
        impact_areas.append(
            "Transport: Reducing fossil-fuel vehicle use may "
            "lower transport-related emissions."
        )

    if waste > 10:
        impact_areas.append(
            "Waste: Reducing and segregating waste can support "
            "resource recovery and reduce avoidable waste."
        )

    if water > 200:
        impact_areas.append(
            "Water: Reducing unnecessary water consumption can "
            "support more efficient use of freshwater resources."
        )

    if not impact_areas:
        impact_areas.append(
            "Your current inputs do not indicate a major high-impact "
            "area. Maintaining these habits can support sustainable living."
        )

    for impact in impact_areas:
        st.info(impact)
            # -----------------------------
    # Sustainability Dashboard
    # -----------------------------

    st.subheader("📊 Sustainability Dashboard")

    chart_data = {
        "Category": [
            "Electricity",
            "Transport",
            "Waste",
            "Water"
        ],
        "Value": [
            electricity,
            transport_value,
            waste,
            water
        ]
    }

    st.bar_chart(
        chart_data,
        x="Category",
        y="Value"
    )

    score = 100
    recommendations = []

    # Electricity analysis
    if electricity > 300:
        score -= 25
        recommendations.append(
            "⚡ Your electricity consumption is relatively high. "
            "Consider energy-efficient appliances, LED lighting, "
            "and switching off unused devices."
        )
    elif electricity > 200:
        score -= 15
        recommendations.append(
            "💡 Your electricity consumption could be reduced by "
            "improving energy efficiency and avoiding unnecessary usage."
        )
    else:
        recommendations.append(
            "✅ Your electricity consumption is within a relatively "
            "lower range."
        )

    # Transport analysis
    if transport == "Petrol / Diesel Car":
        score -= 20
        recommendations.append(
            "🚗 Consider public transport, carpooling, cycling, or "
            "walking when practical to reduce transport-related emissions."
        )

    elif transport == "Motorcycle / Scooter":
        score -= 10
        recommendations.append(
            "🛵 Combining trips and using public transport when possible "
            "could further reduce your transport footprint."
        )

    elif transport == "Electric Vehicle":
        recommendations.append(
            "🔋 Electric transport can reduce direct tailpipe emissions. "
            "Charging from cleaner electricity can improve the benefit further."
        )

    else:
        recommendations.append(
            "🚌 Your selected transport option generally has lower "
            "direct emissions than individual fossil-fuel transport."
        )

    # Waste analysis
    if waste > 10:
        score -= 20
        recommendations.append(
            "♻️ Your estimated weekly waste is high. Focus on reducing "
            "single-use products, reusing materials, and segregating waste."
        )
    elif waste > 5:
        score -= 10
        recommendations.append(
            "♻️ Try reducing avoidable packaging and improving waste segregation."
        )
    else:
        recommendations.append(
            "✅ Your estimated weekly waste is relatively low."
        )

    # Water analysis
    if water > 200:
        score -= 15
        recommendations.append(
            "💧 Your estimated water usage is high. Check for leaks, "
            "reduce unnecessary running water, and reuse water where appropriate."
        )
    elif water > 150:
        score -= 8
        recommendations.append(
            "💧 There may be opportunities to reduce water consumption "
            "through shorter usage periods and fixing leaks."
        )
    else:
        recommendations.append(
            "✅ Your estimated water usage is within a relatively lower range."
        )

    score = max(0, min(100, score))

    # -----------------------------
    # Results
    # -----------------------------
    st.header("📊 Sustainability Assessment")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            "Sustainability Score",
            f"{score}/100"
        )

    with result_col2:
        if score >= 80:
            status = "🌱 Good"
        elif score >= 60:
            status = "🟡 Moderate"
        else:
            status = "🔴 Needs Improvement"

        st.metric(
            "Overall Status",
            status
        )

    st.progress(score / 100)

    st.subheader("💡 Recommendations")

    for recommendation in recommendations:
        st.write(recommendation)

    st.divider()

    st.info(
        "🤖 AI Layer: In the next stage, these inputs and sustainability "
        "patterns will be passed through an AI workflow to generate "
        "context-aware recommendations and explanations."
    )

# -----------------------------
# Responsible AI
# -----------------------------
st.divider()

# -----------------------------
# Responsible AI
# -----------------------------

st.divider()

st.header("🛡️ Responsible AI")

st.markdown(
    """
### ⚖️ Fairness

EcoSense AI does not use sensitive personal characteristics such as
religion, ethnicity, gender, or income to classify sustainability
profiles. Recommendations are based on the sustainability inputs
provided by the user.

### 🔎 Transparency

The system displays the major factors associated with its assessment
so that users can understand why a sustainability category was produced.

### 🧭 Ethics

EcoSense AI provides sustainability guidance rather than claiming to
make scientifically definitive judgments about an individual's
environmental impact.

### 🔐 Privacy

The prototype does not require users to provide personally identifiable
or sensitive information.

### ⚠️ Model Limitation

The current machine-learning model was trained using a synthetic dataset
created for prototype development. Its predictions should therefore not
be treated as scientifically validated environmental measurements.
"""
)