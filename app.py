import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Garment Productivity Predictor",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM STYLING
# ==========================================
st.markdown("""
<style>
.main-title { font-size: 2.1rem; font-weight: 800; margin-bottom: 0.2rem; }
.sub-text { color: #6b7280; margin-bottom: 1.2rem; }
.block-card { background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 16px; padding: 16px 18px; margin-bottom: 12px; }
.result-card { border-radius: 18px; padding: 22px; border: 1px solid #e5e7eb; background: linear-gradient(135deg, #ffffff, #f8fafc); }
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD ASSETS
# ==========================================
@st.cache_resource
def load_assets():
    # Ensure these filenames match your uploaded files exactly
    model = joblib.load("gbm_model.pkl")
    model_columns = joblib.load("gbm_model_columns.pkl")
    return model, model_columns

@st.cache_data
def load_reference_data():
    try:
        # Note the extra space in the filename from your zip: "cleaned_garments_worker_productivity .csv.csv"
        df = pd.read_csv("cleaned_garments_worker_productivity .csv.csv")
        return df
    except Exception:
        return None

try:
    model, model_columns = load_assets()
except Exception as e:
    st.error("Model files not found. Check if 'gbm_model.pkl' and 'gbm_model_columns.pkl' are in the same folder.")
    st.stop()

reference_df = load_reference_data()

# Constants
LABELS = {0: "Low", 1: "Moderate", 2: "High"}
LABEL_EMOJI = {"Low": "🔴", "Moderate": "🟡", "High": "🟢"}
LABEL_TEXT = {
    "Low": "Low productivity is likely. Risk of underperformance detected.",
    "Moderate": "Moderate productivity is likely. Operations appear stable.",
    "High": "High productivity is likely. Strong operating condition."
}

# ==========================================
# HELPERS
# ==========================================
def build_model_input(day, quarter, dept, team, wip, workers, style_change, smv, incentive, overtime, idle_time, idle_men):
    # CRITICAL BUG FIX: Initialize with zeros matching model_columns length
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)

    numeric_features = {
        "team": team,
        "smv": smv,
        "wip": wip,
        "incentive": incentive,
        "idle_time": idle_time,
        "idle_men": idle_men,
        "no_of_workers": workers,
        "over_time_scaled": overtime,
    }

    # Fill numeric columns if they exist in the model
    for col, val in numeric_features.items():
        if col in input_df.columns:
            input_df.at[0, col] = val

    # categorical mapping - ensuring strings match the One-Hot encoding format
    # Fixed: Force style_change to int to prevent "0.0" string bugs
    cat_cols = [
        f"quarter_{quarter}",
        f"department_{dept.lower()}",
        f"day_{day}",
        f"no_of_style_change_{int(style_change)}" 
    ]

    for col in cat_cols:
        if col in input_df.columns:
            input_df.at[0, col] = 1

    return input_df[model_columns]

def get_reference_ranges(df):
    if df is None:
        return {
            "wip": (0, 23122), "incentive": (0, 3600), "idle_time": (0, 300),
            "idle_men": (0, 45), "no_of_workers": (2, 90), "smv": (2.9, 54.6),
            "team": (1, 12), "over_time_scaled": (-2.0, 2.0)
        }
    ranges = {}
    for col in ["wip", "incentive", "idle_time", "idle_men", "no_of_workers", "smv", "team", "over_time_scaled"]:
        if col in df.columns:
            ranges[col] = (float(df[col].min()), float(df[col].max()))
    return ranges

# ==========================================
# SIDEBAR & PRESETS
# ==========================================
ranges = get_reference_ranges(reference_df)
st.sidebar.title("Prototype Controls")

preset = st.sidebar.selectbox("Quick preset", ["Custom Input", "Balanced Setup", "High Performance", "Risky Setup"])

# Preset Logic
preset_data = {
    "Custom Input": {"day": "Monday", "quarter": "Quarter1", "dept": "Sewing", "team": 6, "wip": 500, "workers": 30, "style": 0, "smv": 22.0, "inc": 50, "ot": 0.0, "it": 0, "im": 0},
    "Balanced Setup": {"day": "Tuesday", "quarter": "Quarter2", "dept": "Sewing", "team": 6, "wip": 600, "workers": 34, "style": 0, "smv": 15.0, "inc": 30, "ot": 0.0, "it": 0, "im": 0},
    "High Performance": {"day": "Wednesday", "quarter": "Quarter3", "dept": "Sewing", "team": 5, "wip": 300, "workers": 50, "style": 0, "smv": 10.0, "inc": 200, "ot": 0.2, "it": 0, "im": 0},
    "Risky Setup": {"day": "Thursday", "quarter": "Quarter4", "dept": "Finishing", "team": 10, "wip": 2000, "workers": 15, "style": 2, "smv": 35.0, "inc": 0, "ot": -0.5, "it": 40, "im": 5}
}
d = preset_data[preset]

# ==========================================
# MAIN UI
# ==========================================
st.markdown('<div class="main-title">🏭 Garment Productivity Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Gradient Boosting Analysis Prototype</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Prediction Dashboard", "About"])

with tab1:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("📅 Context")
        day = st.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"], index=0)
        quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"], index=0)
        dept = st.selectbox("Department", ["Sewing", "Finishing"])
        team = st.slider("Team", 1, 12, d["team"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("⚙️ Resources")
        wip = st.number_input("WIP", 0, 23122, d["wip"])
        workers = st.number_input("Workers", 2, 90, d["workers"])
        style_change = st.selectbox("Style Changes", [0, 1, 2], index=d["style"])
        smv = st.number_input("SMV (Complexity)", 2.0, 55.0, d["smv"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("💰 Metrics")
        incentive = st.number_input("Incentive", 0, 3600, d["inc"])
        overtime = st.slider("Overtime (Scaled)", -2.0, 2.0, d["ot"])
        idle_time = st.number_input("Idle Time", 0, 300, d["it"])
        idle_men = st.number_input("Idle Workers", 0, 45, d["im"])
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Generate Forecast", use_container_width=True, type="primary"):
        encoded_df = build_model_input(day, quarter, dept, team, wip, workers, style_change, smv, incentive, overtime, idle_time, idle_men)
        
        # Prediction
        pred_idx = int(model.predict(encoded_df)[0])
        probs = model.predict_proba(encoded_df)[0]
        result = LABELS[pred_idx]
        confidence = float(probs[pred_idx])

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f"## {LABEL_EMOJI[result]} {result} Productivity")
        st.write(LABEL_TEXT[result])
        st.metric("Model Confidence", f"{confidence:.2%}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Probabilities
        st.markdown("### Probability Breakdown")
        for i, label in enumerate(["Low", "Moderate", "High"]):
            st.write(f"**{label}**")
            st.progress(float(probs[i]))

with tab2:
    st.write("This upgraded version fixes the categorical encoding bug by strictly matching the training column names.")
