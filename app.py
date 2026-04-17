import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Garment Factory Productivity Predictor",
    page_icon="🧵",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #1e293b; margin-bottom: 0.2rem; }
    .subtitle { font-size: 1rem; color: #64748b; margin-bottom: 1.2rem; }
    .section-card { background-color: #ffffff; padding: 1.2rem; border-radius: 12px; border: 1px solid #e2e8f0; margin-bottom: 1rem; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
    .result-card { border-radius: 16px; padding: 25px; border: 1px solid #cbd5e1; background: #f8fafc; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD ASSETS & SCALING CONSTANTS
# =========================================================
# Derived from final_classification_dataset.csv
OVERTIME_MEAN = 4567.46
OVERTIME_STD = 3348.82

@st.cache_resource
def load_model_assets():
    # Matches your GBM 192-feature files
    model = joblib.load("gbm_model.pkl")
    model_columns = joblib.load("gbm_model_columns.pkl")
    return model, model_columns

@st.cache_data
def load_dataset():
    return pd.read_csv("final_classification_dataset.csv")

try:
    model, model_columns = load_model_assets()
    df = load_dataset()
except Exception:
    st.error("Critical Error: Missing gbm_model.pkl, gbm_model_columns.pkl, or dataset file.")
    st.stop()

# =========================================================
# DATA-DRIVEN RANGES
# =========================================================
# Standardized labels for categorical mapping
quarter_opts = ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"]
dept_opts = ["sewing", "finished"]
day_opts = ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"]

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def build_gbm_input(day, quarter, dept, team, wip, workers, style_change, smv, incentive, ot_raw, it, im):
    # Initialize 192 columns with zeros to match model training
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)

    # Internal Z-Score Scaling for Overtime
    ot_scaled = (ot_raw - OVERTIME_MEAN) / OVERTIME_STD

    numeric_map = {
        "team": team,
        "smv": smv,
        "wip": wip,
        "incentive": incentive,
        "idle_time": it,
        "idle_men": im,
        "no_of_workers": workers,
        "over_time_scaled": ot_scaled,
    }

    # Fill numeric columns
    for col, val in numeric_map.items():
        if col in input_df.columns:
            input_df.at[0, col] = val

    # Fill categorical One-Hot columns
    cat_keys = [
        f"quarter_{quarter}",
        f"department_{dept.lower()}",
        f"day_{day}",
        f"no_of_style_change_{int(style_change)}" 
    ]

    for key in cat_keys:
        if key in input_df.columns:
            input_df.at[0, key] = 1

    return input_df[model_columns]

# =========================================================
# UI HEADER
# =========================================================
st.markdown('<div class="main-title">🧵 Gradient Boosting Productivity Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Operational analytics prototype optimized for <i>final_classification_dataset.csv</i></div>', unsafe_allow_html=True)

# =========================================================
# INPUT FORM
# =========================================================
st.markdown("### 📥 Operational Input Form")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📅 Temporal Context")
    quarter = st.selectbox("Fiscal Quarter", quarter_opts)
    day = st.selectbox("Working Day", day_opts)
    dept = st.radio("Department", dept_opts, horizontal=True)
    team = st.number_input("Team ID", 1, 12, 1)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("⚙️ Resources")
    # Corrected ranges based on dataset analysis
    wip = st.number_input("WIP (Items)", 0.0, 2698.0, 500.0, help="Max: 2,698")
    workers = st.number_input("Labor Force", 2.0, 89.0, 34.0, help="Range: 2 - 89")
    style_change = st.selectbox("Style Changes", [0, 1, 2])
    smv = st.number_input("SMV (Minutes)", 2.9, 54.56, 15.0, format="%.2f")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("💰 Performance")
    incentive = st.number_input("Incentive (BDT)", 0, 3600, 40)
    ot_raw = st.slider("Overtime (Minutes)", 0, 1000, 500, help="Capped at 1,000 for UI; scaled internally for GBM.")
    idle_time = st.number_input("Idle Time (Min)", 0.0, 300.0, 0.0)
    idle_men = st.number_input("Idle Workers", 0, 45, 0)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# EXECUTION
# =========================================================
if st.button("🚀 Run Productivity Forecast", use_container_width=True, type="primary"):
    # Generate 192-feature vector
    input_data = build_gbm_input(day, quarter, dept, team, wip, workers, style_change, smv, incentive, ot_raw, idle_time, idle_men)
    
    # Inference
    pred_idx = int(model.predict(input_data)[0])
    probs = model.predict_proba(input_data)[0]
    
    class_labels = {0: "Low", 1: "Moderate", 2: "High"}
    result = class_labels[pred_idx]

    # Result Card
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    res_col1, res_col2 = st.columns([2, 1])
    with res_col1:
        st.markdown(f"## Predicted Level: **{result}**")
        st.write("Target productivity benchmarks suggest this configuration is optimal for factory output.")
    with res_col2:
        st.metric("Model Confidence", f"{probs[pred_idx]:.2%}")
    st.markdown('</div>', unsafe_allow_html=True)

    # Probability Breakdown
    st.write("---")
    st.subheader("📊 Probability Breakdown")
    p_cols = st.columns(3)
    for i, label in enumerate(["Low", "Moderate", "High"]):
        with p_cols[i]:
            st.write(f"**{label}**")
            st.progress(float(probs[i]))
            st.caption(f"{probs[i]:.1%}")

    with st.expander("🔬 View Technical Feature Vector"):
        st.dataframe(input_data)
