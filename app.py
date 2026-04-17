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
# LOAD ASSETS & SCALING CONSTANTS
# =========================================================
# Derived from final_classification_dataset.csv statistics
OVERTIME_MEAN = 4567.46
OVERTIME_STD = 3348.82

@st.cache_resource
def load_model_assets():
    # Loading the 192-feature GBM model and columns
    model = joblib.load("gbm_model.pkl")
    model_columns = joblib.load("gbm_model_columns.pkl")
    return model, model_columns

try:
    model, model_columns = load_model_assets()
except Exception:
    st.error("Critical Error: Missing gbm_model.pkl or gbm_model_columns.pkl.")
    st.stop()

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def build_gbm_input(day, quarter, dept, team, wip, workers, style_change, smv, incentive, ot_raw, it, im):
    # Initialize 192 columns with zeros to match training
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)

    # Standardizing the overtime input for the model
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

    for col, val in numeric_map.items():
        if col in input_df.columns:
            input_df.at[0, col] = val

    # Categorical One-Hot Mapping
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
# MAIN UI
# =========================================================
st.markdown("## 🏭 Productivity Forecasting Dashboard")
st.info("The system analyzes 192 operational features to predict performance levels.")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📅 Context")
    day = st.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])
    quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
    dept = st.radio("Department", ["sewing", "finished"], horizontal=True)
    team = st.number_input("Team ID", 1, 12, 1)

with col2:
    st.subheader("⚙️ Resources")
    # Ranges aligned with final_classification_dataset.csv
    wip = st.number_input("WIP (Items)", 0.0, 2698.0, 500.0)
    workers = st.number_input("Workers", 2.0, 89.0, 34.0)
    style_change = st.selectbox("Style Changes", [0, 1, 2])
    smv = st.number_input("SMV (Complexity)", 2.9, 54.56, 15.0)

with col3:
    st.subheader("💰 Metrics")
    incentive = st.number_input("Incentive (BDT)", 0, 3600, 40)
    # UPDATED: Range increased to 15,000 as requested
    ot_raw = st.slider("Overtime (Minutes)", 0, 15000, 4500, help="Operational range expanded to 15,000 minutes.")
    idle_time = st.number_input("Idle Time", 0.0, 300.0, 0.0)
    idle_men = st.number_input("Idle Workers", 0, 45, 0)

if st.button("🚀 Run Forecast", use_container_width=True, type="primary"):
    input_data = build_gbm_input(day, quarter, dept, team, wip, workers, style_change, smv, incentive, ot_raw, idle_time, idle_men)
    
    # Prediction
    pred_idx = int(model.predict(input_data)[0])
    probs = model.predict_proba(input_data)[0]
    result_map = {0: "Low", 1: "Moderate", 2: "High"}
    
    st.success(f"### Predicted Productivity: {result_map[pred_idx]}")
    st.metric("Model Confidence", f"{probs[pred_idx]:.2%}")
    
    with st.expander("View Input Feature Vector"):
        st.dataframe(input_data)
