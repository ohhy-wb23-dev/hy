import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Garment Productivity Predictor",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# LOAD ASSETS
# ==========================================
@st.cache_resource
def load_assets():
    model = joblib.load("gbm_model.pkl")
    model_columns = joblib.load("gbm_model_columns.pkl")
    return model, model_columns

try:
    model, model_columns = load_assets()
except Exception:
    st.error("Error: Model assets (gbm_model.pkl / gbm_model_columns.pkl) missing.")
    st.stop()

# Prediction Labels
LABELS = {0: "Low", 1: "Moderate", 2: "High"}
LABEL_EMOJI = {"Low": "🔴", "Moderate": "🟡", "High": "🟢"}

# ==========================================
# HELPERS & SCALING LOGIC
# ==========================================
# Scaling parameters derived from final_classification_dataset.csv
OVERTIME_MEAN = 4567.46
OVERTIME_STD = 3348.82

def build_model_input(day, quarter, dept, team, wip, workers, style_change, smv, incentive, overtime_raw, idle_time, idle_men):
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)
    
    # Calculate Z-Score for overtime as expected by the model
    overtime_scaled = (overtime_raw - OVERTIME_MEAN) / OVERTIME_STD

    numeric_features = {
        "team": team,
        "smv": smv,
        "wip": wip,
        "incentive": incentive,
        "idle_time": idle_time,
        "idle_men": idle_men,
        "no_of_workers": workers,
        "over_time_scaled": overtime_scaled,
    }

    for col, val in numeric_features.items():
        if col in input_df.columns:
            input_df.at[0, col] = val

    # Categorical Mapping (matches dataset: sewing, finished)
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

# ==========================================
# MAIN UI
# ==========================================
st.title("🏭 Garment Productivity Predictor")
st.caption("Operational Analytics Engine - Refined Dataset Integration")

# Preset data matching the new ranges
d = {"day": "Monday", "quarter": "Quarter1", "dept": "sewing", "team": 1, "wip": 500, "workers": 30, "style": 0, "smv": 20.0, "inc": 50, "ot_raw": 500, "it": 0, "im": 0}

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📅 Context")
    day = st.selectbox("Working Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])
    quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
    dept = st.radio("Department", ["sewing", "finished"], horizontal=True)
    team = st.number_input("Team ID", 1, 12, d["team"])

with col2:
    st.subheader("⚙️ Resources")
    wip = st.number_input("WIP (Items)", 0.0, 2698.0, float(d["wip"]), help="Work in Progress (Max: 2698)")
    workers = st.number_input("Total Workers", 2.0, 89.0, float(d["workers"]), help="Labor Force (Range: 2-89)")
    style_change = st.selectbox("Style Changes", [0, 1, 2], index=d["style"])
    smv = st.number_input("SMV (Complexity)", 2.9, 54.56, float(d["smv"]), format="%.2f", help="Standard Minute Value (Range: 2.9-54.6)")

with col3:
    st.subheader("💰 Performance")
    incentive = st.number_input("Incentive (BDT)", 0, 3600, d["inc"], help="Incentive Pay (Max: 3600)")
    # REFINED: Overtime range set to 1,000 as requested
    overtime_raw = st.slider("Overtime (Minutes)", 0, 1000, d["ot_raw"], help="Operational Overtime. Scaled internally for the model.")
    idle_time = st.number_input("Idle Time (Min)", 0.0, 300.0, float(d["it"]))
    idle_men = st.number_input("Idle Workers", 0, 45, d["im"])

if st.button("🚀 Generate Prediction", use_container_width=True, type="primary"):
    input_data = build_model_input(day, quarter, dept, team, wip, workers, style_change, smv, incentive, overtime_raw, idle_time, idle_men)
    
    pred_idx = int(model.predict(input_data)[0])
    probs = model.predict_proba(input_data)[0]
    result = LABELS[pred_idx]

    st.success(f"### {LABEL_EMOJI[result]} Predicted Productivity: {result}")
    st.metric("Model Confidence", f"{probs[pred_idx]:.2%}")
    
    # Show breakdown
    cols = st.columns(3)
    for i, label in enumerate(["Low", "Moderate", "High"]):
        cols[i].write(f"**{label}**")
        cols[i].progress(float(probs[i]))

with st.expander("Technical: View Encoded Feature Vector (1x192)"):
    if 'input_data' in locals():
        st.dataframe(input_data)
