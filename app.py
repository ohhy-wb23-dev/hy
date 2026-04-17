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
# CUSTOM STYLING (Professional/Academic)
# ==========================================
st.markdown("""
<style>
.main-title { font-size: 2.2rem; font-weight: 800; color: #1e293b; margin-bottom: 0.2rem; }
.sub-text { color: #64748b; margin-bottom: 1.5rem; font-style: italic; }
.block-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 22px; margin-bottom: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.result-card { border-radius: 16px; padding: 25px; border: 1px solid #cbd5e1; background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); }
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD ASSETS
# ==========================================
@st.cache_resource
def load_assets():
    # These must match your 192-feature model files
    model = joblib.load("gbm_model.pkl")
    model_columns = joblib.load("gbm_model_columns.pkl")
    return model, model_columns

try:
    model, model_columns = load_assets()
except Exception as e:
    st.error("Critical Error: Model assets (gbm_model.pkl / gbm_model_columns.pkl) not found in directory.")
    st.stop()

# Constants for Prediction Output
LABELS = {0: "Low", 1: "Moderate", 2: "High"}
LABEL_EMOJI = {"Low": "🔴", "Moderate": "🟡", "High": "🟢"}
LABEL_TEXT = {
    "Low": "Productivity is significantly below target. Immediate resource reallocation or process intervention recommended.",
    "Moderate": "Productivity is within stable operating parameters. Monitoring for optimization opportunities is advised.",
    "High": "Optimal production efficiency achieved. Current workflow exceeds performance benchmarks."
}

# ==========================================
# CORE LOGIC: FEATURE MAPPING (192-COLUMNS)
# ==========================================
def build_model_input(day, quarter, dept, team, wip, workers, style_change, smv, incentive, overtime, idle_time, idle_men):
    # Initialize a zero-filled DataFrame with exactly 192 columns from the training set
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)

    # 1. Map Numeric Features (Matching column names in training)
    numeric_features = {
        "team": team,
        "smv": smv,
        "wip": wip,
        "incentive": incentive,
        "idle_time": idle_time,
        "idle_men": idle_men,
        "no_of_workers": workers,
        "over_time_scaled": overtime, # Z-score scaled value
    }

    for col, val in numeric_features.items():
        if col in input_df.columns:
            input_df.at[0, col] = val

    # 2. Map Categorical Features (One-Hot Encoding format)
    # Ensuring lowercase and exact string matches for 'sewing'/'finished'
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
# SIDEBAR CONTROLS & PRESETS
# ==========================================
st.sidebar.title("Simulation Controls")
preset = st.sidebar.selectbox("Load Scenario Preset", ["Custom Input", "Baseline (Standard)", "High Pressure", "Downtime Event"])

# Presets derived from Dataset Statistics
preset_logic = {
    "Custom Input": {"day": "Monday", "quarter": "Quarter1", "dept": "sewing", "team": 6, "wip": 500, "workers": 30, "style": 0, "smv": 22.0, "inc": 50, "ot": 0.0, "it": 0, "im": 0},
    "Baseline (Standard)": {"day": "Thursday", "quarter": "Quarter1", "dept": "sewing", "team": 1, "wip": 1100, "workers": 34, "style": 0, "smv": 15.0, "inc": 40, "ot": 0.0, "it": 0, "im": 0},
    "High Pressure": {"day": "Wednesday", "quarter": "Quarter3", "dept": "sewing", "team": 5, "wip": 2000, "workers": 60, "style": 1, "smv": 45.0, "inc": 100, "ot": 1.2, "it": 0, "im": 0},
    "Downtime Event": {"day": "Saturday", "quarter": "Quarter4", "dept": "finished", "team": 10, "wip": 400, "workers": 12, "style": 0, "smv": 10.0, "inc": 0, "ot": -0.8, "it": 150, "im": 20}
}
d = preset_logic[preset]

# ==========================================
# MAIN INTERFACE
# ==========================================
st.markdown('<div class="main-title">🏭 Garment Productivity Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Predictive Operational Analytics for Garment Manufacturing Optimization</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📊 Prediction Dashboard", "🔬 Technical Specifications"])

with tab1:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("📅 Contextual Data")
        day = st.selectbox("Working Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"], index=0)
        quarter = st.selectbox("Fiscal Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"], index=0)
        dept = st.radio("Department", ["sewing", "finished"], horizontal=True, help="Note: 'finished' represents the Finishing/QC department in the dataset.")
        team = st.number_input("Team Identification", 1, 12, d["team"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("⚙️ Resource Allocation")
        # Ranges precisely matched to final_classification_dataset.csv
        wip = st.number_input("Work In Progress (WIP)", 0.0, 2698.0, float(d["wip"]), step=1.0, help="Count of unfinished items. Max observed: 2,698.")
        workers = st.number_input("Labor Force (Workers)", 2.0, 89.0, float(d["workers"]), step=1.0, help="Total workers per team. Range: 2 - 89.")
        style_change = st.select_slider("No. of Style Changes", options=[0, 1, 2], value=d["style"])
        smv = st.number_input("SMV (Complexity)", 2.9, 54.56, float(d["smv"]), format="%.2f", help="Standard Minute Value: Time required to complete a task.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("💰 Incentive & Downtime")
        incentive = st.number_input("Incentive (BDT)", 0, 3600, d["inc"], help="Performance-based financial incentives.")
        overtime = st.slider("Overtime (Scaled Z-Score)", -2.0, 2.0, d["ot"], help="Standardized overtime value where 0.0 is the mean.")
        idle_time = st.number_input("Idle Time (Minutes)", 0.0, 300.0, float(d["it"]), help="Time production was stopped.")
        idle_men = st.number_input("Idle Workers Count", 0, 45, d["im"], help="Number of workers idle during downtime.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    if st.button("🚀 Execute Productivity Forecast", use_container_width=True, type="primary"):
        with st.spinner('Processing 192-feature vector through GBM...'):
            input_vector = build_model_input(day, quarter, dept, team, wip, workers, style_change, smv, incentive, overtime, idle_time, idle_men)
            
            # Prediction Inference
            pred_idx = int(model.predict(input_vector)[0])
            probs = model.predict_proba(input_vector)[0]
            result = LABELS[pred_idx]
            confidence = float(probs[pred_idx])

            # Results Presentation
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            res_col1, res_col2 = st.columns([2, 1])
            with res_col1:
                st.markdown(f"## {LABEL_EMOJI[result]} {result} Productivity")
                st.info(LABEL_TEXT[result])
            with res_col2:
                st.metric("Model Confidence", f"{confidence:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)

            # Probability Breakdown
            st.write("---")
            st.subheader("🔍 Probabilistic Distribution")
            p_cols = st.columns(3)
            for i, label in enumerate(["Low", "Moderate", "High"]):
                with p_cols[i]:
                    st.write(f"**{label}**")
                    st.progress(float(probs[i]))
                    st.caption(f"{probs[i]:.1%}")

with tab2:
    st.subheader("Architecture & Data Analysis")
    st.write("This application leverages a **Gradient Boosting Machine (GBM)** trained on 1,197 operational records.")
    
    st.markdown("#### Operational Ranges Used (Verified)")
    range_df = pd.DataFrame({
        "Feature": ["WIP (Items)", "Labor Force", "SMV (Min)", "Incentive (BDT)", "Idle Time (Min)"],
        "Minimum": [0.0, 2.0, 2.9, 0, 0.0],
        "Maximum": [2698.0, 89.0, 54.56, 3600, 300.0]
    })
    st.table(range_df)

    with st.expander("View Active 192-Feature Encoded Vector"):
        if 'input_vector' in locals():
            st.dataframe(input_vector)
        else:
            st.info("Perform a prediction to see the encoded feature mapping.")
