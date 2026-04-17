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
.main-title { font-size: 2.1rem; font-weight: 800; color: #1e293b; margin-bottom: 0.2rem; }
.sub-text { color: #64748b; margin-bottom: 1.5rem; font-style: italic; }
.block-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; margin-bottom: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.result-card { border-radius: 16px; padding: 25px; border: 1px solid #cbd5e1; background: #f1f5f9; }
</style>
""", unsafe_allow_html=True)

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
except Exception as e:
    st.error("Missing model assets. Please check gbm_model.pkl and gbm_model_columns.pkl.")
    st.stop()

# Constants
LABELS = {0: "Low", 1: "Moderate", 2: "High"}
LABEL_EMOJI = {"Low": "🔴", "Moderate": "🟡", "High": "🟢"}
LABEL_TEXT = {
    "Low": "Productivity is significantly below target. Immediate intervention required.",
    "Moderate": "Productivity is stable but shows room for optimization.",
    "High": "Optimal operating conditions. Productivity exceeds performance benchmarks."
}

# ==========================================
# HELPERS
# ==========================================
def build_model_input(day, quarter, dept, team, wip, workers, style_change, smv, incentive, overtime, idle_time, idle_men):
    # Initialize with 192 columns (zeros)
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)

    # Numeric Feature Mapping
    numeric_map = {
        "team": team,
        "smv": smv,
        "wip": wip,
        "incentive": incentive,
        "idle_time": idle_time,
        "idle_men": idle_men,
        "no_of_workers": workers,
        "over_time_scaled": overtime,
    }

    for col, val in numeric_map.items():
        if col in input_df.columns:
            input_df.at[0, col] = val

    # Categorical Feature Mapping (One-Hot Logic)
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
# SIDEBAR & PRESETS
# ==========================================
st.sidebar.header("📊 Simulation Controls")
preset = st.sidebar.selectbox("Choose Scenario", ["Custom", "Average Baseline", "High Workload", "Resource Interruption"])

# Refined Preset Data based on your new ranges
preset_logic = {
    "Custom": {"team": 6, "wip": 500, "workers": 30, "style": 0, "smv": 22.0, "inc": 50, "ot": 0.0, "it": 0, "im": 0},
    "Average Baseline": {"team": 1, "wip": 1190, "workers": 34, "style": 0, "smv": 15.0, "inc": 38, "ot": 0.0, "it": 0, "im": 0},
    "High Workload": {"team": 5, "wip": 2500, "workers": 55, "style": 1, "smv": 45.0, "inc": 100, "ot": 1.2, "it": 0, "im": 0},
    "Resource Interruption": {"team": 12, "wip": 500, "workers": 15, "style": 0, "smv": 10.0, "inc": 0, "ot": -0.8, "it": 120, "im": 10}
}
d = preset_logic[preset]

# ==========================================
# MAIN UI
# ==========================================
st.markdown('<div class="main-title">🏭 Garment Productivity Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Advanced Analytics for Operational Performance (192-Feature GBM Model)</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Dashboard", "Model Specification"])

with tab1:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("📅 Context")
        day = st.selectbox("Working Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])
        quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
        dept = st.radio("Dept", ["Sewing", "Finishing"], horizontal=True)
        team = st.number_input("Team ID", 1, 12, d["team"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("⚙️ Resources")
        wip = st.number_input("WIP (Items)", 0, 23122, d["wip"], help="Work in Progress")
        workers = st.number_input("Team Size", 2, 90, d["workers"], help="Total labor assigned")
        style_change = st.selectbox("Style Changes", [0, 1, 2], index=d["style"])
        smv = st.number_input("SMV (Minutes)", 2.9, 55.0, d["smv"], help="Time allocated per task")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("💰 Performance")
        incentive = st.number_input("Incentive (BDT)", 0, 3600, d["inc"], help="Financial motivation")
        overtime = st.slider("Overtime (Scaled)", -2.0, 2.0, d["ot"], help="Standardized Z-Score: 0 is the factory mean")
        idle_time = st.number_input("Idle Time (Min)", 0, 300, d["it"], help="Production interruption duration")
        idle_men = st.number_input("Idle Workers", 0, 45, d["im"], help="Labor affected by downtime")
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Generate Forecast", use_container_width=True, type="primary"):
        input_data = build_model_input(day, quarter, dept, team, wip, workers, style_change, smv, incentive, overtime, idle_time, idle_men)
        
        # Inference
        pred_idx = int(model.predict(input_data)[0])
        probs = model.predict_proba(input_data)[0]
        result = LABELS[pred_idx]
        
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        res_col1, res_col2 = st.columns([3, 1])
        with res_col1:
            st.markdown(f"## {LABEL_EMOJI[result]} {result} Productivity Predicted")
            st.write(f"**Status:** {LABEL_TEXT[result]}")
        with res_col2:
            st.metric("Confidence", f"{probs[pred_idx]:.2%}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Probability Breakdown
        st.markdown("### 🔍 Probability Distribution")
        cols = st.columns(3)
        for i, label in enumerate(["Low", "Moderate", "High"]):
            with cols[i]:
                st.write(f"**{label}**")
                st.progress(float(probs[i]))
                st.caption(f"{probs[i]:.1%}")

with tab2:
    st.subheader("Feature Vector Structure")
    st.write(f"The model processes a total of **{len(model_columns)}** unique features.")
    st.info("Scaling Reference: Overtime is normalized using a Z-score transformation where 0 represents the average factory output.")
    with st.expander("Show Active Encoded Features"):
        st.dataframe(input_data)
