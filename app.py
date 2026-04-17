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
# CUSTOM STYLING
# ==========================================
st.markdown("""
<style>
.main-title { font-size: 2.2rem; font-weight: 800; color: #1e293b; margin-bottom: 0.2rem; }
.sub-text { color: #64748b; margin-bottom: 1.5rem; }
.block-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 22px; margin-bottom: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.result-card { border-radius: 16px; padding: 25px; border: 1px solid #cbd5e1; background: #f8fafc; }
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
except Exception:
    st.error("Missing model files (gbm_model.pkl / gbm_model_columns.pkl).")
    st.stop()

# Output Labels
LABELS = {0: "Low", 1: "Moderate", 2: "High"}
LABEL_EMOJI = {"Low": "🔴", "Moderate": "🟡", "High": "🟢"}

# ==========================================
# SCALING CONSTANTS (From Dataset Stats)
# ==========================================
# Derived from final_classification_dataset.csv
OVERTIME_MEAN = 4567.46
OVERTIME_STD = 3348.82

def build_model_input(day, quarter, dept, team, wip, workers, style_change, smv, incentive, ot_raw, idle_time, idle_men):
    input_df = pd.DataFrame(0, index=[0], columns=model_columns) # 192 features

    # Model requires 'over_time_scaled' (Z-score)
    ot_scaled = (ot_raw - OVERTIME_MEAN) / OVERTIME_STD

    numeric_map = {
        "team": team,
        "smv": smv,
        "wip": wip,
        "incentive": incentive,
        "idle_time": idle_time,
        "idle_men": idle_men,
        "no_of_workers": workers,
        "over_time_scaled": ot_scaled,
    }

    for col, val in numeric_map.items():
        if col in input_df.columns:
            input_df.at[0, col] = val

    # Categorical One-Hot Encoding
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
st.markdown('<div class="main-title">🏭 Garment Productivity Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Dataset-Validated Operational Analysis Prototype</div>', unsafe_allow_html=True)

# Default values based on dataset medians
d = {"day": "Monday", "quarter": "Quarter1", "dept": "sewing", "team": 1, "wip": 500, "workers": 34, "style": 0, "smv": 15.0, "inc": 40, "ot_raw": 500}

tab1, tab2 = st.tabs(["📊 Prediction Dashboard", "⚙️ Statistics"])

with tab1:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("📅 Context")
        day = st.selectbox("Working Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])
        quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
        dept = st.radio("Department", ["sewing", "finished"], horizontal=True)
        team = st.number_input("Team ID", 1, 12, d["team"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("⚙️ Resources")
        # CORRECTED RANGES
        wip = st.number_input("WIP (Items)", 0.0, 2698.0, float(d["wip"]), help="Max observed: 2,698")
        workers = st.number_input("Total Workers", 2.0, 89.0, float(d["workers"]), help="Range: 2 - 89")
        style_change = st.selectbox("Style Changes", [0, 1, 2], index=d["style"])
        smv = st.number_input("SMV (Complexity)", 2.9, 54.56, float(d["smv"]), format="%.2f")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("💰 Metrics")
        incentive = st.number_input("Incentive (BDT)", 0, 3600, d["inc"], help="Max observed: 3,600")
        # CORRECTED: Range up to 1,000
        ot_raw = st.slider("Overtime (Minutes)", 0, 25920, d["ot_raw"])
        idle_time = st.number_input("Idle Time (Min)", 0.0, 300.0, 0.0)
        idle_men = st.number_input("Idle Workers", 0, 45, 0)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🚀 Run Forecast", use_container_width=True, type="primary"):
        input_data = build_model_input(day, quarter, dept, team, wip, workers, style_change, smv, incentive, ot_raw, 0.0, 0)
        
        pred_idx = int(model.predict(input_data)[0])
        probs = model.predict_proba(input_data)[0]
        result = LABELS[pred_idx]

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f"## {LABEL_EMOJI[result]} {result} Productivity")
        st.metric("Model Confidence", f"{probs[pred_idx]:.2%}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### Probability Distribution")
        p_cols = st.columns(3)
        for i, label in enumerate(["Low", "Moderate", "High"]):
            p_cols[i].write(f"**{label}**")
            p_cols[i].progress(float(probs[i]))

with tab2:
    st.write("### Data Range Validation")
    st.table(pd.DataFrame({
        "Parameter": ["WIP", "Incentive", "SMV", "Workers", "Idle Time", "Overtime"],
        "Corrected Max": [2698.0, 3600, 54.56, 89.0, 300.0, 25920.0] # Added Overtime here
    }))
