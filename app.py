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
    # Ensure these filenames match your local files
    model = joblib.load("gbm_model.pkl")
    model_columns = joblib.load("gbm_model_columns.pkl")
    return model, model_columns

try:
    model, model_columns = load_assets()
except Exception:
    st.error("Critical Error: Missing 'gbm_model.pkl' or 'gbm_model_columns.pkl'.")
    st.stop()

# Output Mapping
LABELS = {0: "Low", 1: "Moderate", 2: "High"}
LABEL_EMOJI = {"Low": "🔴", "Moderate": "🟡", "High": "🟢"}
LABEL_DESC = {
    "Low": "Performance is significantly below target. Immediate resource intervention is required.",
    "Moderate": "Operations are stable. Productivity meets baseline expectations.",
    "High": "Optimal performance. The team is operating at peak efficiency."
}

# ==========================================
# SCALING CONSTANTS & INPUT BUILDER
# ==========================================
# Calculated from final_classification_dataset.csv statistics
OVERTIME_MEAN = 4567.46
OVERTIME_STD = 3348.82

def build_model_input(day, quarter, dept, team, wip, workers, style_change, smv, incentive, ot_raw, idle_time, idle_men):
    # Initialize zero vector for 192 features
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)

    # Internal Scaling: Convert raw minutes to Z-Score for the model
    ot_scaled = (ot_raw - OVERTIME_MEAN) / OVERTIME_STD

    # Map Numeric Features
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

    # Map Categorical Features (One-Hot)
    # Uses lowercase 'sewing' and 'finished' to match dataset
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
st.markdown('<div class="sub-text">Advanced Analytics Optimized for Industrial Efficiency</div>', unsafe_allow_html=True)

# Define a default state for inputs
d = {"day": "Monday", "quarter": "Quarter1", "dept": "sewing", "team": 1, "wip": 500, "workers": 30, "style": 0, "smv": 20.0, "inc": 50, "ot_raw": 500, "it": 0, "im": 0}

tab1, tab2 = st.tabs(["📊 Prediction Dashboard", "🔬 Data Context"])

with tab1:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("📅 Contextual Data")
        day = st.selectbox("Working Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])
        quarter = st.selectbox("Fiscal Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
        dept = st.radio("Department", ["sewing", "finished"], horizontal=True, help="Refined to match dataset categories.")
        team = st.number_input("Team Identification", 1, 12, d["team"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("⚙️ Resource Allocation")
        # Ranges based on final_classification_dataset.csv
        wip = st.number_input("WIP (Items)", 0.0, 2698.0, float(d["wip"]), step=1.0, help="Work In Progress (Max: 2,698)")
        workers = st.number_input("Labor Force (Workers)", 2.0, 89.0, float(d["workers"]), step=1.0, help="Team Size (Range: 2 - 89)")
        style_change = st.select_slider("Style Changes", options=[0, 1, 2], value=d["style"])
        smv = st.number_input("SMV (Complexity)", 2.9, 54.56, float(d["smv"]), format="%.2f", help="Standard Minute Value (Range: 2.9 - 54.56)")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("💰 Performance Metrics")
        incentive = st.number_input("Incentive (BDT)", 0, 3600, d["inc"], help="Performance pay (Max: 3,600)")
        # REQUESTED: Overtime range up to 1,000
        ot_raw = st.slider("Overtime (Minutes)", 0, 1000, d["ot_raw"], help="Operational minutes; scaled internally to Z-Score for model processing.")
        idle_time = st.number_input("Idle Time (Min)", 0.0, 300.0, float(d["it"]))
        idle_men = st.number_input("Idle Workers Count", 0, 45, d["im"])
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🚀 Execute Forecast", use_container_width=True, type="primary"):
        input_data = build_model_input(day, quarter, dept, team, wip, workers, style_change, smv, incentive, ot_raw, idle_time, idle_men)
        
        # Prediction
        pred_idx = int(model.predict(input_data)[0])
        probs = model.predict_proba(input_data)[0]
        result = LABELS[pred_idx]

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        r_col1, r_col2 = st.columns([3, 1])
        with r_col1:
            st.markdown(f"## {LABEL_EMOJI[result]} {result} Productivity")
            st.write(LABEL_DESC[result])
        with r_col2:
            st.metric("Model Confidence", f"{probs[pred_idx]:.2%}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Probabilities
        st.write("---")
        st.subheader("🔍 Probabilistic Distribution")
        p_cols = st.columns(3)
        for i, label in enumerate(["Low", "Moderate", "High"]):
            with p_cols[i]:
                st.write(f"**{label}**")
                st.progress(float(probs[i]))
                st.caption(f"{probs[i]:.1%}")

with tab2:
    st.subheader("Architecture Summary")
    st.write("Current operational ranges strictly aligned with `final_classification_dataset.csv` statistics:")
    
    analysis_df = pd.DataFrame({
        "Feature": ["WIP", "Workers", "SMV", "Incentive", "Overtime (Input)"],
        "Operational Range": ["0 - 2,698 Items", "2 - 89 Staff", "2.90 - 54.56 Min", "0 - 3,600 BDT", "0 - 1,000 Min"],
        "Internal Processing": ["Raw", "Raw", "Raw", "Raw", f"Z-Score (μ={OVERTIME_MEAN:.1f})"]
    })
    st.table(analysis_df)

    with st.expander("View Active 192-Feature Vector"):
        if 'input_data' in locals():
            st.dataframe(input_data)
        else:
            st.info("Initiate a forecast to visualize the encoded data mapping.")
