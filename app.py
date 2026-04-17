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
.sub-text { color: #64748b; margin-bottom: 1.5rem; }
.block-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; margin-bottom: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
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
    st.error("Model assets not found.")
    st.stop()

# Labels
LABELS = {0: "Low", 1: "Moderate", 2: "High"}
LABEL_EMOJI = {"Low": "🔴", "Moderate": "🟡", "High": "🟢"}
LABEL_TEXT = {
    "Low": "Productivity is likely below target. Intervention recommended.",
    "Moderate": "Productivity is stable. Standard operating conditions.",
    "High": "High productivity is likely. Optimal performance detected."
}

# ==========================================
# HELPERS
# ==========================================
def build_model_input(day, quarter, dept, team, wip, workers, style_change, smv, incentive, overtime, idle_time, idle_men):
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

    for col, val in numeric_features.items():
        if col in input_df.columns:
            input_df.at[0, col] = val

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

# ==========================================
# PRESETS & UI
# ==========================================
# Updated presets based on dataset stats
preset_data = {
    "Custom Input": {"day": "Monday", "quarter": "Quarter1", "dept": "Sewing", "team": 6, "wip": 500, "workers": 30, "style": 0, "smv": 22.0, "inc": 50, "ot": 0.0, "it": 0, "im": 0},
    "Balanced Setup": {"day": "Tuesday", "quarter": "Quarter2", "dept": "Sewing", "team": 4, "wip": 1000, "workers": 35, "style": 0, "smv": 15.0, "inc": 40, "ot": 0.0, "it": 0, "im": 0},
    "High Intensity": {"day": "Wednesday", "quarter": "Quarter3", "dept": "Sewing", "team": 2, "wip": 200, "workers": 60, "style": 1, "smv": 45.0, "inc": 150, "ot": 1.5, "it": 0, "im": 0}
}

st.sidebar.title("App Controls")
preset = st.sidebar.selectbox("Quick Scenario", list(preset_data.keys()))
d = preset_data[preset]

st.markdown('<div class="main-title">🏭 Garment Productivity Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Optimized using actual data distributions from <i>final_classification_dataset.csv</i></div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Prediction Dashboard", "Technical Data View"])

with tab1:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("📅 Context")
        day = st.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"], index=0)
        quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"], index=0)
        dept = st.radio("Department", ["Sewing", "Finished"], help="Updated to match dataset classes.")
        team = st.slider("Team Number", 1, 12, d["team"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("⚙️ Resources")
        # Ranges refined based on dataset analysis
        wip = st.number_input("WIP (Items)", 0, 2698, d["wip"], help="Max observed: 2,698")
        workers = st.number_input("Total Workers", 2, 89, d["workers"], help="Range: 2 to 89")
        style_change = st.selectbox("Style Changes", [0, 1, 2], index=d["style"])
        smv = st.number_input("SMV (Complexity)", 2.9, 54.6, d["smv"], format="%.2f")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("💰 Metrics")
        incentive = st.number_input("Incentive (BDT)", 0, 3600, d["inc"])
        overtime = st.slider("Overtime (Scaled)", -2.0, 2.0, d["ot"], help="Normalized value (Z-Score)")
        idle_time = st.number_input("Idle Time (Min)", 0, 300, d["it"])
        idle_men = st.number_input("Idle Workers", 0, 45, d["im"])
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Generate Forecast", use_container_width=True, type="primary"):
        input_data = build_model_input(day, quarter, dept, team, wip, workers, style_change, smv, incentive, overtime, idle_time, idle_men)
        
        pred_idx = int(model.predict(input_data)[0])
        probs = model.predict_proba(input_data)[0]
        result = LABELS[pred_idx]

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f"## {LABEL_EMOJI[result]} {result} Productivity")
        st.write(LABEL_TEXT[result])
        st.metric("Model Confidence", f"{probs[pred_idx]:.2%}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### Probability Distribution")
        p_cols = st.columns(3)
        for i, label in enumerate(["Low", "Moderate", "High"]):
            with p_cols[i]:
                st.write(f"**{label}**")
                st.progress(float(probs[i]))

with tab2:
    st.subheader("Input Feature Vector")
    st.write("This table shows the 192-feature vector currently being processed by the GBM model.")
    if 'input_data' in locals():
        st.dataframe(input_data)
    else:
        st.info("Run a prediction to see the vector data.")
