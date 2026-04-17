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
# CUSTOM STYLING (Refined for Presentation)
# ==========================================
st.markdown("""
<style>
.main-title { font-size: 2.3rem; font-weight: 800; color: #1e293b; margin-bottom: 0.5rem; }
.sub-text { color: #64748b; margin-bottom: 2rem; font-size: 1.1rem; }
.block-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 15px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.result-card {
    border-radius: 12px;
    padding: 25px;
    background: #f8fafc;
    border-left: 5px solid #3b82f6;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD ASSETS
# ==========================================
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("gbm_model.pkl")
        model_columns = joblib.load("gbm_model_columns.pkl")
        return model, model_columns
    except FileNotFoundError:
        st.error("Missing model files! Please upload 'gbm_model.pkl' and 'gbm_model_columns.pkl'.")
        st.stop()

@st.cache_data
def load_reference_data():
    try:
        return pd.read_csv("cleaned_garments_worker_productivity .csv.csv")
    except:
        return None

model, model_columns = load_assets()
reference_df = load_reference_data()

# Global Constants
LABELS = {0: "Low", 1: "Moderate", 2: "High"}
LABEL_EMOJI = {"Low": "🔴", "Moderate": "🟡", "High": "🟢"}

# ==========================================
# REFINED HELPERS
# ==========================================
def build_model_input(data_dict):
    """Builds a DataFrame that matches the exact training column structure."""
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)
    
    # 1. Fill numeric values
    for col in ["team", "smv", "wip", "incentive", "idle_time", "idle_men", "no_of_workers", "over_time_scaled"]:
        if col in input_df.columns:
            input_df.at[0, col] = data_dict.get(col, 0)
    
    # 2. Map Categorical (One-Hot)
    cat_mappings = [
        f"quarter_{data_dict['quarter']}",
        f"department_{data_dict['dept'].lower()}",
        f"day_{data_dict['day']}",
        f"no_of_style_change_{data_dict['style_change']}"
    ]
    
    for col in cat_mappings:
        if col in input_df.columns:
            input_df.at[0, col] = 1
            
    return input_df

# ==========================================
# UI LAYOUT
# ==========================================
st.markdown('<div class="main-title">🏭 Factory Productivity Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Gradient Boosting Engine for Garment Manufacturing Analysis</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📊 Prediction Engine", "📖 Model Information"])

with tab1:
    # Sidebar Presets
    st.sidebar.header("Controls")
    scenario = st.sidebar.selectbox("Load Scenario", ["Standard", "High Efficiency", "Risk Warning"])
    
    # Default values based on scenario
    defaults = {
        "Standard": {"wip": 500, "incentive": 50, "workers": 30, "overtime": 0.0},
        "High Efficiency": {"wip": 200, "incentive": 200, "workers": 50, "overtime": 0.5},
        "Risk Warning": {"wip": 2000, "incentive": 0, "workers": 15, "overtime": -0.5}
    }[scenario]

    # Input Columns
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("Context")
        day = st.selectbox("Working Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])
        dept = st.radio("Department", ["Sewing", "Finishing"])
        team = st.number_input("Team Number", 1, 12, 6)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("Resources")
        wip = st.number_input("WIP", 0, 25000, defaults["wip"])
        workers = st.slider("Worker Count", 2, 90, defaults["workers"])
        smv = st.number_input("SMV (Complexity)", 2.0, 60.0, 22.0)
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("Metrics")
        incentive = st.number_input("Incentive", 0, 4000, defaults["incentive"])
        overtime = st.slider("Overtime Scaled", -2.0, 2.0, defaults["overtime"])
        idle_time = st.number_input("Idle Time", 0, 300, 0)
        st.markdown('</div>', unsafe_allow_html=True)

    # PREDICTION LOGIC
    if st.button("🚀 Analyze Productivity", use_container_width=True, type="primary"):
        input_data = {
            "day": day, "quarter": "Quarter1", "dept": dept, "team": team,
            "wip": wip, "no_of_workers": workers, "style_change": 0, "smv": smv,
            "incentive": incentive, "over_time_scaled": overtime, "idle_time": idle_time, "idle_men": 0
        }
        
        X_input = build_model_input(input_data)
        
        # GBM Prediction
        pred_idx = model.predict(X_input)[0]
        probs = model.predict_proba(X_input)[0]
        result = LABELS[pred_idx]
        
        # Result Display
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        res_col1, res_col2 = st.columns([2, 1])
        
        with res_col1:
            st.header(f"{LABEL_EMOJI[result]} Predicted Tier: {result}")
            st.write(f"The model is **{probs[pred_idx]:.1%}** confident in this forecast.")
            
        with res_col2:
            st.metric("Risk Score", "Low" if result == "High" else "High")
        st.markdown('</div>', unsafe_allow_html=True)

        # Probabilities
        st.subheader("Probability Breakdown")
        p_cols = st.columns(3)
        for i, label in enumerate(["Low", "Moderate", "High"]):
            with p_cols[i]:
                st.write(f"**{label}**")
                st.progress(float(probs[i]))
                st.write(f"{probs[i]:.1%}")

with tab2:
    st.write("### Model Specs")
    st.info("Algorithm: Gradient Boosting Classifier")
    st.write("This model was trained on the UCI Garment Workers Productivity Dataset.")
    if reference_df is not None:
        st.write("### Dataset Preview")
        st.dataframe(reference_df.head(5))
