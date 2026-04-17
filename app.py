import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- CONFIGURATION ---
st.set_page_config(page_title="Productivity Forecast Pro", layout="wide")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    # Ensure these names match what you saved in your notebook
    model = joblib.load('gbm_model.pkl') 
    return model

model = load_assets()

# --- UI DESIGN ---
st.title("🧵 Garment Factory Productivity Predictor")
st.markdown("### Model: **Optimized Gradient Boosting Classifier**")
st.info("Validation active. This model uses engineered features to predict production efficiency.")

form_is_invalid = False

# --- INPUT COLUMNS ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📅 Time & Place")
    day = st.selectbox("Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])
    quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
    dept = st.selectbox("Department", ["Sewing", "Finishing"])
    team = st.slider("Team Number", 1, 12, 1)

with col2:
    st.subheader("⚙️ Resource Allocation")
    
    wip = st.number_input("Work in Progress (wip)", value=500)
    if wip > 23122:
        st.error("⚠️ Value exceeds training range (Max: 23,122)")
        form_is_invalid = True
        
    workers = st.number_input("Number of Workers", value=30, min_value=2, max_value=90)
    
    style_change = st.selectbox("Number of Style Changes", [0, 1, 2])
    
    smv = st.number_input("SMV (Complexity)", value=22.0, min_value=2.9, max_value=54.6)

with col3:
    st.subheader("💰 Incentives & Metrics")
    
    incentive = st.number_input("Incentive Amount", value=0, max_value=3600)
    
    # Matching the 'over_time_scaled' feature from your GBM training
    overtime = st.number_input("Overtime (Scaled Value)", value=0.0, step=0.1)
    
    idle_time = st.number_input("Idle Time (Mins)", value=0, max_value=300)
    idle_men = st.number_input("Idle Workers Count", value=0, max_value=45)

# --- PREDICTION LOGIC ---
st.divider()

if form_is_invalid:
    st.warning("Please correct the errors above to enable the prediction.")
    st.button("Generate Productivity Forecast", disabled=True)
else:
    if st.button("Generate Productivity Forecast", use_container_width=True):
        # 1. Prepare input as a raw DataFrame
        # The Pipeline in your notebook handles Encoding and Scaling automatically
        input_dict = {
            'quarter': [quarter],
            'department': [dept.lower()],
            'day': [day],
            'team': [team],
            'smv': [smv],
            'wip': [wip],
            'over_time_scaled': [overtime], # Feature used in your final GBM
            'incentive': [incentive],
            'idle_time': [idle_time],
            'idle_men': [idle_men],
            'no_of_style_change': [style_change],
            'no_of_workers': [workers]
        }
        
        input_df = pd.DataFrame(input_dict)

        # 2. Predict using the Pipeline
        prediction_idx = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]
        
        # Labels from your notebook analysis
        labels = ['Low', 'Moderate', 'High']
        result = labels[prediction_idx]
        
        # 3. Enhanced Results Display
        st.markdown(f"<h2 style='text-align: center;'>Predicted Tier: {result}</h2>", unsafe_allow_html=True)
        
        # Display Probability bars
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Low Risk", f"{probs[0]:.1%}")
        m_col2.metric("Moderate Risk", f"{probs[1]:.1%}")
        m_col3.metric("High Performance", f"{probs[2]:.1%}")

        if result == 'High':
            st.success(f"Confidence: {probs[2]:.2%} - Optimized production detected.")
            st.balloons()
        elif result == 'Moderate':
            st.warning(f"Confidence: {probs[1]:.2%} - Operating within standard range.")
        else:
            st.error(f"Confidence: {probs[0]:.2%} - High risk of target shortfall.")
