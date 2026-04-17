import streamlit as st
import pandas as pd
import joblib
import numpy as np

# These imports are CRITICAL for joblib to load the gbm_model.pkl pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures

# --- CONFIGURATION ---
st.set_page_config(page_title="Garment Productivity Predictor (GBM)", layout="wide")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    # 1. Load the trained Pipeline (Model + Preprocessing)
    # The error in your screenshot was likely due to missing the imports above
    model = joblib.load('gbm_model.pkl')
    
    # 2. Load the specific column list used during training
    model_columns = joblib.load('gbm_model_columns.pkl')
    
    return model, model_columns

# Unpack both assets
model, model_columns = load_assets()

# --- UI DESIGN ---
st.title("🧵 Garment Factory Productivity Predictor")
st.markdown("### Model: Optimized Gradient Boosting Classifier")
st.info("Input the production metrics below to forecast the productivity tier.")

# Track form validity
form_is_invalid = False

# --- INPUT COLUMNS ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📅 Contextual Data")
    quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
    department = st.selectbox("Department", ["sewing", "finished"])
    day = st.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])
    team = st.slider("Team Number", 1, 12, 1)

with col2:
    st.subheader("⚙️ Production Metrics")
    smv = st.number_input("SMV (Standard Minute Value)", value=22.0, min_value=2.0, max_value=60.0)
    
    wip = st.number_input("Work in Progress (WIP)", value=500.0)
    if wip > 23122:
        st.error("⚠️ Max WIP allowed is 23,122")
        form_is_invalid = True
        
    no_of_workers = st.number_input("Number of Workers", value=30.0)
    if no_of_workers > 90 or no_of_workers < 2:
        st.error("⚠️ Range allowed: 2 to 90")
        form_is_invalid = True

    no_of_style_change = st.selectbox("Style Changes", [0, 1, 2])

with col3:
    st.subheader("💰 Incentives & Time")
    incentive = st.number_input("Incentive Amount", value=50)
    if incentive > 3600:
        st.error("⚠️ Max Incentive is 3,600")
        form_is_invalid = True

    over_time_scaled = st.slider("Overtime (Scaled Index)", -2.0, 7.0, 0.0)
    
    idle_time = st.number_input("Idle Time (Mins)", value=0.0)
    if idle_time > 300:
        st.error("⚠️ Max Idle Time is 300")
        form_is_invalid = True

    idle_men = st.number_input("Idle Workers Count", value=0)
    if idle_men > 45:
        st.error("⚠️ Max Idle Workers is 45")
        form_is_invalid = True

st.divider()

# --- PREDICTION LOGIC ---
if form_is_invalid:
    st.warning("Please correct the validation errors to proceed.")
    st.button("Generate Forecast", disabled=True)
else:
    if st.button("Generate Productivity Forecast", use_container_width=True):
        # Create a dictionary matching the raw column names expected by the preprocessor
        data_dict = {
            'quarter': [quarter],
            'department': [department],
            'day': [day],
            'team': [team],
            'smv': [smv],
            'wip': [wip],
            'incentive': [incentive],
            'idle_time': [idle_time],
            'idle_men': [idle_men],
            'no_of_style_change': [no_of_style_change],
            'no_of_workers': [no_of_workers],
            'over_time_scaled': [over_time_scaled]
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame(data_dict)
        
        # Ensure column order matches exactly with the training feature schema
        input_df = input_df[model_columns]

        # Use the Pipeline to predict
        # This handles scaling and encoding automatically
        prediction_idx = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]
        
        labels = ['Low', 'Moderate', 'High']
        result = labels[prediction_idx]
        
        # Display Results
        st.markdown(f"## Predicted Productivity Tier: **{result}**")
        
        # Determine confidence and color based on index (0=Low, 1=Moderate, 2=High)
        if result == 'High':
            st.success(f"Confidence: {probs[2]:.2%} — Optimal production levels expected.")
            st.balloons()
        elif result == 'Moderate':
            st.warning(f"Confidence: {probs[1]:.2%} — Standard operational capacity.")
        else:
            st.error(f"Confidence: {probs[0]:.2%} — High probability of production lag.")
