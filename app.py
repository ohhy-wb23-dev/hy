import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. CONFIGURATION & ASSETS ---
st.set_page_config(page_title="Productivity Forecast Pro", layout="wide")

@st.cache_resource
def load_assets():
    # Loading both files: the pipeline and the column reference
    model = joblib.load('gbm_model.pkl')
    model_columns = joblib.load('gbm_model_columns.pkl')
    return model, model_columns

try:
    model, model_columns = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# --- 2. UI DESIGN ---
st.title("🧵 Garment Factory Productivity Predictor")
st.markdown("### Model: **Optimized Gradient Boosting Classifier**")
st.info("Input the production parameters below to predict the productivity tier (Low, Moderate, or High).")

form_is_invalid = False

# --- 3. INPUT COLUMNS ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📅 Time & Place")
    day = st.selectbox("Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])
    quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
    dept = st.selectbox("Department", ["Sewing", "Finishing"])
    team = st.slider("Team Number", 1, 12, 1)

with col2:
    st.subheader("⚙️ Resource Allocation")
    
    # Validation based on training data limits
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
    
    # over_time_scaled is a key feature in your GBM model
    overtime = st.number_input("Overtime (Scaled Value)", value=0.0, step=0.1)
    
    idle_time = st.number_input("Idle Time (Mins)", value=0, max_value=300)
    idle_men = st.number_input("Idle Workers Count", value=0, max_value=45)

# --- 4. PREDICTION LOGIC ---
st.divider()

if form_is_invalid:
    st.warning("Please correct the errors above to enable the prediction.")
else:
    if st.button("Generate Productivity Forecast", use_container_width=True):
        # Create initial DataFrame with raw inputs
        input_dict = {
            'quarter': [quarter],
            'department': [dept.lower()],
            'day': [day],
            'team': [team],
            'smv': [smv],
            'wip': [wip],
            'over_time_scaled': [overtime],
            'incentive': [incentive],
            'idle_time': [idle_time],
            'idle_men': [idle_men],
            'no_of_style_change': [style_change],
            'no_of_workers': [workers]
        }
        
        input_df = pd.DataFrame(input_dict)

        # IMPORTANT: Align with the exact columns used during training
        # This handles the order and presence of features before the Pipeline takes over
        try:
            # We ensure all columns from model_columns exist, filling missing with 0
            # This is crucial if your model expects specific dummy variables
            for col in model_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            # Reorder columns to match the pkl reference
            input_df = input_df[model_columns]

            # Generate Prediction
            prediction_idx = model.predict(input_df)[0]
            probs = model.predict_proba(input_df)[0]
            
            # Target labels mapping
            labels = ['Low', 'Moderate', 'High']
            result = labels[prediction_idx]
            
            # --- 5. RESULTS DISPLAY ---
            st.markdown(f"<h2 style='text-align: center;'>Predicted Tier: {result}</h2>", unsafe_allow_html=True)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Low Risk", f"{probs[0]:.1%}")
            m2.metric("Moderate", f"{probs[1]:.1%}")
            m3.metric("High Output", f"{probs[2]:.1%}")

            if result == 'High':
                st.success("Target likely to be exceeded. Excellent resource configuration.")
                st.balloons()
            elif result == 'Moderate':
                st.warning("On track for standard targets. Monitor for bottlenecks.")
            else:
                st.error("High risk of target shortfall. Review resource allocation.")
                
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info("Check if your gbm_model_columns.pkl matches the input features.")
