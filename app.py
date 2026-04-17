import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import PolynomialFeatures

# --- CONFIGURATION ---
st.set_page_config(page_title="Garment Productivity Predictor", layout="wide")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    # Loading the GBM model and columns saved in Stage 1/3 of your notebook
    model = joblib.load('gbm_model.pkl')
    model_columns = joblib.load('gbm_model_columns.pkl')
    return model, model_columns

try:
    model, model_columns = load_assets()
except Exception as e:
    st.error("Model files not found. Please ensure 'gbm_model.pkl' and 'gbm_model_columns.pkl' are in the directory.")
    st.stop()

# --- UI DESIGN ---
st.title("🧵 Garment Factory Productivity Predictor")
st.markdown("### Model: **Optimized Gradient Boosting Classifier**")
st.info("This model predicts productivity tiers based on historical garment factory data with a 71% test accuracy.")

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
    
    wip = st.number_input("Work in Progress (wip)", value=500, help="Max recorded: 23,122")
    if wip > 25000:
        st.error("⚠️ Value significantly higher than training data.")
        form_is_invalid = True
        
    workers = st.number_input("Number of Workers", value=30, min_value=2, max_value=100)
    
    style_change = st.selectbox("Number of Style Changes", ["0", "1", "2"])
    
    smv = st.number_input("SMV (Complexity)", value=22.0, min_value=2.0, max_value=60.0)

with col3:
    st.subheader("💰 Incentives & Metrics")
    
    incentive = st.number_input("Incentive Amount", value=0, min_value=0, max_value=4000)
    
    # Matching the 'over_time_scaled' input from your training set
    overtime = st.number_input("Overtime (Scaled Value)", value=0.0, step=0.1)
    
    idle_time = st.number_input("Idle Time (Mins)", value=0, min_value=0, max_value=300)
    
    idle_men = st.number_input("Idle Workers Count", value=0, min_value=0, max_value=50)

# --- PREDICTION LOGIC ---
st.divider()

if form_is_invalid:
    st.warning("Please correct the errors above to enable the prediction.")
else:
    if st.button("Generate Productivity Forecast", use_container_width=True):
        # 1. Create Raw Input DataFrame
        # We start with the original features expected by the pipeline before polynomial expansion
        raw_data = {
            'team': [team],
            'smv': [smv],
            'wip': [wip],
            'incentive': [incentive],
            'idle_time': [idle_time],
            'idle_men': [idle_men],
            'no_of_workers': [workers],
            'over_time_scaled': [overtime],
            'quarter': [quarter],
            'department': [dept.lower()],
            'day': [day],
            'no_of_style_change': [int(style_change)]
        }
        
        input_df = pd.DataFrame(raw_data)

        # 2. Prediction
        # Since your 'best_model' is a Pipeline (Preprocessor + Classifier),
        # it will handle the OneHotEncoding and Scaling automatically.
        try:
            prediction_idx = model.predict(input_df)[0]
            probs = model.predict_proba(input_df)[0]
            
            # Mapping back to the target names from your notebook
            labels = ['Low', 'Moderate', 'High']
            result = labels[prediction_idx]
            
            # 3. DISPLAY RESULTS
            st.markdown(f"<h2 style='text-align: center;'>Predicted Tier: {result}</h2>", unsafe_allow_html=True)
            
            # Progress bars for probabilities
            cols = st.columns(3)
            for i, label in enumerate(labels):
                cols[i].metric(label, f"{probs[i]:.1%}")
                cols[i].progress(probs[i])

            if result == 'High':
                st.success("High productivity forecast. The team is likely to exceed targets.")
                st.balloons()
            elif result == 'Moderate':
                st.warning("Moderate productivity forecast. Keep monitoring resource efficiency.")
            else:
                st.error("Low productivity forecast. Intervention may be required to meet targets.")
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.info("Ensure the input features match the model's expected training format.")
