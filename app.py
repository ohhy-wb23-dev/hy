import streamlit as st
import pandas as pd
import joblib
import sklearn
import numpy as np

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Garment Productivity Predictor",
    page_icon="🧵",
    layout="wide"
)

# --- 2. ASSET LOADING (With Version Protection) ---
@st.cache_resource
def load_assets():
    try:
        # These must be in the same folder as app.py on GitHub
        model = joblib.load('gbm_model.pkl')
        model_cols = joblib.load('gbm_model_columns.pkl')
        return model, model_cols
    except AttributeError as e:
        st.error("### 🛑 Version Mismatch Detected")
        st.warning(f"The model was trained with **scikit-learn 1.6.1**, but this app is running **{sklearn.__version__}**.")
        st.info("👉 **FIX:** Update your `requirements.txt` to: `scikit-learn==1.6.1` and reboot the app.")
        st.stop()
    except FileNotFoundError:
        st.error("### 📂 Files Not Found")
        st.info("Ensure `gbm_model.pkl` and `gbm_model_columns.pkl` are in the root directory of your repository.")
        st.stop()
    except Exception as e:
        st.error(f"### ❌ Unexpected Error: {e}")
        st.stop()

model, model_columns = load_assets()

# --- 3. UI HEADER ---
st.title("🧵 Garment Factory Productivity Predictor")
st.markdown("""
Predict the productivity level (**Low, Moderate, High**) of factory teams based on operational metrics. 
This tool uses an **Optimized Gradient Boosting Pipeline**.
""")
st.divider()

# --- 4. INPUT TABS/COLUMNS ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📅 Schedule & Dept")
    day = st.selectbox("Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])
    quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
    # Map to 'sewing' or 'finished' to match your training data labels
    dept_choice = st.selectbox("Department", ["Sewing", "Finishing"])
    dept = "sewing" if dept_choice == "Sewing" else "finished"
    team = st.slider("Team Number", 1, 12, 1)

with col2:
    st.subheader("⚙️ Resource Specs")
    smv = st.number_input("Standard Minute Value (SMV)", value=22.0, min_value=2.0, max_value=60.0)
    wip = st.number_input("Work in Progress (WIP)", value=500.0, min_value=0.0)
    workers = st.number_input("Number of Workers", value=30.0, min_value=1.0)
    style_change = st.selectbox("Number of Style Changes", [0, 1, 2])

with col3:
    st.subheader("💰 Performance Metrics")
    incentive = st.number_input("Incentive Amount", value=0, min_value=0)
    overtime = st.number_input("Overtime (Scaled Value)", value=0.0, step=0.1)
    idle_time = st.number_input("Idle Time (Mins)", value=0.0)
    idle_men = st.number_input("Idle Workers Count", value=0)

# --- 5. PREDICTION LOGIC ---
st.divider()

if st.button("🚀 Generate Productivity Forecast", use_container_width=True):
    # Constructing the input in the raw format expected by the Pipeline
    input_dict = {
        'quarter': [quarter],
        'department': [dept],
        'day': [day],
        'team': [team],
        'smv': [smv],
        'wip': [wip],
        'incentive': [incentive],
        'idle_time': [idle_time],
        'idle_men': [idle_men],
        'no_of_style_change': [int(style_change)],
        'no_of_workers': [workers],
        'over_time_scaled': [overtime]
    }
    
    input_df = pd.DataFrame(input_dict)

    try:
        # CRITICAL: Reorder columns to match 'gbm_model_columns.pkl' exactly
        input_df = input_df[model_columns]

        # Get Prediction and Probabilities
        prediction_idx = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]
        
        # Label Mapping (Matches your notebook mapping)
        labels = ['Low', 'Moderate', 'High']
        result = labels[prediction_idx]
        
        # Result Display
        st.markdown(f"<h2 style='text-align: center;'>Predicted Productivity: <span style='color:#1E90FF;'>{result}</span></h2>", unsafe_allow_html=True)
        
        # Metric Probabilities
        m1, m2, m3 = st.columns(3)
        m1.metric("Low Risk", f"{probs[0]:.1%}")
        m2.metric("Moderate", f"{probs[1]:.1%}")
        m3.metric("High Output", f"{probs[2]:.1%}")

        if result == 'High':
            st.success("✅ Target likely to be exceeded. Excellent resource balance.")
            st.balloons()
        elif result == 'Moderate':
            st.warning("⚠️ Standard performance. Monitor for minor bottlenecks.")
        else:
            st.error("🚨 High risk of shortfall. Review incentive and worker allocation.")

    except Exception as e:
        st.error(f"Prediction Failure: {e}")
        st.info("This is usually caused by column names mismatching the trained model.")
