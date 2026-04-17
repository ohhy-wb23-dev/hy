import streamlit as st
import pandas as pd
import joblib
import sklearn

# --- CONFIGURATION & ASSETS ---
st.set_page_config(page_title="Productivity Forecast Pro", layout="wide")

@st.cache_resource
def load_assets():
    try:
        # Loading both files: the pipeline and the column reference
        model = joblib.load('gbm_model.pkl')
        model_columns = joblib.load('gbm_model_columns.pkl')
        return model, model_columns
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'gbm_model.pkl' and 'gbm_model_columns.pkl' are in the directory.")
        st.stop()
    except AttributeError as e:
        st.error(f"Version Mismatch: The model was saved with a different version of scikit-learn. Current version: {sklearn.__version__}")
        st.stop()

model, model_columns = load_assets()

# --- INPUT UI ---
st.title("🧵 Garment Factory Productivity Predictor")
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
    workers = st.number_input("Number of Workers", value=30, min_value=2, max_value=90)
    style_change = st.selectbox("Number of Style Changes", [0, 1, 2])
    smv = st.number_input("SMV (Complexity)", value=22.0)

with col3:
    st.subheader("💰 Incentives & Metrics")
    incentive = st.number_input("Incentive Amount", value=0)
    overtime = st.number_input("Overtime (Scaled Value)", value=0.0)
    idle_time = st.number_input("Idle Time (Mins)", value=0)
    idle_men = st.number_input("Idle Workers Count", value=0)

# --- PREDICTION ---
if st.button("Generate Productivity Forecast", use_container_width=True):
    # Prepare raw input matching the training columns
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

    # Reorder columns to match the pkl reference used in training
    input_df = input_df[model_columns]

    # Generate Prediction
    prediction_idx = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]
    
    labels = ['Low', 'Moderate', 'High']
    result = labels[prediction_idx]
    
    st.markdown(f"<h2 style='text-align: center;'>Predicted Tier: {result}</h2>", unsafe_allow_html=True)
    
    # Progress bars for probabilities
    m1, m2, m3 = st.columns(3)
    m1.metric("Low", f"{probs[0]:.1%}")
    m2.metric("Moderate", f"{probs[1]:.1%}")
    m3.metric("High", f"{probs[2]:.1%}")
