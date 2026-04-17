import streamlit as st
import pandas as pd
import joblib
import sklearn

# --- 1. SETTINGS & ASSETS ---
st.set_page_config(page_title="Productivity Forecast Pro", layout="wide")

@st.cache_resource
def load_ml_assets():
    try:
        # Loading the specific files you dumped in cell 41 of your notebook
        model = joblib.load('gbm_model.pkl')
        model_cols = joblib.load('gbm_model_columns.pkl')
        return model, model_cols
    except FileNotFoundError:
        st.error("❌ Model files missing. Ensure 'gbm_model.pkl' and 'gbm_model_columns.pkl' are in this folder.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Version Error: {e}")
        st.info(f"Colab Version: 1.6.1 | Your Version: {sklearn.__version__}")
        st.stop()

model, model_columns = load_ml_assets()

# --- 2. USER INTERFACE ---
st.title("🧵 Garment Productivity Predictor")
st.markdown("### Model: **Optimized Gradient Boosting Pipeline**")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📅 Context")
    day = st.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])
    quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
    dept = st.selectbox("Department", ["Sewing", "Finishing"])
    team = st.slider("Team Number", 1, 12, 1)

with col2:
    st.subheader("⚙️ Resources")
    wip = st.number_input("Work in Progress (wip)", value=500.0)
    workers = st.number_input("No. of Workers", value=30.0)
    style_change = st.selectbox("Style Changes", [0, 1, 2])
    smv = st.number_input("SMV (Complexity)", value=22.0)

with col3:
    st.subheader("💰 Metrics")
    incentive = st.number_input("Incentive", value=0)
    # over_time_scaled was added as a specific feature in your refined model
    overtime = st.number_input("Overtime (Scaled Value)", value=0.0)
    idle_time = st.number_input("Idle Time (Mins)", value=0.0)
    idle_men = st.number_input("Idle Workers Count", value=0)

# --- 3. PREDICTION ENGINE ---
st.divider()

if st.button("Generate Productivity Forecast", use_container_width=True):
    # Construct raw dataframe using the original feature names
    input_dict = {
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
    
    input_df = pd.DataFrame(input_dict)

    try:
        # CRITICAL: Reorder columns to match 'gbm_model_columns.pkl' exactly
        input_df = input_df[model_columns]

        # Predict using the Pipeline (it handles scaling/encoding internally)
        prediction_idx = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]
        
        # UI Mapping consistent with notebook
        labels = ['Low', 'Moderate', 'High']
        result = labels[prediction_idx]
        
        st.markdown(f"<h2 style='text-align: center;'>Predicted Tier: {result}</h2>", unsafe_allow_html=True)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Low Prob.", f"{probs[0]:.1%}")
        m2.metric("Moderate Prob.", f"{probs[1]:.1%}")
        m3.metric("High Prob.", f"{probs[2]:.1%}")

        if result == 'High':
            st.success("Target likely to be exceeded.")
            st.balloons()
        elif result == 'Moderate':
            st.warning("On track for standard targets.")
        else:
            st.error("High risk of falling below target.")

    except Exception as e:
        st.error(f"Prediction logic failed: {e}")
