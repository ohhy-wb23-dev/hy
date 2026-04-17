import streamlit as st
import pandas as pd
import joblib
import sklearn

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Garment Productivity Predictor", layout="wide")

# --- 2. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        # Loading the assets exactly as you dumped them in cell 41
        model = joblib.load('gbm_model.pkl')
        model_columns = joblib.load('gbm_model_columns.pkl')
        return model, model_columns
    except FileNotFoundError:
        st.error("Model files not found. Ensure 'gbm_model.pkl' and 'gbm_model_columns.pkl' are in the same folder as this script.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info(f"Your current scikit-learn version is {sklearn.__version__}. Ensure it matches your training environment.")
        st.stop()

model, model_columns = load_assets()

# --- 3. UI DESIGN ---
st.title("🧵 Garment Factory Productivity Predictor")
st.markdown("### Model: **Optimized Gradient Boosting (Pipeline)**")
st.info("Validation is active. The model predicts productivity into Low, Moderate, or High tiers.")

# --- 4. INPUT COLUMNS ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📅 Time & Place")
    day = st.selectbox("Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])
    quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
    dept = st.selectbox("Department", ["Sewing", "Finishing"])
    team = st.slider("Team Number", 1, 12, 1)

with col2:
    st.subheader("⚙️ Resource Allocation")
    wip = st.number_input("Work in Progress (wip)", value=500.0)
    workers = st.number_input("Number of Workers", value=30.0, min_value=2.0, max_value=90.0)
    style_change = st.selectbox("Number of Style Changes", [0, 1, 2])
    smv = st.number_input("SMV (Complexity)", value=22.0, min_value=2.9, max_value=55.0)

with col3:
    st.subheader("💰 Incentives & Metrics")
    incentive = st.number_input("Incentive Amount", value=0)
    overtime = st.number_input("Overtime (Scaled Value)", value=0.0)
    idle_time = st.number_input("Idle Time (Mins)", value=0.0)
    idle_men = st.number_input("Idle Workers Count", value=0)

# --- 5. PREDICTION LOGIC ---
st.divider()

if st.button("Generate Productivity Forecast", use_container_width=True):
    # Prepare the input dictionary with original column names before encoding
    # The Pipeline handles the encoding and scaling internally
    input_dict = {
        'quarter': [quarter],
        'department': [dept.lower()],
        'day': [day],
        'team': [team],
        'smv': [smv],
        'wip': [wip],
        'incentive': [incentive],
        'idle_time': [idle_time],
        'idle_men': [idle_men],
        'no_of_workers': [workers],
        'no_of_style_change': [style_change],
        'over_time_scaled': [overtime]
    }
    
    input_df = pd.DataFrame(input_dict)

    try:
        # Align inputs with the specific training columns
        input_df = input_df[model_columns]

        # Prediction and Probabilities
        prediction_idx = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]
        
        # Mapping from your notebook: {Low: 0, Moderate: 1, High: 2}
        labels = ['Low', 'Moderate', 'High']
        result = labels[prediction_idx]
        
        # Display Results
        st.markdown(f"<h2 style='text-align: center;'>Predicted Tier: {result}</h2>", unsafe_allow_html=True)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Low", f"{probs[0]:.1%}")
        m2.metric("Moderate", f"{probs[1]:.1%}")
        m3.metric("High", f"{probs[2]:.1%}")

        if result == 'High':
            st.success("High productivity forecast. Expected to meet or exceed targets.")
            st.balloons()
        elif result == 'Moderate':
            st.warning("Moderate productivity forecast. Monitor for potential bottlenecks.")
        else:
            st.error("Low productivity forecast. Intervention may be required.")
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")
