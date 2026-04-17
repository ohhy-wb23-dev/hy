import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Garment Factory Productivity Predictor",
    page_icon="🧵",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 1.2rem;
    }
    .section-card {
        background-color: #f8fafc;
        padding: 1rem 1rem 0.8rem 1rem;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    .small-note {
        color: #6b7280;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD ASSETS & SCALING CONSTANTS
# =========================================================
# Statistics from final_classification_dataset.csv
OVERTIME_MEAN = 4567.46
OVERTIME_STD = 3348.82

@st.cache_resource
def load_model_assets():
    # Updated to your Gradient Boosting files
    model = joblib.load("gbm_model.pkl")
    model_columns = joblib.load("gbm_model_columns.pkl")
    return model, model_columns

@st.cache_data
def load_dataset():
    return pd.read_csv("final_classification_dataset.csv")

try:
    model, model_columns = load_model_assets()
    df = load_dataset()
except Exception:
    st.error("Critical Error: Missing gbm_model.pkl, gbm_model_columns.pkl, or dataset file.")
    st.stop()

# =========================================================
# DATA-DRIVEN OPTIONS
# =========================================================
quarter_options = sorted(df["quarter"].dropna().unique().tolist())
department_options = sorted(df["department"].dropna().unique().tolist())
day_options = sorted(df["day"].dropna().unique().tolist())
style_change_options = sorted(df["no_of_style_change"].dropna().unique().tolist())

# Numeric ranges
smv_min, smv_max = float(df["smv"].min()), float(df["smv"].max())
wip_min, wip_max = int(df["wip"].min()), int(df["wip"].max())
# Adjusted max to 15,000 as per your previous request
over_time_min, over_time_max = 0, 15000 
incentive_min, incentive_max = int(df["incentive"].min()), int(df["incentive"].max())
idle_time_min, idle_time_max = int(df["idle_time"].min()), int(df["idle_time"].max())
idle_men_min, idle_men_max = int(df["idle_men"].min()), int(df["idle_men"].max())
workers_min, workers_max = int(df["no_of_workers"].min()), int(df["no_of_workers"].max())

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def set_dummy_value(input_df, prefix, value):
    # Matches your 192-column naming convention
    col = f"{prefix}_{str(value).lower()}" if prefix == "department" else f"{prefix}_{value}"
    if col in input_df.columns:
        input_df[col] = 1

def normalize_prediction(pred):
    class_map = {0: "Low", 1: "Moderate", 2: "High"}
    return class_map.get(int(pred), str(pred))

def get_result_message(result):
    if result == "High":
        return "success", "The current input pattern suggests strong production performance."
    elif result == "Moderate":
        return "warning", "The current input pattern suggests average but stable production performance."
    else:
        return "error", "The current input pattern suggests a risk of lower productivity."

def get_recommendations(result, wip, over_time, incentive, idle_time, idle_men, workers, style_change):
    recs = []
    if result == "Low":
        recs.append("Reduce idle time and idle workers to improve operational efficiency.")
        recs.append("Review whether the current workload is balanced with the number of workers.")
        recs.append("Evaluate incentive strategy to improve worker motivation.")
    elif result == "Moderate":
        recs.append("The production line is operating within a normal range but has room for improvement.")
        recs.append("Better workload planning may help raise productivity to the high tier.")
    else:
        recs.append("The current production setup appears efficient and well balanced.")
    return recs

# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="main-title">🧵 AI-Powered Garment Factory Productivity Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">A machine learning-based decision support prototype using Gradient Boosting Machine (GBM).</div>',
    unsafe_allow_html=True
)
st.success("✅ System Status: GBM Model loaded successfully and ready for prediction.")

# =========================================================
# INPUT AREA
# =========================================================
st.markdown("## 📥 Production Input Form")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📅 Time & Context")
    quarter = st.selectbox("Quarter", quarter_options)
    department = st.selectbox("Department", department_options)
    day = st.selectbox("Day of the Week", day_options)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("⚙️ Production Factors")
    smv = st.number_input("SMV (Complexity)", min_value=smv_min, max_value=smv_max, value=22.0, format="%.2f")
    wip = st.number_input("Work in Progress (WIP)", min_value=wip_min, max_value=wip_max, value=500)
    no_of_style_change = st.selectbox("Number of Style Changes", style_change_options)
    no_of_workers = st.number_input("Number of Workers", min_value=workers_min, max_value=workers_max, value=30)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("💰 Time & Efficiency Metrics")
    over_time = st.number_input("Overtime (Minutes)", min_value=0, max_value=20000, value=1000)
    incentive = st.number_input("Incentive Amount", min_value=incentive_min, max_value=incentive_max, value=50)
    idle_time = st.number_input("Idle Time (Mins)", min_value=idle_time_min, max_value=idle_time_max, value=0)
    idle_men = st.number_input("Idle Workers Count", min_value=idle_men_min, max_value=idle_men_max, value=0)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# PREDICTION
# =========================================================
st.divider()

if st.button("Generate Productivity Forecast", use_container_width=True):
    # Initialize 192 features with zeros
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)

    # Standardize Overtime internally
    ot_scaled = (over_time - OVERTIME_MEAN) / OVERTIME_STD

    # Numeric fields
    numeric_fields = {
        "smv": smv,
        "wip": wip,
        "over_time_scaled": ot_scaled,
        "incentive": incentive,
        "idle_time": idle_time,
        "idle_men": idle_men,
        "no_of_workers": no_of_workers,
        "team": 1 # Default team if not in UI
    }

    for col, val in numeric_fields.items():
        if col in input_df.columns:
            input_df[col] = val

    # One-Hot Encoding for categories
    set_dummy_value(input_df, "quarter", quarter)
    set_dummy_value(input_df, "department", department)
    set_dummy_value(input_df, "day", day)
    set_dummy_value(input_df, "no_of_style_change", no_of_style_change)

    # Predict
    raw_pred = model.predict(input_df[model_columns])[0]
    result = normalize_prediction(raw_pred)
    probs = model.predict_proba(input_df[model_columns])[0]

    status_type, status_msg = get_result_message(result)
    st.markdown("## 📊 Prediction Results")

    confidence = float(np.max(probs))
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Productivity", result)
    c2.metric("Confidence Score", f"{confidence:.2%}")
    c3.metric("Workers", int(no_of_workers))

    if status_type == "success": st.success(status_msg)
    elif status_type == "warning": st.warning(status_msg)
    else: st.error(status_msg)

    t1, t2, t3, t4 = st.tabs(["📌 Summary", "📈 Confidence", "💡 Recommendations", "🤖 GBM Explanation"])
    with t1: st.dataframe(input_df[model_columns]) # Technical view
    with t2:
        prob_df = pd.DataFrame({"Level": ["Low", "Moderate", "High"], "Probability": probs})
        st.bar_chart(prob_df.set_index("Level"))
    with t3:
        for i, rec in enumerate(get_recommendations(result, wip, over_time, incentive, idle_time, idle_men, no_of_workers, no_of_style_change), 1):
            st.write(f"**{i}.** {rec}")
    with t4: st.write("The system uses **Gradient Boosting (GBM)** to optimize prediction accuracy through sequential error correction.")
