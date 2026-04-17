import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Garment Productivity Predictor",
    page_icon="🧵",
    layout="wide"
)

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    model = joblib.load("gbm_model.pkl")
    model_columns = joblib.load("gbm_model_columns.pkl")
    return model, model_columns

model, model_columns = load_assets()

# --- UI HEADER ---
st.title("🧵 Garment Factory Productivity Predictor")
st.info("**Model Info:** Tuned Gradient Boosting Model. Validation is active for each field.")

# Track whether any input is invalid
form_is_invalid = False

# --- INPUT LAYOUT ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📅 Time & Place")
    day = st.selectbox(
        "Day of the Week",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"]
    )
    quarter = st.selectbox(
        "Quarter",
        ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"]
    )
    dept = st.selectbox(
        "Department",
        ["Sewing", "Finishing"]
    )
    team = st.slider("Team Number", min_value=1, max_value=12, value=1)

with col2:
    st.subheader("⚙️ Resource Allocation")

    wip = st.number_input("Work in Progress (wip)", min_value=0, value=500)
    if wip > 23122:
        st.error("⚠️ Limit Exceeded: Max wip is 23,122")
        form_is_invalid = True

    workers = st.number_input("Number of Workers", min_value=0, value=30)
    if workers > 90 or workers < 2:
        st.error("⚠️ Limit Exceeded: Range is 2 to 90")
        form_is_invalid = True

    style_change = st.selectbox(
        "Number of Style Changes",
        ["0", "1", "2"]
    )

    smv = st.number_input("SMV (Complexity)", min_value=0.0, value=22.0)
    if smv > 54.6 or smv < 2.9:
        st.error("⚠️ Limit Exceeded: Range is 2.9 to 54.6")
        form_is_invalid = True

with col3:
    st.subheader("💰 Incentives & Metrics")

    incentive = st.number_input("Incentive Amount", min_value=0, value=100)
    if incentive > 3600:
        st.error("⚠️ Limit Exceeded: Max Incentive is 3,600")
        form_is_invalid = True

    overtime = st.slider("Overtime (Scaled)", min_value=-2.0, max_value=2.0, value=0.0)

    idle_time = st.number_input("Idle Time (Mins)", min_value=0, value=0)
    if idle_time > 300:
        st.error("⚠️ Limit Exceeded: Max idle Time is 300")
        form_is_invalid = True

    idle_men = st.number_input("Idle Workers Count", min_value=0, value=0)
    if idle_men > 45:
        st.error("⚠️ Limit Exceeded: Max idle Workers is 45")
        form_is_invalid = True

# --- PREDICTION SECTION ---
st.divider()

if form_is_invalid:
    st.warning("Please correct the errors above to enable the prediction.")
    st.button("Generate Productivity Forecast", disabled=True, use_container_width=True)
else:
    if st.button("Generate Productivity Forecast", use_container_width=True):
        # Create empty input dataframe with all required model columns
        input_df = pd.DataFrame(0, index=[0], columns=model_columns)

        # Fill numeric features
        if "team" in input_df.columns:
            input_df["team"] = team
        if "smv" in input_df.columns:
            input_df["smv"] = smv
        if "wip" in input_df.columns:
            input_df["wip"] = wip
        if "incentive" in input_df.columns:
            input_df["incentive"] = incentive
        if "idle_time" in input_df.columns:
            input_df["idle_time"] = idle_time
        if "idle_men" in input_df.columns:
            input_df["idle_men"] = idle_men
        if "no_of_workers" in input_df.columns:
            input_df["no_of_workers"] = workers
        if "over_time_scaled" in input_df.columns:
            input_df["over_time_scaled"] = overtime

        # Helper for dummy encoding
        def set_dummy(category, value):
            col_name = f"{category}_{value}"
            if col_name in input_df.columns:
                input_df[col_name] = 1

        # Set categorical values
        set_dummy("quarter", quarter)
        set_dummy("department", dept.lower())
        set_dummy("day", day)
        set_dummy("no_of_style_change", style_change)

        # Final alignment
        input_df = input_df[model_columns]

        # Prediction
        prediction_idx = model.predict(input_df)[0]

        # Predict probabilities if supported
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)[0]
        else:
            probs = None

        labels = ["Low", "Moderate", "High"]
        result = labels[int(prediction_idx)]

        st.markdown(f"## Predicted Tier: **{result}**")

        if result == "High":
            if probs is not None:
                st.success(f"Confidence: {probs[2]:.2%} - Optimized production detected.")
            else:
                st.success("Optimized production detected.")
            st.balloons()

        elif result == "Moderate":
            if probs is not None:
                st.warning(f"Confidence: {probs[1]:.2%} - Operating within standard range.")
            else:
                st.warning("Operating within standard range.")

        else:
            if probs is not None:
                st.error(f"Confidence: {probs[0]:.2%} - High risk of target shortfall.")
            else:
                st.error("High risk of target shortfall.")
