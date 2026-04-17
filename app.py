import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =========================
# LOAD DATA & MODEL
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("final_classification_dataset.csv")
    return df

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

df = load_data()
model = load_model()

# =========================
# GET DYNAMIC RANGES
# =========================
def get_ranges(df):
    ranges = {}
    for col in df.columns:
        if df[col].dtype != "object":
            ranges[col] = (df[col].min(), df[col].max())
    return ranges

ranges = get_ranges(df)

# =========================
# HELPER FUNCTION
# =========================
def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

# =========================
# TITLE
# =========================
st.title("Garment Productivity Prediction System")

st.markdown("""
This system predicts worker productivity based on operational inputs.
All input ranges are dynamically derived from the final dataset.
""")

# =========================
# PRESET EXAMPLE
# =========================
st.subheader("Preset Example")

preset = {
    "wip": int(df["wip"].mean()),
    "no_of_workers": int(df["no_of_workers"].mean()),
    "smv": float(df["smv"].mean()),
    "incentive": int(df["incentive"].mean()),
    "idle_time": int(df["idle_time"].mean()),
    "idle_men": int(df["idle_men"].mean()),
    "over_time_scaled": float(df["over_time_scaled"].mean()) if "over_time_scaled" in df.columns else 0.0
}

# Clamp preset values
for key in preset:
    if key in ranges:
        preset[key] = clamp(preset[key], ranges[key][0], ranges[key][1])

# =========================
# INPUT SECTION
# =========================
st.subheader("Input Parameters")

col1, col2 = st.columns(2)

with col1:
    wip = st.number_input(
        "Work In Progress (WIP)",
        int(ranges["wip"][0]), int(ranges["wip"][1]),
        preset["wip"]
    )
    st.caption(f"Range: {ranges['wip'][0]} – {ranges['wip'][1]}")

    workers = st.number_input(
        "Number of Workers",
        int(ranges["no_of_workers"][0]), int(ranges["no_of_workers"][1]),
        preset["no_of_workers"]
    )
    st.caption(f"Range: {ranges['no_of_workers'][0]} – {ranges['no_of_workers'][1]}")

    smv = st.number_input(
        "SMV (Standard Minute Value)",
        float(ranges["smv"][0]), float(ranges["smv"][1]),
        preset["smv"]
    )
    st.caption(f"Range: {ranges['smv'][0]} – {ranges['smv'][1]}")

with col2:
    incentive = st.number_input(
        "Incentive",
        int(ranges["incentive"][0]), int(ranges["incentive"][1]),
        preset["incentive"]
    )
    st.caption(f"Range: {ranges['incentive'][0]} – {ranges['incentive'][1]}")

    idle_time = st.number_input(
        "Idle Time",
        int(ranges["idle_time"][0]), int(ranges["idle_time"][1]),
        preset["idle_time"]
    )
    st.caption(f"Range: {ranges['idle_time'][0]} – {ranges['idle_time'][1]}")

    idle_men = st.number_input(
        "Idle Workers",
        int(ranges["idle_men"][0]), int(ranges["idle_men"][1]),
        preset["idle_men"]
    )
    st.caption(f"Range: {ranges['idle_men'][0]} – {ranges['idle_men'][1]}")

# =========================
# OVERTIME (SPECIAL CASE)
# =========================
st.subheader("Additional Factor")

if "over_time_scaled" in ranges:
    overtime = st.slider(
        "Overtime (Scaled)",
        float(ranges["over_time_scaled"][0]),
        float(ranges["over_time_scaled"][1]),
        preset["over_time_scaled"]
    )
else:
    overtime = st.slider("Overtime (Scaled)", -2.0, 2.0, 0.0)

# =========================
# PREDICTION
# =========================
st.subheader("Prediction")

if st.button("Predict Productivity"):

    input_data = pd.DataFrame({
        "wip": [wip],
        "no_of_workers": [workers],
        "smv": [smv],
        "incentive": [incentive],
        "idle_time": [idle_time],
        "idle_men": [idle_men],
        "over_time_scaled": [overtime]
    })

    prediction = model.predict(input_data)[0]

    # =========================
    # CLASSIFICATION LOGIC
    # =========================
    if prediction < 0.5:
        category = "Low Productivity"
    elif prediction < 0.75:
        category = "Moderate Productivity"
    else:
        category = "High Productivity"

    # =========================
    # OUTPUT
    # =========================
    st.success(f"Predicted Productivity: {prediction:.2f}")
    st.info(f"Category: {category}")

# =========================
# DATA PREVIEW
# =========================
with st.expander("View Dataset Summary"):
    st.write(df.describe())

# =========================
# FOOTER
# =========================
st.markdown("""
---
System designed with dataset-driven input validation to ensure consistency 
between training data and prediction environment.
""")
