import streamlit as st
import pandas as pd
import pickle

# =========================
# LOAD DATA & MODEL
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("final_classification_dataset.csv")

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

df = load_data()
model = load_model()

# =========================
# GET RANGES FROM DATASET
# =========================
def get_ranges(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    return {col: (df[col].min(), df[col].max()) for col in numeric_cols}

ranges = get_ranges(df)

# =========================
# TITLE
# =========================
st.title("Garment Worker Productivity Prediction")

st.markdown("""
This system predicts productivity based on real production factors.
All input ranges are dynamically based on the dataset to ensure valid predictions.
""")

# =========================
# DEFAULT VALUES (MEAN)
# =========================
def get_default(col):
    return float(df[col].mean())

# =========================
# INPUT SECTION
# =========================
st.subheader("Input Parameters")

col1, col2 = st.columns(2)

with col1:
    wip = st.number_input(
        "Work In Progress (WIP)",
        min_value=int(ranges["wip"][0]),
        max_value=int(ranges["wip"][1]),
        value=int(get_default("wip"))
    )
    st.caption(f"Range: {ranges['wip'][0]} – {ranges['wip'][1]}")

    smv = st.number_input(
        "SMV (Standard Minute Value)",
        min_value=float(ranges["smv"][0]),
        max_value=float(ranges["smv"][1]),
        value=get_default("smv")
    )
    st.caption(f"Range: {ranges['smv'][0]} – {ranges['smv'][1]}")

    workers = st.number_input(
        "Number of Workers",
        min_value=int(ranges["no_of_workers"][0]),
        max_value=int(ranges["no_of_workers"][1]),
        value=int(get_default("no_of_workers"))
    )
    st.caption(f"Range: {ranges['no_of_workers'][0]} – {ranges['no_of_workers'][1]}")

with col2:
    incentive = st.number_input(
        "Incentive",
        min_value=int(ranges["incentive"][0]),
        max_value=int(ranges["incentive"][1]),
        value=int(get_default("incentive"))
    )
    st.caption(f"Range: {ranges['incentive'][0]} – {ranges['incentive'][1]}")

    idle_time = st.number_input(
        "Idle Time",
        min_value=int(ranges["idle_time"][0]),
        max_value=int(ranges["idle_time"][1]),
        value=int(get_default("idle_time"))
    )
    st.caption(f"Range: {ranges['idle_time'][0]} – {ranges['idle_time'][1]}")

    idle_men = st.number_input(
        "Idle Workers",
        min_value=int(ranges["idle_men"][0]),
        max_value=int(ranges["idle_men"][1]),
        value=int(get_default("idle_men"))
    )
    st.caption(f"Range: {ranges['idle_men'][0]} – {ranges['idle_men'][1]}")

# =========================
# OVERTIME (RAW VALUES)
# =========================
st.subheader("Overtime")

overtime = st.number_input(
    "Overtime (Minutes)",
    min_value=int(ranges["over_time"][0]),
    max_value=int(ranges["over_time"][1]),
    value=int(get_default("over_time"))
)
st.caption(f"Range: {ranges['over_time'][0]} – {ranges['over_time'][1]}")

# =========================
# PREDICTION
# =========================
st.subheader("Prediction")

if st.button("Predict"):

    input_data = pd.DataFrame({
        "wip": [wip],
        "no_of_workers": [workers],
        "smv": [smv],
        "incentive": [incentive],
        "idle_time": [idle_time],
        "idle_men": [idle_men],
        "over_time": [overtime]
    })

    prediction = model.predict(input_data)[0]

    # =========================
    # CLASSIFICATION
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
# DATA OVERVIEW
# =========================
with st.expander("Dataset Summary"):
    st.write(df.describe())

# =========================
# FOOTER
# =========================
st.markdown("""
---
This application ensures consistency between dataset, model training, 
and user input ranges to improve prediction reliability.
""")
