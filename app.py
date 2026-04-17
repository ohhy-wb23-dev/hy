
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Garment Productivity Predictor",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM STYLING
# ==========================================
st.markdown("""
<style>
.main-title {
    font-size: 2.1rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}
.sub-text {
    color: #6b7280;
    margin-bottom: 1.2rem;
}
.block-card {
    background: #f8fafc;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 16px 18px;
    margin-bottom: 12px;
}
.result-card {
    border-radius: 18px;
    padding: 22px;
    border: 1px solid #e5e7eb;
    background: linear-gradient(135deg, #ffffff, #f8fafc);
}
.small-note {
    color: #6b7280;
    font-size: 0.9rem;
}
.metric-label {
    font-size: 0.9rem;
    color: #6b7280;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD ASSETS
# ==========================================
@st.cache_resource
def load_assets():
    model = joblib.load("gbm_model.pkl")
    model_columns = joblib.load("gbm_model_columns.pkl")
    return model, model_columns


@st.cache_data
def load_reference_data():
    try:
        df = pd.read_csv("cleaned_garments_worker_productivity.csv")
        return df
    except Exception:
        return None


try:
    model, model_columns = load_assets()
except Exception as e:
    st.error("Unable to load model files. Please ensure 'gbm_model.pkl' and 'gbm_model_columns.pkl' are in the same folder as app.py.")
    st.exception(e)
    st.stop()

reference_df = load_reference_data()

LABELS = {0: "Low", 1: "Moderate", 2: "High"}
LABEL_EMOJI = {"Low": "🔴", "Moderate": "🟡", "High": "🟢"}
LABEL_TEXT = {
    "Low": "Low productivity is likely. Current inputs suggest a risk of underperformance.",
    "Moderate": "Moderate productivity is likely. Operations appear stable, but improvement is still possible.",
    "High": "High productivity is likely. Current inputs indicate a strong operating condition."
}

# ==========================================
# HELPERS
# ==========================================
def get_reference_ranges(df):
    if df is None:
        return {
            "wip": (0, 23122),
            "incentive": (0, 3600),
            "idle_time": (0, 300),
            "idle_men": (0, 45),
            "no_of_workers": (2, 89),
            "smv": (2.9, 54.6),
            "team": (1, 12),
            "over_time_scaled": (-1.36, 6.38),
        }

    ranges = {}
    for col in ["wip", "incentive", "idle_time", "idle_men", "no_of_workers", "smv", "team", "over_time_scaled"]:
        ranges[col] = (float(df[col].min()), float(df[col].max()))
    return ranges


def get_reference_medians(df):
    if df is None:
        return {
            "wip": 586,
            "incentive": 0,
            "idle_time": 0,
            "idle_men": 0,
            "no_of_workers": 34,
            "smv": 15.26,
            "team": 6,
            "over_time_scaled": -0.18,
        }
    return {
        "wip": float(df["wip"].median()),
        "incentive": float(df["incentive"].median()),
        "idle_time": float(df["idle_time"].median()),
        "idle_men": float(df["idle_men"].median()),
        "no_of_workers": float(df["no_of_workers"].median()),
        "smv": float(df["smv"].median()),
        "team": float(df["team"].median()),
        "over_time_scaled": float(df["over_time_scaled"].median()),
    }


def build_model_input(day, quarter, dept, team, wip, workers, style_change, smv, incentive, overtime, idle_time, idle_men):
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)

    numeric_features = {
        "team": team,
        "smv": smv,
        "wip": wip,
        "incentive": incentive,
        "idle_time": idle_time,
        "idle_men": idle_men,
        "no_of_workers": workers,
        "over_time_scaled": overtime,
    }

    for col, val in numeric_features.items():
        if col in input_df.columns:
            input_df.loc[0, col] = val

    quarter_col = f"quarter_{quarter}"
    dept_col = f"department_{dept.lower()}"
    day_col = f"day_{day}"
    style_col = f"no_of_style_change_{style_change}"

    for col in [quarter_col, dept_col, day_col, style_col]:
        if col in input_df.columns:
            input_df.loc[0, col] = 1

    return input_df[model_columns]


def validate_inputs(values, ranges):
    issues = []
    for key, value in values.items():
        if key in ranges:
            min_v, max_v = ranges[key]
            if value < min_v or value > max_v:
                issues.append(f"{key} must stay between {min_v:.2f} and {max_v:.2f}.")
    return issues


def get_recommendations(result, wip, incentive, idle_time, idle_men, workers, medians):
    tips = []

    if idle_time > medians["idle_time"]:
        tips.append("Reduce idle time by checking downtime causes, line balancing, or machine interruptions.")
    if idle_men > medians["idle_men"]:
        tips.append("Review workforce allocation because idle workers are above the typical level.")
    if wip > medians["wip"]:
        tips.append("High WIP may slow workflow visibility, so consider tightening work-in-progress control.")
    if incentive < medians["incentive"]:
        tips.append("Consider whether the current incentive level is sufficient to support target performance.")
    if workers < medians["no_of_workers"]:
        tips.append("A lower worker count may limit output capacity for the current task setting.")

    if result == "High" and not tips:
        tips.append("The current configuration looks healthy. Focus on keeping idle time low and workflow stable.")
    elif result == "Moderate" and len(tips) < 2:
        tips.append("Moderate results suggest operations are acceptable, but one or two adjustments may improve consistency.")
    elif result == "Low" and len(tips) < 2:
        tips.append("The result indicates a weaker operating condition. Prioritize reducing idleness and reviewing workforce or WIP settings.")

    return tips[:4]


def save_history(record):
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    st.session_state.prediction_history.insert(0, record)
    st.session_state.prediction_history = st.session_state.prediction_history[:10]


# ==========================================
# SIDEBAR
# ==========================================
ranges = get_reference_ranges(reference_df)
medians = get_reference_medians(reference_df)

st.sidebar.title("Prototype Controls")
st.sidebar.caption("Professional demo version for presentation")

preset = st.sidebar.selectbox(
    "Quick preset",
    ["Custom Input", "Balanced Setup", "High Performance Setup", "Risky Setup"]
)

show_reference = st.sidebar.toggle("Show dataset reference panel", value=True)
show_history = st.sidebar.toggle("Show prediction history", value=True)

if st.sidebar.button("Clear prediction history", use_container_width=True):
    st.session_state.prediction_history = []

st.sidebar.markdown("### Model Scope")
st.sidebar.write("This prototype predicts **Low, Moderate, or High productivity** using a trained **Gradient Boosting** model.")

with st.sidebar.expander("Training-compatible categories"):
    st.write("Days: Monday, Tuesday, Wednesday, Thursday, Saturday, Sunday")
    st.write("Departments: Sewing, Finishing")
    st.write("Quarter: Quarter1 to Quarter5")
    st.write("Style Change: 0, 1, 2")

# ==========================================
# HEADER
# ==========================================
st.markdown('<div class="main-title">🏭 Garment Productivity Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">A cleaner prototype with stronger presentation value, better validation, richer outputs, and operational recommendations.</div>',
    unsafe_allow_html=True
)

top1, top2, top3, top4 = st.columns(4)
top1.metric("Model Type", "Gradient Boosting")
top2.metric("Output Classes", "3 Levels")
top3.metric("Input Features", "20 Columns")
top4.metric("Status", "Ready")

# ==========================================
# PRESET VALUES
# ==========================================
preset_values = {
    "Custom Input": {
        "day": "Monday", "quarter": "Quarter1", "dept": "Sewing", "team": 6,
        "wip": 586, "workers": 34, "style_change": 0, "smv": 15.26,
        "incentive": 0, "overtime": -0.18, "idle_time": 0, "idle_men": 0
    },
    "Balanced Setup": {
        "day": "Tuesday", "quarter": "Quarter2", "dept": "Sewing", "team": 6,
        "wip": 650, "workers": 34, "style_change": 0, "smv": 15.0,
        "incentive": 30, "overtime": 0.0, "idle_time": 0, "idle_men": 0
    },
    "High Performance Setup": {
        "day": "Wednesday", "quarter": "Quarter3", "dept": "Sewing", "team": 5,
        "wip": 300, "workers": 45, "style_change": 0, "smv": 10.0,
        "incentive": 200, "overtime": 0.2, "idle_time": 0, "idle_men": 0
    },
    "Risky Setup": {
        "day": "Thursday", "quarter": "Quarter4", "dept": "Finishing", "team": 10,
        "wip": 1800, "workers": 20, "style_change": 2, "smv": 30.0,
        "incentive": 0, "overtime": -0.5, "idle_time": 45, "idle_men": 10
    }
}
defaults = preset_values[preset]

# ==========================================
# INPUT SECTION
# ==========================================
tab1, tab2 = st.tabs(["Prediction Dashboard", "About the Prototype"])

with tab1:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("📅 Context Data")
        day = st.selectbox("Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"],
                           index=["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"].index(defaults["day"]))
        quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"],
                               index=["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"].index(defaults["quarter"]))
        dept = st.selectbox("Department", ["Sewing", "Finishing"],
                            index=["Sewing", "Finishing"].index(defaults["dept"]))
        team = st.slider("Team Number", int(ranges["team"][0]), int(ranges["team"][1]), int(defaults["team"]))
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("⚙️ Operational Inputs")
        wip = st.number_input("Work in Progress (WIP)", min_value=int(ranges["wip"][0]), max_value=int(ranges["wip"][1]), value=int(defaults["wip"]), step=10)
        workers = st.number_input("Number of Workers", min_value=int(ranges["no_of_workers"][0]), max_value=int(ranges["no_of_workers"][1]), value=int(defaults["workers"]), step=1)
        style_change = st.selectbox("Number of Style Changes", [0, 1, 2], index=[0, 1, 2].index(defaults["style_change"]))
        smv = st.number_input("SMV (Standard Minute Value)", min_value=float(ranges["smv"][0]), max_value=float(ranges["smv"][1]), value=float(defaults["smv"]), step=0.1)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("💰 Performance Metrics")
        incentive = st.number_input("Incentive Amount", min_value=int(ranges["incentive"][0]), max_value=int(ranges["incentive"][1]), value=int(defaults["incentive"]), step=10)
        overtime = st.slider("Overtime (Scaled)", float(ranges["over_time_scaled"][0]), float(ranges["over_time_scaled"][1]), float(defaults["overtime"]), step=0.01)
        idle_time = st.number_input("Idle Time (Minutes)", min_value=int(ranges["idle_time"][0]), max_value=int(ranges["idle_time"][1]), value=int(defaults["idle_time"]), step=1)
        idle_men = st.number_input("Idle Workers", min_value=int(ranges["idle_men"][0]), max_value=int(ranges["idle_men"][1]), value=int(defaults["idle_men"]), step=1)
        st.markdown('</div>', unsafe_allow_html=True)

    input_values = {
        "team": team,
        "wip": wip,
        "no_of_workers": workers,
        "smv": smv,
        "incentive": incentive,
        "over_time_scaled": overtime,
        "idle_time": idle_time,
        "idle_men": idle_men
    }
    issues = validate_inputs(input_values, ranges)

    if issues:
        for issue in issues:
            st.warning(issue)

    action_col1, action_col2 = st.columns([2, 1])
    run_prediction = action_col1.button("Run Productivity Prediction", use_container_width=True, type="primary")
    show_input_table = action_col2.button("Preview Encoded Inputs", use_container_width=True)

    encoded_df = build_model_input(day, quarter, dept, team, wip, workers, style_change, smv, incentive, overtime, idle_time, idle_men)

    if show_input_table:
        with st.expander("Encoded model input", expanded=True):
            st.dataframe(encoded_df, use_container_width=True)

    if run_prediction and not issues:
        prediction_idx = int(model.predict(encoded_df)[0])
        probs = model.predict_proba(encoded_df)[0] if hasattr(model, "predict_proba") else None
        result = LABELS[prediction_idx]
        confidence = float(probs[prediction_idx]) if probs is not None else None

        save_history({
            "Prediction": result,
            "Confidence": f"{confidence:.2%}" if confidence is not None else "N/A",
            "Day": day,
            "Department": dept,
            "WIP": wip,
            "Workers": workers,
            "Incentive": incentive,
            "Idle Time": idle_time,
            "Idle Men": idle_men
        })

        st.markdown("### Prediction Result")
        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        r1, r2, r3 = st.columns([1.4, 1, 1])
        with r1:
            st.markdown(f"## {LABEL_EMOJI[result]} {result} Productivity")
            st.write(LABEL_TEXT[result])
        with r2:
            st.metric("Predicted Class", result)
        with r3:
            if confidence is not None:
                st.metric("Model Confidence", f"{confidence:.2%}")
            else:
                st.metric("Model Confidence", "Unavailable")

        st.markdown('</div>', unsafe_allow_html=True)

        detail1, detail2 = st.columns([1.15, 1])

        with detail1:
            st.markdown("#### Probability Distribution")
            if probs is not None:
                prob_df = pd.DataFrame({
                    "Productivity Level": ["Low", "Moderate", "High"],
                    "Probability": probs
                })
                st.dataframe(
                    prob_df.assign(Probability=lambda x: x["Probability"].map(lambda y: f"{y:.2%}")),
                    use_container_width=True,
                    hide_index=True
                )
                for level, prob in zip(["Low", "Moderate", "High"], probs):
                    st.write(f"**{level}**")
                    st.progress(float(prob))
            else:
                st.info("Probability output is not available for this model.")

        with detail2:
            st.markdown("#### Operational Recommendations")
            for tip in get_recommendations(result, wip, incentive, idle_time, idle_men, workers, medians):
                st.write(f"• {tip}")

            st.markdown("#### Input Summary")
            summary_df = pd.DataFrame({
                "Input": ["Day", "Quarter", "Department", "Team", "WIP", "Workers", "Style Changes", "SMV", "Incentive", "Overtime", "Idle Time", "Idle Workers"],
                "Value": [day, quarter, dept, team, wip, workers, style_change, smv, incentive, overtime, idle_time, idle_men]
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

    if show_reference:
        st.markdown("### Dataset Reference Panel")
        ref1, ref2 = st.columns(2)

        with ref1:
            ref_df = pd.DataFrame({
                "Feature": ["WIP", "Workers", "SMV", "Incentive", "Idle Time", "Idle Men", "Overtime (Scaled)"],
                "Typical Median": [
                    medians["wip"], medians["no_of_workers"], round(medians["smv"], 2), medians["incentive"],
                    medians["idle_time"], medians["idle_men"], round(medians["over_time_scaled"], 2)
                ]
            })
            st.dataframe(ref_df, use_container_width=True, hide_index=True)

        with ref2:
            if reference_df is not None:
                class_counts = reference_df["productivity_level"].value_counts().reindex(["Low", "Moderate", "High"]).fillna(0).astype(int)
                class_df = pd.DataFrame({
                    "Productivity Level": class_counts.index,
                    "Records": class_counts.values
                })
                st.dataframe(class_df, use_container_width=True, hide_index=True)
                st.caption("This panel helps explain the training data distribution during presentation.")
            else:
                st.info("Reference dataset not found. Add the CSV file to enable this panel.")

    if show_history:
        st.markdown("### Recent Prediction History")
        history = st.session_state.get("prediction_history", [])
        if history:
            st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
        else:
            st.info("No predictions have been run yet.")

with tab2:
    st.subheader("What improved in this version?")
    st.write("This upgraded prototype is designed to look more professional and be easier to explain during a presentation.")
    st.write("Key improvements include a cleaner dashboard layout, preset scenarios, better validation, probability distribution output, operational recommendations, encoded input preview, and recent prediction history.")
    st.write("The interface also avoids unsupported categories such as Friday, since the trained model was not built with that day in its encoded columns.")
    st.write("For best deployment results, make sure app.py, gbm_model.pkl, gbm_model_columns.pkl, and cleaned_garments_worker_productivity.csv are stored in the same folder.")
