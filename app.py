# ==========================================
# HELPERS (Refined for 192-column mapping)
# ==========================================
def build_model_input(day, quarter, dept, team, wip, workers, style_change, smv, incentive, overtime, idle_time, idle_men):
    # Initialize a zero-filled DataFrame with all 192 columns
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)

    # 1. Map Numeric Features
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
            input_df.at[0, col] = val

    # 2. Map Categorical Features (One-Hot Logic)
    # We match the specific naming convention used during your model training
    cat_targets = [
        f"quarter_{quarter}",
        f"department_{dept.lower()}",
        f"day_{day}",
        f"no_of_style_change_{int(style_change)}" 
    ]

    for col in cat_targets:
        if col in input_df.columns:
            input_df.at[0, col] = 1
        else:
            # Helpful for debugging hidden mismatches in your 192 columns
            st.sidebar.warning(f"Feature not found in model: {col}")

    return input_df[model_columns]

# ==========================================
# MAIN UI (Refined for Academic Polish)
# ==========================================
st.markdown('<div class="main-title">🏭 Garment Productivity Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Predictive Analytics Prototype | Aras Tinggi Data Analysis</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Prediction Dashboard", "📈 Model Insights", "ℹ️ About"])

with tab1:
    # Organize inputs into three distinct functional groups
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("📅 Operational Context")
        day = st.selectbox("Working Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])
        quarter = st.selectbox("Fiscal Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
        dept = st.radio("Department Type", ["Sewing", "Finishing"], horizontal=True)
        team = st.number_input("Team Identification", 1, 30, d["team"]) # Adjust max based on your data
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("⚙️ Resource Allocation")
        wip = st.number_input("Work In Progress (WIP)", 0, 25000, d["wip"])
        workers = st.number_input("Total Workers", 2, 120, d["workers"])
        style_change = st.select_slider("Style Changes", options=[0, 1, 2], value=d["style"])
        smv = st.number_input("SMV (Workload Complexity)", 2.0, 60.0, d["smv"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("💰 Efficiency Metrics")
        incentive = st.number_input("Financial Incentive", 0, 5000, d["inc"])
        overtime = st.slider("Overtime (Normalized)", -2.0, 2.0, d["ot"])
        idle_time = st.number_input("Idle Time (Minutes)", 0, 600, d["it"])
        idle_men = st.number_input("Idle Laborers", 0, 100, d["im"])
        st.markdown('</div>', unsafe_allow_html=True)

    # Execution Button
    if st.button("🚀 Run Prediction Model", use_container_width=True, type="primary"):
        input_data = build_model_input(day, quarter, dept, team, wip, workers, style_change, smv, incentive, overtime, idle_time, idle_men)
        
        # Inference
        pred_idx = int(model.predict(input_data)[0])
        probs = model.predict_proba(input_data)[0]
        result = LABELS[pred_idx]
        confidence = float(probs[pred_idx])

        # Result Presentation
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        r_col1, r_col2 = st.columns([2, 1])
        with r_col1:
            st.markdown(f"### Predicted Level: {LABEL_EMOJI[result]} **{result}**")
            st.write(f"**Analysis:** {LABEL_TEXT[result]}")
        with r_col2:
            st.metric("Model Confidence", f"{confidence:.2%}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Probability Breakdown Visualization
        st.write("#### Confidence Distribution")
        p1, p2, p3 = st.columns(3)
        for i, label in enumerate(["Low", "Moderate", "High"]):
            with [p1, p2, p3][i]:
                st.write(f"{label}")
                st.progress(float(probs[i]))
                st.caption(f"{probs[i]:.2%}")

with tab2:
    st.subheader("Feature Analysis")
    st.write("This dashboard analyzes 192 distinct features to determine the most likely productivity outcome.")
    # Show the user what is actually being sent to the model (for academic transparency)
    with st.expander("View Encoded Input Vector (1x192)"):
        st.dataframe(input_data)

with tab3:
    st.info("System built for operational optimization in garment manufacturing using Gradient Boosting.")
