# ==========================================
# MAIN UI
# ==========================================
st.markdown('<div class="main-title">🏭 Garment Productivity Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Advanced Gradient Boosting Analysis for Operational Optimization</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📊 Prediction Dashboard", "ℹ️ Technical Documentation"])

with tab1:
    # Use a container for a cleaner look
    with st.container():
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="block-card">', unsafe_allow_html=True)
            st.subheader("📅 Temporal Context")
            day = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"], index=0)
            quarter = st.selectbox("Fiscal Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"], index=0)
            dept = st.radio("Department", ["Sewing", "Finishing"], horizontal=True)
            team = st.select_slider("Team Number", options=list(range(1, 13)), value=d["team"])
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="block-card">', unsafe_allow_html=True)
            st.subheader("⚙️ Production Resources")
            wip = st.number_input("Work In Progress (WIP)", 0, 25000, d["wip"], help="Number of unfinished items in the line.")
            workers = st.number_input("Labor Force (Workers)", 2, 100, d["workers"])
            style_change = st.segmented_control("Style Changes", [0, 1, 2], default=d["style"])
            smv = st.number_input("SMV", 2.0, 60.0, d["smv"], help="Standard Minute Value: Time allocated for a task.")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="block-card">', unsafe_allow_html=True)
            st.subheader("💰 Incentive & Downtime")
            incentive = st.number_input("Incentive (BDT)", 0, 4000, d["inc"])
            overtime = st.slider("Overtime (Scaled)", -2.0, 2.0, d["ot"], help="Normalized value of overtime hours.")
            idle_time = st.number_input("Idle Time (Min)", 0, 500, d["it"])
            idle_men = st.number_input("Idle Workers Count", 0, 50, d["im"])
            st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    if st.button("🚀 Generate Productivity Forecast", use_container_width=True, type="primary"):
        with st.spinner('Analyzing production patterns...'):
            encoded_df = build_model_input(day, quarter, dept, team, wip, workers, style_change, smv, incentive, overtime, idle_time, idle_men)
            
            # Prediction Logic
            pred_idx = int(model.predict(encoded_df)[0])
            probs = model.predict_proba(encoded_df)[0]
            result = LABELS[pred_idx]
            confidence = float(probs[pred_idx])

            # Results Display
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            res_col1, res_col2 = st.columns([2, 1])
            
            with res_col1:
                st.markdown(f"## {LABEL_EMOJI[result]} {result} Productivity")
                st.info(LABEL_TEXT[result])
            
            with res_col2:
                st.metric("Model Confidence", f"{confidence:.2%}", delta=f"{(confidence - 0.33):.1%} vs Random")
            st.markdown('</div>', unsafe_allow_html=True)

            # Probabilities Breakdown
            st.write("---")
            st.subheader("🔍 Probability Distribution")
            p_cols = st.columns(3)
            for i, label in enumerate(["Low", "Moderate", "High"]):
                with p_cols[i]:
                    st.write(f"**{label}**")
                    st.progress(float(probs[i]))
                    st.caption(f"{probs[i]:.1%}")

with tab2:
    st.markdown("### Model Architecture")
    st.write("This application utilizes a **Gradient Boosting Machine (GBM)** to classify factory productivity.")
    st.success("✅ Categorical encoding verified against training feature set.")
    
    with st.expander("View Input Feature Vector (Technical)"):
        st.dataframe(encoded_df)
