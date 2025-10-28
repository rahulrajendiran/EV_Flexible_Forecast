import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# ---------- PAGE CONFIG ----------
st.set_page_config(layout="wide", page_title="EV Flex Forecast")

MODELS_DIR = "models"
RESULTS_DIR = "results"
PROCESSED_PATH = os.path.join(RESULTS_DIR, "processed_ev_data.csv")

# ---------- HEADER ----------
st.title("🔋 EV Flexible Regulation Forecast — Interactive Dashboard")
st.markdown(
    """
    A minimalistic interface for forecasting **Flexible kW** using trained ML and probabilistic models.
    You can view predictions from uploaded data or manually enter parameters to get real-time forecasts.
    """
)

# ---------- LOAD DATA ----------
uploaded = st.file_uploader("📂 Upload processed_ev_data.csv (optional)", type="csv")
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.success("✅ Uploaded dataset loaded.")
elif os.path.exists(PROCESSED_PATH):
    df = pd.read_csv(PROCESSED_PATH)
    st.info("ℹ️ Loaded processed dataset from results/")
else:
    st.warning("⚠️ No processed dataset found. Please upload a processed CSV or run preprocessing first.")
    df = pd.DataFrame()  # empty placeholder

# ---------- SIDEBAR ----------
st.sidebar.header("⚙️ Dashboard Controls")
page = st.sidebar.radio("Choose View", ["Dataset Forecast", "Manual Forecast"])

# ---------- LOAD MODELS ----------
point_model_path = os.path.join(MODELS_DIR, "lightgbm_point_model.pkl")
q10_path = os.path.join(MODELS_DIR, "quantile_q10.pkl")
q50_path = os.path.join(MODELS_DIR, "quantile_q50.pkl")
q90_path = os.path.join(MODELS_DIR, "quantile_q90.pkl")

features = ['duration_min', 'Energy Consumed (kWh)', 'start_hour', 'day_of_week', 'is_weekend']


# ========== PAGE 1: DATASET FORECAST ==========
if page == "Dataset Forecast":
    if df.empty:
        st.stop()

    required = ["Charging Start Time","Charging End Time","Energy Consumed (kWh)",
                "duration_min","start_hour","day_of_week","is_weekend","flexible_kW"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns in dataset: {missing}")
        st.stop()

    st.subheader("📊 Dataset Sample")
    st.dataframe(df.head())

    model_choice = st.sidebar.selectbox("Model Type", ["LightGBM (point)", "Probabilistic (Q10/Q50/Q90)"])
    n_display = st.sidebar.slider("Show last N test rows", 20, 1000, 200)

    if model_choice == "LightGBM (point)":
        if not os.path.exists(point_model_path):
            st.error("Point model not found.")
            st.stop()
        model = joblib.load(point_model_path)
        preds = model.predict(df[features])
        df_out = df.copy()
        df_out["LightGBM_Pred"] = preds

        st.subheader("📈 Predictions (last rows)")
        st.dataframe(df_out[["Charging Start Time","Energy Consumed (kWh)","duration_min","flexible_kW","LightGBM_Pred"]].tail(n_display))

        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(df_out["flexible_kW"].tail(n_display).values, label="True", linewidth=1.5)
        ax.plot(df_out["LightGBM_Pred"].tail(n_display).values, label="Pred", alpha=0.9)
        ax.legend()
        ax.set_title("True vs LightGBM Prediction (last N)")
        st.pyplot(fig)

    else:
        if not (os.path.exists(q10_path) and os.path.exists(q50_path) and os.path.exists(q90_path)):
            st.error("Quantile models not found.")
            st.stop()
        q10 = joblib.load(q10_path)
        q50 = joblib.load(q50_path)
        q90 = joblib.load(q90_path)

        pred_q10 = q10.predict(df[features])
        pred_q50 = q50.predict(df[features])
        pred_q90 = q90.predict(df[features])

        dfp = df.copy()
        dfp["Q10"] = pred_q10
        dfp["Q50"] = pred_q50
        dfp["Q90"] = pred_q90

        st.subheader("🎯 Probabilistic Predictions (last rows)")
        st.dataframe(dfp[["Charging Start Time","Energy Consumed (kWh)","duration_min","flexible_kW","Q10","Q50","Q90"]].tail(n_display))

        h = min(n_display, len(dfp))
        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(dfp["flexible_kW"].tail(h).values, label="True", color="k")
        ax.plot(dfp["Q50"].tail(h).values, label="Median (Q50)")
        ax.fill_between(range(h), dfp["Q10"].tail(h).values, dfp["Q90"].tail(h).values, alpha=0.2, label="Q10–Q90")
        ax.legend()
        ax.set_title("Probabilistic prediction intervals (last N)")
        st.pyplot(fig)


# ========== PAGE 2: MANUAL FORECAST ==========
elif page == "Manual Forecast":
    st.subheader("🧭 Manual Flexible kW Prediction")
    st.markdown("Enter the EV charging session parameters to generate a real-time flexible power forecast.")

    with st.form("manual_form"):
        col1, col2 = st.columns(2)
        with col1:
            start_time = st.text_input("Charging Start Time (YYYY-MM-DD HH:MM)", "")
            end_time = st.text_input("Charging End Time (YYYY-MM-DD HH:MM)", "")
            energy = st.number_input("Energy Consumed (kWh)", min_value=0.0, value=10.0, step=0.1)
            duration_min = st.number_input("Duration (minutes)", min_value=0.0, value=60.0, step=1.0)
        with col2:
            start_hour = st.slider("Start Hour (0–23)", 0, 23, 10)
            day_of_week = st.selectbox("Day of Week (0=Mon ... 6=Sun)", list(range(7)), index=0)
            is_weekend = st.selectbox("Is Weekend?", [0, 1], index=0)

        model_choice_manual = st.radio("Select Model", ["LightGBM (point)", "Probabilistic (Q10/Q50/Q90)"])
        submit = st.form_submit_button("🔮 Predict Flexible kW")

    if submit:
        input_data = pd.DataFrame([{
            "duration_min": duration_min,
            "Energy Consumed (kWh)": energy,
            "start_hour": start_hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend
        }])

        if model_choice_manual == "LightGBM (point)":
            if not os.path.exists(point_model_path):
                st.error("Model not found.")
            else:
                model = joblib.load(point_model_path)
                pred = model.predict(input_data)[0]
                st.success(f"⚡ Predicted Flexible kW (LightGBM): **{pred:.3f} kW**")

        else:
            if not (os.path.exists(q10_path) and os.path.exists(q50_path) and os.path.exists(q90_path)):
                st.error("Quantile models missing.")
            else:
                q10 = joblib.load(q10_path)
                q50 = joblib.load(q50_path)
                q90 = joblib.load(q90_path)
                p10 = q10.predict(input_data)[0]
                p50 = q50.predict(input_data)[0]
                p90 = q90.predict(input_data)[0]
                st.success(f"🔹 Predicted Range (Q10–Q90): **{p10:.3f} – {p90:.3f} kW** (Median: {p50:.3f} kW)")

        st.markdown("---")
        st.markdown("✅ **Interpretation:** The prediction provides an estimated flexible power demand for the session. Probabilistic forecasts include uncertainty ranges for better grid planning.")
