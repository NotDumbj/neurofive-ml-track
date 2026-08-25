"""
Flight Delay Risk Estimator - Streamlit App
=============================================
Loads model_artifacts.joblib (produced by train_model.py) and lets a user
enter flight details to get a live arrival-delay risk score.

Run:
    streamlit run app.py
"""

import datetime as dt

import os
import joblib
import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_PATH = os.path.join(APP_DIR, "model_artifacts.joblib")

st.set_page_config(page_title="Flight Delay Risk Estimator", page_icon="✈️", layout="centered")


@st.cache_resource
def load_artifact():
    return joblib.load(ARTIFACT_PATH)


def engineer_features(artifact, airline, origin, dest, flight_date, dep_hour, distance):
    month = flight_date.month
    day_of_week = flight_date.isoweekday()  # 1=Mon ... 7=Sun
    is_weekend = int(day_of_week in artifact["weekend_days"])
    is_peak_season = int(month in artifact["peak_months"])

    bins = artifact["distance_tier_bins"]
    labels = artifact["distance_tier_labels"]
    distance_tier = pd.cut([distance], bins=bins, labels=labels)[0]

    route = f"{origin}_{dest}"
    route_delay_rate = artifact["route_rates"].get(route, artifact["global_prior"])

    row = pd.DataFrame([{
        "AIRLINE": airline,
        "ORIGIN": origin,
        "DEST": dest,
        "MONTH": month,
        "DAY_OF_WEEK": day_of_week,
        "DEP_HOUR": dep_hour,
        "IS_WEEKEND": is_weekend,
        "DISTANCE": distance,
        "DISTANCE_TIER": distance_tier,
        "IS_PEAK_SEASON": is_peak_season,
        "ROUTE_DELAY_RATE": route_delay_rate,
    }])
    return row


def risk_bucket(prob, threshold):
    """
    Bucket a probability into Low/Medium/High risk relative to the F1-optimal
    decision threshold found via precision-recall tuning in the training
    notebook (falls back to 0.5 if an older artifact doesn't have one saved).
    """
    if prob < threshold * 0.5:
        return "Low", "🟢"
    elif prob < threshold:
        return "Medium", "🟡"
    else:
        return "High", "🔴"


def main():
    st.title("✈️ Flight Delay Risk Estimator")
    st.caption(
        "Estimates the probability that a US domestic flight arrives 15+ minutes late, "
        "based on historical route, carrier, and seasonal delay patterns."
    )

    try:
        artifact = load_artifact()
    except FileNotFoundError:
        st.error(
            "No trained model found. Run `python train_model.py` first to generate "
            "model_artifacts.joblib, then restart the app."
        )
        st.stop()

    with st.form("flight_form"):
        col1, col2 = st.columns(2)
        with col1:
            airline = st.selectbox("Airline", options=artifact["top_carriers"])
            origin = st.selectbox("Origin airport", options=artifact["top_airports"])
            flight_date = st.date_input("Flight date", value=dt.date.today())
        with col2:
            dest_options = [d for d in artifact["top_destinations"] if d != origin] or artifact["top_destinations"]
            dest = st.selectbox("Destination airport", options=dest_options)
            dep_hour = st.slider("Scheduled departure hour (24h)", 0, 23, 9)
            distance = st.number_input("Route distance (miles)", min_value=50, max_value=6000, value=800, step=50)

        submitted = st.form_submit_button("Estimate delay risk", use_container_width=True)

    if submitted:
        if origin == dest:
            st.warning("Origin and destination can't be the same airport.")
            st.stop()

        row = engineer_features(artifact, airline, origin, dest, flight_date, dep_hour, distance)
        proba = artifact["pipeline"].predict_proba(row)[0, 1]
        base_rate = artifact["base_delay_rate"]
        # Clamp: the notebook tunes this threshold by maximizing F1 on the test
        # set, which can occasionally overfit to an extreme value on noisy data.
        # Keep it within a sane band around the base rate as a safety net.
        raw_threshold = artifact.get("recommended_threshold", 0.5)
        threshold = min(max(raw_threshold, base_rate * 0.5), 0.9)
        bucket, emoji = risk_bucket(proba, threshold)
        predicted_delayed = proba >= threshold

        st.divider()
        col_a, col_b = st.columns(2)
        col_a.metric("Predicted delay probability", f"{proba:.1%}")
        col_b.metric("Model call", "Delayed" if predicted_delayed else "On-time")
        st.markdown(f"### {emoji} Risk level: **{bucket}**")
        st.caption(
            f"Historical baseline delay rate: {base_rate:.1%}  •  "
            f"Decision threshold: {threshold:.1%} (F1-optimal, tuned on the held-out test set)"
        )
        st.progress(min(proba, 1.0))

        st.info(
            "This is a probability estimate from a historical dataset, not a guarantee. "
            "It does not account for real-time weather, air traffic control conditions, "
            "or same-day disruptions.",
            icon="ℹ️",
        )

    with st.expander("Model details & benchmark results"):
        st.write(f"**Model in production:** {artifact['best_model_name']}")
        st.write(f"**Decision threshold:** {artifact.get('recommended_threshold', 0.5):.1%} "
                 f"(chosen by maximizing F1 on the held-out test set)")

        st.markdown("**Held-out test set benchmark**")
        st.dataframe(artifact["results_df"], use_container_width=True, hide_index=True)

        if "cv_results_df" in artifact:
            st.markdown("**5-fold cross-validation (ROC-AUC)**")
            st.dataframe(artifact["cv_results_df"], use_container_width=True, hide_index=True)

        st.caption(
            "Champion model selected by highest ROC-AUC on a held-out 20% test split, "
            "cross-checked against 5-fold cross-validation. See the project README for "
            "the full methodology."
        )


if __name__ == "__main__":
    main()