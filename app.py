"""Streamlit app for a simple Smart Agriculture prototype."""

from __future__ import annotations

import os
import pickle
from typing import Dict, List, Tuple

import pandas as pd
import requests
import streamlit as st

from train_model import MODEL_PATH, generate_synthetic_data, train_and_save_model


FEATURES: List[str] = [
    "temperature",
    "humidity",
    "soil_moisture",
    "nitrogen",
    "phosphorus",
    "potassium",
    "pump_status",
]


def ensure_model() -> None:
    """Create and save model.pkl if missing."""
    if not os.path.exists(MODEL_PATH):
        df = generate_synthetic_data(rows=1000, random_state=42)
        train_and_save_model(df, model_path=MODEL_PATH)


def load_model():
    """Load trained model from disk."""
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def fetch_latest_data(channel_id: str, read_api_key: str) -> Dict[str, float]:
    """Fetch latest sensor values from ThingSpeak."""
    url = (
        f"https://api.thingspeak.com/channels/{channel_id}/feeds/last.json"
        f"?api_key={read_api_key}"
    )
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()

    # Map ThingSpeak fields to our model feature names.
    # field1..field7 follow user sensor order.
    mapped = {
        "temperature": float(data.get("field1") or 0),
        "humidity": float(data.get("field2") or 0),
        "soil_moisture": float(data.get("field3") or 0),
        "nitrogen": float(data.get("field4") or 0),
        "phosphorus": float(data.get("field5") or 0),
        "potassium": float(data.get("field6") or 0),
        "pump_status": int(float(data.get("field7") or 0)),
    }
    return mapped


def fetch_recent_moisture(channel_id: str, read_api_key: str, results: int = 20) -> pd.DataFrame:
    """Fetch recent soil moisture values for a simple trend chart."""
    url = (
        f"https://api.thingspeak.com/channels/{channel_id}/feeds.json"
        f"?api_key={read_api_key}&results={results}"
    )
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    payload = response.json()

    rows = []
    for item in payload.get("feeds", []):
        rows.append(
            {
                "timestamp": item.get("created_at"),
                "soil_moisture": float(item.get("field3") or 0),
            }
        )

    history_df = pd.DataFrame(rows)
    if not history_df.empty:
        history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
        history_df = history_df.set_index("timestamp")
    return history_df


def explain_prediction(action: str, values: Dict[str, float]) -> str:
    """Provide farmer-friendly explanation text."""
    if action == "Turn Pump ON":
        return "Soil moisture is low. Turn ON irrigation pump."
    if action == "Turn Pump OFF":
        return "Soil moisture is high. Turn OFF irrigation pump to avoid overwatering."
    if action == "Add Nitrogen":
        return "Nitrogen is low. Add nitrogen fertilizer to support plant growth."
    if action == "Add Phosphorus":
        return "Phosphorus is low. Add phosphorus fertilizer for root and flower development."
    if action == "Add Potassium":
        return "Potassium is low. Add potassium fertilizer to improve crop strength."
    return "All key values are in a healthy range. No action needed right now."


def predict_action(model, sensor_values: Dict[str, float]) -> Tuple[str, str]:
    """Run model inference and return action + explanation."""
    input_df = pd.DataFrame([sensor_values])[FEATURES]
    prediction = model.predict(input_df)[0]
    explanation = explain_prediction(prediction, sensor_values)
    return prediction, explanation


def main() -> None:
    st.set_page_config(page_title="Smart Agriculture Prototype", layout="centered")
    st.title("🌱 Smart Agriculture System (Prototype)")
    st.caption("IoT Device → ThingSpeak → Decision Tree → Streamlit")

    # Auto-refresh every 20 seconds.
    st.markdown(
        """
        <script>
        setTimeout(function(){ window.location.reload(); }, 20000);
        </script>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar config lets you quickly run local demos.
    st.sidebar.header("ThingSpeak Configuration")
    channel_id = st.sidebar.text_input(
        "CHANNEL_ID", value=os.getenv("CHANNEL_ID", ""), help="Your ThingSpeak channel ID"
    )
    read_api_key = st.sidebar.text_input(
        "READ_API_KEY",
        value=os.getenv("READ_API_KEY", ""),
        type="password",
        help="Your ThingSpeak READ API key",
    )

    ensure_model()
    model = load_model()

    if not channel_id or not read_api_key:
        st.info("Please enter CHANNEL_ID and READ_API_KEY in the sidebar.")
        return

    try:
        latest_values = fetch_latest_data(channel_id, read_api_key)
        action, explanation = predict_action(model, latest_values)

        st.subheader("Current Sensor Values")
        col1, col2 = st.columns(2)
        col1.metric("Temperature (°C)", f"{latest_values['temperature']:.1f}")
        col2.metric("Humidity (%)", f"{latest_values['humidity']:.1f}")
        col1.metric("Soil Moisture (%)", f"{latest_values['soil_moisture']:.1f}")
        col2.metric("Pump Status", "ON" if latest_values["pump_status"] == 1 else "OFF")
        col1.metric("Nitrogen", f"{latest_values['nitrogen']:.1f}")
        col2.metric("Phosphorus", f"{latest_values['phosphorus']:.1f}")
        st.metric("Potassium", f"{latest_values['potassium']:.1f}")

        st.subheader("AI Recommendation")
        st.success(f"Predicted Action: **{action}**")
        st.write(explanation)

        st.subheader("Recent Soil Moisture Trend")
        history_df = fetch_recent_moisture(channel_id, read_api_key, results=20)
        if not history_df.empty:
            st.line_chart(history_df["soil_moisture"])
        else:
            st.warning("No recent moisture data available from ThingSpeak.")

    except Exception as exc:
        st.error(f"Could not fetch data or make prediction: {exc}")


if __name__ == "__main__":
    main()
