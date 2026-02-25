# Smart Agriculture Prototype (ThingSpeak + ML + Streamlit)

This project is a simple academic prototype:

1. Fetch latest real-time sensor values from ThingSpeak.
2. Generate synthetic historical data for model training.
3. Train a `DecisionTreeClassifier`.
4. Predict farmer actions in real time.
5. Show everything in a clean Streamlit dashboard.

## Sensor Mapping (ThingSpeak fields)

- `field1` → Temperature
- `field2` → Humidity
- `field3` → Soil Moisture
- `field4` → Nitrogen
- `field5` → Phosphorus
- `field6` → Potassium
- `field7` → Pump Status (`0` OFF, `1` ON)

## Setup

```bash
pip install -r requirements.txt
```

## 1) Train Model

```bash
python train_model.py
```

This generates:
- `synthetic_training_data.csv`
- `model.pkl`

## 2) Run Streamlit App

```bash
streamlit run app.py
```

Inside Streamlit sidebar, enter:
- `CHANNEL_ID`
- `READ_API_KEY`

The app auto-refreshes every 20 seconds.

## Synthetic Label Rules

- If soil moisture < 30 → Turn Pump ON
- If soil moisture > 70 → Turn Pump OFF
- If nitrogen < 40 → Add Nitrogen
- If phosphorus < 40 → Add Phosphorus
- If potassium < 40 → Add Potassium
- Else → No Action Needed
