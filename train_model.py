"""Train a simple DecisionTree model for Smart Agriculture recommendations."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


MODEL_PATH = "model.pkl"
DATA_PATH = "synthetic_training_data.csv"


def generate_synthetic_data(rows: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic sensor data and rule-based action labels."""
    rng = np.random.default_rng(random_state)

    df = pd.DataFrame(
        {
            "temperature": rng.uniform(15, 40, rows).round(2),
            "humidity": rng.uniform(30, 90, rows).round(2),
            "soil_moisture": rng.uniform(10, 90, rows).round(2),
            "nitrogen": rng.uniform(10, 100, rows).round(2),
            "phosphorus": rng.uniform(10, 100, rows).round(2),
            "potassium": rng.uniform(10, 100, rows).round(2),
            "pump_status": rng.integers(0, 2, rows),
        }
    )

    # Apply the rule-based labels exactly as requested.
    def label_row(row: pd.Series) -> str:
        if row["soil_moisture"] < 30:
            return "Turn Pump ON"
        if row["soil_moisture"] > 70:
            return "Turn Pump OFF"
        if row["nitrogen"] < 40:
            return "Add Nitrogen"
        if row["phosphorus"] < 40:
            return "Add Phosphorus"
        if row["potassium"] < 40:
            return "Add Potassium"
        return "No Action Needed"

    df["recommendation"] = df.apply(label_row, axis=1)
    return df


def train_and_save_model(df: pd.DataFrame, model_path: str = MODEL_PATH) -> float:
    """Train a DecisionTreeClassifier and save it to disk."""
    feature_cols = [
        "temperature",
        "humidity",
        "soil_moisture",
        "nitrogen",
        "phosphorus",
        "potassium",
        "pump_status",
    ]

    X = df[feature_cols]
    y = df["recommendation"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return accuracy


def main() -> None:
    """Generate data, train model, and save outputs."""
    synthetic_df = generate_synthetic_data(rows=1000, random_state=42)
    synthetic_df.to_csv(DATA_PATH, index=False)

    accuracy = train_and_save_model(synthetic_df, model_path=MODEL_PATH)

    print(f"Synthetic training data saved to: {DATA_PATH}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Validation accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
