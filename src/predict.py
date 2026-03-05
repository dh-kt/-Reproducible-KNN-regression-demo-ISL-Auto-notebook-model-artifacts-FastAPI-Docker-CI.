# src/predict.py
import joblib
import pandas as pd
from pathlib import Path

FEATURE_ORDER = ['displacement','horsepower','weight','acceleration','cylinders']

class Predictor:
    def __init__(self, model_path: str, scaler_path: str):
        model_path = Path(model_path)
        scaler_path = Path(scaler_path)
        if not model_path.exists() or not scaler_path.exists():
            raise FileNotFoundError("Model or scaler not found. Provide correct paths.")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict_one(self, features: dict) -> float:
        row = pd.DataFrame([features], columns=FEATURE_ORDER)
        x_scaled = self.scaler.transform(row)
        pred = self.model.predict(x_scaled)
        return float(pred[0])
