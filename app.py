# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import os
from src.predict import Predictor

MODEL_PATH = "/content/model_data/knn_weighted.joblib"
SCALER_PATH = "/content/model_data/scaler.joblib"

app = FastAPI(title="Auto MPG predictor (KNN)")

class CarFeatures(BaseModel):
    displacement: float
    horsepower: float
    weight: float
    acceleration: float
    cylinders: int

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise RuntimeError("Model or scaler not found. Ensure model_data contains saved artifacts.")

predictor = Predictor(MODEL_PATH, SCALER_PATH)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Auto MPG predictor. POST to /predict with feature JSON."}

@app.post("/predict")
def predict(payload: CarFeatures):
    features = payload.dict()
    pred = predictor.predict_one(features)
    return {"predicted_mpg": pred}
