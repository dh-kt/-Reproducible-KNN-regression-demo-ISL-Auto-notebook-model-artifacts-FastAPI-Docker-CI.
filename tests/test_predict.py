# tests/test_predict.py
from src.predict import Predictor
import os

def test_predictor_load_and_predict():
    model_path = os.path.join("model_data", "knn_weighted.joblib")
    scaler_path = os.path.join("model_data", "scaler.joblib")
    assert os.path.exists(model_path)
    assert os.path.exists(scaler_path)
    p = Predictor(model_path, scaler_path)
    sample = {'displacement':150.0,'horsepower':95.0,'weight':2000.0,'acceleration':15.5,'cylinders':4}
    pred = p.predict_one(sample)
    assert isinstance(pred, float)
    assert 0 < pred < 100
