# app/app.py
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import json
import os

app = FastAPI(title="Virtual Diabetes Clinic API")

MODEL_PATH = "model/model.joblib"
METRICS_PATH = "model/metrics.json"

# Load model and metrics at startup
if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Model file not found. Please train the model first.")
model_bundle = joblib.load(MODEL_PATH)
model = model_bundle["model"]
scaler = model_bundle["scaler"]

with open(METRICS_PATH) as f:
    metrics = json.load(f)
MODEL_VERSION = metrics.get("model_version", "unknown")

@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}

@app.post("/predict")
def predict(features: dict):
    try:
        x = np.array([[features[k] for k in sorted(features.keys())]])
        x_scaled = scaler.transform(x)
        pred = model.predict(x_scaled)[0]
        return {"prediction": float(pred)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
