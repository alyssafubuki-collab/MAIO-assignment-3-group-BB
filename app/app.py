from fastapi import FastAPI, HTTPException
import joblib
import os
import json
import pandas as pd

app = FastAPI()

MODEL_PATH = "model/model.joblib"
METRICS_PATH = "model/metrics.json"

# Load model and metrics
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} not found. Run train.py first.")
model = joblib.load(MODEL_PATH)

with open(METRICS_PATH) as f:
    metrics = json.load(f)

@app.get("/health")
def health():
    return {"status": "ok", "model_version": metrics.get("model_version", "unknown")}

@app.post("/predict")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        return {"prediction": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
