from fastapi import FastAPI, HTTPException
import joblib
import os
import json
import numpy as np

app = FastAPI()

MODEL_PATH = "model/model.joblib"
METRICS_PATH = "model/metrics.json"

# Charger le modèle
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} not found. Run train.py first!")

model = joblib.load(MODEL_PATH)

# Charger les metrics
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
else:
    metrics = {"rmse": None, "model_version": "v0.1"}

@app.get("/health")
def health():
    return {"status": "ok", "model_version": metrics.get("model_version", "v0.1")}

@app.post("/predict")
def predict(payload: dict):
    try:
        features = ["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"]
        x = [payload[f] for f in features]
        x = np.array(x).reshape(1, -1)
        pred = model.predict(x)[0]
        return {"prediction": float(pred)}
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
