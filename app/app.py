from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import json
import os

MODEL_PATH = "model/model.joblib"
METRICS_PATH = "model/metrics.json"

if not os.path.exists(MODEL_PATH) or not os.path.exists(METRICS_PATH):
    raise FileNotFoundError("model.joblib ou metrics.json introuvable. Run train.py first.")

model = joblib.load(MODEL_PATH)
with open(METRICS_PATH) as f:
    metrics = json.load(f)

app = FastAPI(title="Virtual Diabetes Clinic Triage API")

class PatientFeatures(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float

@app.get("/health")
def health():
    return {"status": "ok", "model_version": metrics.get("model_version", "unknown")}

@app.post("/predict")
def predict(features: PatientFeatures):
    try:
        X = np.array([[features.age, features.sex, features.bmi, features.bp,
                       features.s1, features.s2, features.s3, features.s4,
                       features.s5, features.s6]], dtype=float)
        pred = float(model.predict(X)[0])

        response = {"prediction": pred}

        # If the trained metrics.json contains a threshold for 'high-risk', include a boolean flag
        if "high_risk_threshold" in metrics:
            threshold = float(metrics["high_risk_threshold"])
            response["high_risk"] = pred >= threshold

        return response

    except Exception as e:
        # Retourne une erreur JSON lisible
        raise HTTPException(status_code=400, detail=f"Invalid input or prediction error: {str(e)}")
