from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import json
import os

# Charger modèle
MODEL_PATH = "model/model.joblib"
METRICS_PATH = "model/metrics.json"

if not os.path.exists(MODEL_PATH) or not os.path.exists(METRICS_PATH):
    raise FileNotFoundError("model.joblib ou metrics.json introuvable")

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
    return {"status": "ok", "model_version": metrics["model_version"]}

@app.post("/predict")
def predict(features: PatientFeatures):
    try:
        X = np.array([[features.age, features.sex, features.bmi, features.bp,
                       features.s1, features.s2, features.s3, features.s4,
                       features.s5, features.s6]])
        pred = model.predict(X)[0]
        return {"prediction": float(pred)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
