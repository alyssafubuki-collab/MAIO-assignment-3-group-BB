from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import json
import os

MODEL_PATH = "model/model.joblib"
SCALER_PATH = "model/scaler.joblib"
METRICS_PATH = "model/metrics.json"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ model.joblib introuvable. Exécute d'abord train.py ou train_v2_rf.py")
if not os.path.exists(METRICS_PATH):
    raise FileNotFoundError("❌ metrics.json introuvable. Exécute d'abord train.py ou train_v2_rf.py")

# Chargement du modèle et des métadonnées
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

with open(METRICS_PATH) as f:
    metrics = json.load(f)

app = FastAPI(title="Virtual Diabetes Clinic Triage API")

# ----- Pydantic model for input validation -----
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

# ----- Health endpoint -----
@app.get("/health")
def health():
    return {"status": "ok", "model_version": metrics.get("model_version", "unknown")}

# ----- Predict endpoint -----
@app.post("/predict")
def predict(features: PatientFeatures):
    try:
        X_raw = np.array([[features.age, features.sex, features.bmi, features.bp,
                           features.s1, features.s2, features.s3, features.s4,
                           features.s5, features.s6]], dtype=float)
        
        if scaler is not None:
            X = scaler.transform(X_raw)
        else:
            X = X_raw

        pred = float(model.predict(X)[0])
        response = {"prediction": pred}

        # Ajout du flag "high_risk" si le threshold existe dans metrics.json
        if "high_risk_threshold" in metrics:
            threshold = float(metrics["high_risk_threshold"])
            response["high_risk"] = pred >= threshold

        return response

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input or prediction error: {str(e)}")
