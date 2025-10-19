# Virtual Diabetes Clinic Triage

Machine learning service to predict short-term diabetes progression risk.  
Built with **FastAPI**, **scikit-learn**, and deployed via **GitHub Actions** & **Docker**.

---

## Overview
Each week, nurses review patient check-ins (vitals, labs, lifestyle notes) to decide who needs a follow-up call.  
This service predicts a continuous **progression index** — higher = worse progression risk.

---

## Model Versions

| Version | Model | RMSE | Notes |
|----------|--------|------|--------|
| v0.1 | LinearRegression | 53.85 | Baseline |
| v0.2 | Ridge Regression | 53.77 | Slight improvement |

---

## Run Locally

### 1. Create and activate virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train model manually
```bash
python train.py          # for v0.1
python train_v2.py       # for v0.2 (Ridge)
```

This will create:

```
model/model.joblib
model/metrics.json
```

### 3. Start FastAPI app
```bash
uvicorn app.app:app --host 0.0.0.0 --port 8080
```

---

## API Endpoints

### GET /health
Returns model status and version.
```json
{
  "status": "ok",
  "model_version": "v0.2"
}
```

### POST /predict
Takes a JSON of standardized diabetes features and returns a numeric prediction.
```json
{
  "age": 0.02,
  "sex": -0.044,
  "bmi": 0.06,
  "bp": -0.03,
  "s1": -0.02,
  "s2": 0.03,
  "s3": -0.02,
  "s4": 0.02,
  "s5": 0.02,
  "s6": -0.001
}
```

Response example:
```json
{
  "prediction": 235.94
}
```

---

## GitHub Actions CI/CD
Two main workflows:
- **Push / PR workflow** — Runs lint, tests, quick training, and uploads artifacts.  
- **Release workflow (v* tags)** — Runs full training, builds Docker image, pushes to GHCR, and attaches metrics to the release.

---

## Docker

### Build image
```bash
docker build -t virtual-diabetes:v0.2 .
```

### Run container
```bash
docker run -p 8080:8080 virtual-diabetes:v0.2
```

### Test prediction
```bash
curl -X POST http://127.0.0.1:8080/predict \
-H "Content-Type: application/json" \
-d '{"age":0.02,"sex":-0.044,"bmi":0.06,"bp":-0.03,"s1":-0.02,"s2":0.03,"s3":-0.02,"s4":0.02,"s5":0.02,"s6":-0.001}'
```

---

## Project Structure
```
├── app/
│   └── app.py
├── model/
│   ├── model.joblib
│   └── metrics.json
├── train.py
├── train_v2.py
├── requirements.txt
├── Dockerfile
├── CHANGELOG.md
└── README.md
```

---

## Authors
- Alyssa, Cyrielle, Adeline, Kamelia 
- MAIO MLOps Assignment — Virtual Diabetes Clinic Case Study
