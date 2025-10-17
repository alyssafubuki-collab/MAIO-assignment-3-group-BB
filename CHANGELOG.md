# CHANGELOG

## v0.1 — Baseline (Linear Regression)
- **Model:** `LinearRegression`
- **Pipeline:** `StandardScaler + LinearRegression`
- **RMSE:** **53.853** 
- **Notes:**
  - Description: Basic regression baseline using StandardScaler + LinearRegression.
  - Simple baseline model.
  - Serves as reference for future iterations.
  - FastAPI `/predict` endpoint functional.
    

---

## v0.2 — Ridge Regression (improvement)
- **Model:** `Ridge(alpha=1.0)`
- **Pipeline:** `StandardScaler + Ridge`
- **RMSE:** **53.777**
- **Improvements:**
  - Slight RMSE improvement.
  - More stable predictions due to L2 regularization.
  - Same API structure (`/predict` and `/health`).
- **Artifacts:**
  - `model/model.joblib`
  - `model/metrics.json`
- **Model version:** `v0.2`


