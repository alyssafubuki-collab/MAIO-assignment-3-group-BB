# CHANGELOG

## v0.1 — Baseline (Linear Regression)
- **Model:** `LinearRegression`
- **Pipeline:** `StandardScaler + LinearRegression`
- **RMSE:** **53.85**
- **Notes:**
  - Simple baseline model.
  - Serves as reference for future iterations.
  - FastAPI `/predict` endpoint functional.

---

## v0.2 — Ridge Regression (improvement)
- **Model:** `Ridge(alpha=1.0)`
- **Pipeline:** `StandardScaler + Ridge`
- **RMSE:** **53.81**
- **Improvements:**
  - Slight RMSE improvement.
  - More stable predictions due to L2 regularization.
  - Same API structure (`/predict` and `/health`).
- **Artifacts:**
  - `model/model.joblib`
  - `model/metrics.json`
- **Model version:** `v0.2`

---

## v0.2-rf — Random Forest (experimental)
- **Model:** `RandomForestRegressor(n_estimators=200, random_state=42)`
- **Pipeline:** `StandardScaler + RandomForestRegressor`
- **RMSE:** **54.59**
- **Other metrics:**
  - `high_risk_precision`: **0.89**
  - `high_risk_recall`: **0.38**
  - `high_risk_threshold`: **214**
- **Notes:**
  - Slightly higher RMSE but better high-risk detection.
  - Good if the clinic prioritizes sensitivity.
  - Experimental version — not default in production.
