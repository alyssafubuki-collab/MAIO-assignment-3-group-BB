# CHANGELOG

## v0.1 — Baseline (Linear Regression)
- **Model:** `LinearRegression`
- **Pipeline:** `StandardScaler + LinearRegression`
- **RMSE:** **53.853**
- **High-risk threshold:** 75th percentile of training targets
- **Precision: 1.000**
- **Recall: 0.300**
- **Notes:**
  - Description: Basic regression baseline using StandardScaler + LinearRegression.
  - Simple baseline model.
  - Serves as reference for future iterations.
  - FastAPI `/predict` endpoint functional.
  - The model showed good general performance but limited sensitivity to high-risk cases.
    

---

## v0.2 — Ridge Regression (improvement)
- **Model:** `Ridge(alpha=1.0)`
- **Pipeline:** `StandardScaler + Ridge`
- **RMSE:** **53.777**
- **High-risk threshold:** 75th percentile of training targets
- **Precision: 1.000**
- **Recall: 0.300**
- **Improvements:**
  - Slight RMSE improvement.
  - More stable predictions due to L2 regularization.
  - Same API structure (`/predict` and `/health`).
  - However, classification metrics (precision/recall) remained unchanged, indicating that both models behave similarly around the chosen threshold.
- **Artifacts:**
  - `model/model.joblib`
  - `model/metrics.json`
- **Model version:** `v0.2`

---

Notes :

The precision of 1.0 suggests that all predicted high-risk cases were correct but very few were identified, leading to low recall (0.3).
This behavior is consistent with a high threshold or conservative predictions.
Future iterations could explore:
- Adjusting the high-risk threshold (e.g., 65th percentile).
- Testing non-linear models (e.g., Random Forests).
- Using ROC-based threshold optimization to balance precision and recall automatically.


