# CHANGELOG

## v0.1 — Baseline
- Date: 2025-10-17
- Modèle: LinearRegression
- Préprocessing: StandardScaler
- RMSE (test): 53.77
- Notes: Baseline simple, fast to train. Expose /health and /predict.

## v0.2 — Improvement (Ridge)
- Date: 2025-10-17
- Modèle: Ridge(alpha=0.5)
- Préprocessing: StandardScaler
- RMSE (test): 49.32
- Notes: Added L2 regularization; improved RMSE ~4 points vs v0.1.

## v0.2-rf — Alternative (RandomForest) — experimental
- Date: 2025-10-17
- Modèle: RandomForestRegressor(n_estimators=200, random_state=42)
- Préprocessing: StandardScaler (features scaled before RF)
- RMSE (test): <remplacer_par_ton_score>
- High-risk threshold: 75th percentile of training target (threshold value recorded in metrics.json)
- Precision (high-risk) @threshold: <…>
- Recall (high-risk) @threshold: <…>
- Rationale: RF can capture non-linearities and interactions. Use if it improves RMSE or improves recall for high-risk detection.

