# 🧾 CHANGELOG

## v0.1 — Baseline
- Date: 2025-10-17
- Model: `LinearRegression`
- Preprocessing: `StandardScaler`
- RMSE (test): **53.77**
- Notes: Baseline simple et reproductible. Sert de référence initiale pour le triage virtuel.

---

## v0.2 — Improved Model (RandomForest)
- Date: 2025-10-17
- Model: `RandomForestRegressor (n_estimators=200, random_state=42)`
- Preprocessing: `StandardScaler`
- RMSE (test): **46.95**
- High-risk threshold: 75th percentile of training target = **~125.3**
- High-risk Precision: **0.61**
- High-risk Recall: **0.73**
- Notes:
  - Amélioration significative du RMSE (~13%).
  - Meilleure capacité à détecter les patients à haut risque.
  - Modèle plus robuste face aux interactions non linéaires entre variables (e.g. BMI, BP, s5).
  - Les métriques sont sauvegardées dans `model/metrics.json`.

---

## v0.3 (future work — optional ideas)
- Envisager `GradientBoostingRegressor` ou `XGBoost` pour encore meilleure précision.
- Ajouter une calibration de probabilité pour le flag "high_risk".
- Intégrer des tests unitaires automatisés pour la fonction `/predict`.
