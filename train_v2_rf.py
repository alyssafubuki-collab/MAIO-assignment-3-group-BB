# train_v2_rf.py
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import json
import numpy as np
from sklearn.metrics import precision_score, recall_score

RANDOM_STATE = 42
N_ESTIMATORS = 200

# 1) Load data
Xy = load_diabetes(as_frame=True)
X = Xy.frame.drop(columns=["target"])
y = Xy.frame["target"]

# 2) Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# 3) Scale (optional for RF, kept for parity)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4) Train RandomForest
rf = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train_scaled, y_train)

# 5) Save folder
os.makedirs("model", exist_ok=True)
joblib.dump(rf, "model/model.joblib")
joblib.dump(scaler, "model/scaler.joblib")  # optional: save scaler too

# 6) Predict and RMSE
y_pred = rf.predict(X_test_scaled)
rmse = float(np.sqrt(((y_test - y_pred) ** 2).mean()))

# 7) Define high-risk threshold (75th percentile on train targets)
threshold = float(np.percentile(y_train, 75.0))

# 8) Compute classification metrics (precision/recall) on test set
y_test_high = (y_test >= threshold).astype(int)
y_pred_high = (y_pred >= threshold).astype(int)

precision = None
recall = None
if y_pred_high.sum() == 0:
    precision = 0.0
    recall = 0.0
else:
    precision = float(precision_score(y_test_high, y_pred_high, zero_division=0))
    recall = float(recall_score(y_test_high, y_pred_high, zero_division=0))

metrics = {
    "rmse": rmse,
    "model_version": "v0.2-rf",
    "high_risk_threshold": threshold,
    "high_risk_precision": precision,
    "high_risk_recall": recall,
    "n_estimators": N_ESTIMATORS,
    "random_state": RANDOM_STATE
}

with open("model/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ RandomForest model v0.2-rf trained and saved!")
print(f"RMSE: {rmse}, threshold: {threshold}, precision: {precision}, recall: {recall}")
