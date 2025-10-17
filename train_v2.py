from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import json

# === Load and split dataset ===
Xy = load_diabetes(as_frame=True)
X = Xy.frame.drop(columns=["target"])
y = Xy.frame["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Scale features ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Ridge Regression (improved) ===
model = Ridge(alpha=1.0, random_state=42)
model.fit(X_train_scaled, y_train)

# === Predict and evaluate ===
y_pred = model.predict(X_test_scaled)
rmse = ((y_test - y_pred) ** 2).mean() ** 0.5

# === Save model & metrics ===
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.joblib")

metrics = {
    "rmse": rmse,
    "model_version": "v0.2"
}
with open("model/metrics.json", "w") as f:
    json.dump(metrics, f)

print(f"✅ Ridge Regression (v0.2) trained — RMSE: {rmse:.3f}")
