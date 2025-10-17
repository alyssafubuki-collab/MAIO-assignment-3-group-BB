import json
import joblib
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import os

# Load dataset
Xy = load_diabetes(as_frame=True)
X = Xy.frame.drop(columns=["target"])
y = Xy.frame["target"]

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Ridge model
model = Ridge(alpha=1.0, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Save model and metrics
os.makedirs("model", exist_ok=True)
joblib.dump({"scaler": scaler, "model": model}, "model/model.joblib")

metrics = {"rmse": rmse, "model_version": "v0.2"}
with open("model/metrics.json", "w") as f:
    json.dump(metrics, f)

print(f"✅ Ridge Regression (v0.2) trained — RMSE: {rmse:.3f}")
