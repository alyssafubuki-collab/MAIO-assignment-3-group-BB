from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, precision_score, recall_score
import numpy as np, joblib, os, json

# === Load data ===
Xy = load_diabetes(as_frame=True)
X = Xy.frame.drop(columns=["target"])
y = Xy.frame["target"]

# === Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Scale ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Train model ===
model = Ridge(alpha=1.0, random_state=42)
model.fit(X_train_scaled, y_train)

# === Predict ===
y_pred = model.predict(X_test_scaled)
rmse = mean_squared_error(y_test, y_pred, squared=False)

# === High-risk flag ===
threshold = np.percentile(y_train, 75)
y_true_flag = (y_test > threshold).astype(int)
y_pred_flag = (y_pred > threshold).astype(int)
precision = precision_score(y_true_flag, y_pred_flag)
recall = recall_score(y_true_flag, y_pred_flag)

# === Save ===
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.joblib")

metrics = {
    "rmse": rmse,
    "precision": precision,
    "recall": recall,
    "threshold": threshold,
    "model_version": "v0.2",
}
with open("model/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"✅ Ridge Regression (v0.2) trained — RMSE: {rmse:.3f}")
print(f"📈 Precision: {precision:.3f}, Recall: {recall:.3f}")
