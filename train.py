app/app.py


from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import json

Xy = load_diabetes(as_frame=True)
X = Xy.frame.drop(columns=["target"])
y = Xy.frame["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Create model folder if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save model
joblib.dump(model, "model/model.joblib")

# Save metrics
y_pred = model.predict(X_test_scaled)
rmse = ((y_test - y_pred) ** 2).mean() ** 0.5
metrics = {"rmse": rmse, "model_version": "v0.1"}
with open("model/metrics.json", "w") as f:
    json.dump(metrics, f)

print("Model trained and saved!")


