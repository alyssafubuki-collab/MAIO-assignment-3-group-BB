from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import json
import numpy as np

# 1️⃣ Charger les données
Xy = load_diabetes(as_frame=True)
X = Xy.frame.drop(columns=["target"])
y = Xy.frame["target"]

# 2️⃣ Séparer train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3️⃣ Mise à l’échelle
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4️⃣ Nouveau modèle : Ridge Regression
model = Ridge(alpha=0.5)
model.fit(X_train_scaled, y_train)

# 5️⃣ Créer dossier model si inexistant
os.makedirs("model", exist_ok=True)

# 6️⃣ Sauvegarder modèle
joblib.dump(model, "model/model.joblib")

# 7️⃣ Calculer métriques
y_pred = model.predict(X_test_scaled)
rmse = np.sqrt(((y_test - y_pred) ** 2).mean())

metrics = {
    "rmse": float(rmse),
    "model_version": "v0.2"
}

# 8️⃣ Sauvegarder métriques
with open("model/metrics.json", "w") as f:
    json.dump(metrics, f)

print("✅ Nouveau modèle Ridge (v0.2) entraîné et sauvegardé !")
print(f"RMSE: {metrics['rmse']}")
