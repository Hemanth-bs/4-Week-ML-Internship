"""Week 2: regression and classification baselines with held-out evaluation."""
import json
from pathlib import Path
import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

OUT = Path(__file__).resolve().parent
results = {"regression": [], "classification": []}
# Regression
d = load_diabetes()
Xtr, Xte, ytr, yte = train_test_split(d.data, d.target, test_size=0.2, random_state=42)
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
}
for name, model in models.items():
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    results["regression"].append({
        "model": name, "MAE": float(mean_absolute_error(yte, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(yte, pred))), "R2": float(r2_score(yte, pred))
    })
# Classification
c = load_breast_cancer()
Xtr, Xte, ytr, yte = train_test_split(c.data, c.target, test_size=0.2, stratify=c.target, random_state=42)
models = {
    "Scaled Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=42)),
    "Random Forest Classifier": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
}
for name, model in models.items():
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    prob = model.predict_proba(Xte)[:, 1]
    results["classification"].append({
        "model": name, "accuracy": float(accuracy_score(yte, pred)),
        "precision": float(precision_score(yte, pred)), "recall": float(recall_score(yte, pred)),
        "f1": float(f1_score(yte, pred)), "roc_auc": float(roc_auc_score(yte, prob))
    })
results["split"] = "80% train / 20% test; random_state=42; classification stratified"
(OUT / "results.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
