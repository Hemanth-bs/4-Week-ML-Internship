"""Train and serialize a breast-cancer classifier. Run: python train_and_api.py"""
import json
from pathlib import Path
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

OUT = Path(__file__).resolve().parent
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, stratify=data.target, random_state=42
)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=42))
model.fit(X_train, y_train)
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]
joblib.dump({"model": model, "feature_names": list(data.feature_names), "target_names": list(data.target_names)}, OUT / "model.joblib")
result = {
    "dataset": "Wisconsin Diagnostic Breast Cancer (scikit-learn built-in)",
    "train_rows": int(len(y_train)), "test_rows": int(len(y_test)),
    "accuracy": float(accuracy_score(y_test, pred)),
    "precision": float(precision_score(y_test, pred)),
    "recall": float(recall_score(y_test, pred)),
    "f1": float(f1_score(y_test, pred)),
    "roc_auc": float(roc_auc_score(y_test, prob)),
    "artifact": "model.joblib",
    "sample_input_first_5_features": [float(v) for v in X_test[0, :5]],
    "sample_prediction_class_index": int(model.predict(X_test[0:1])[0]),
    "sample_prediction_class_name": str(data.target_names[int(model.predict(X_test[0:1])[0])]),
    "sample_probability_class_1": float(model.predict_proba(X_test[0:1])[0, 1])
}
(OUT / "results.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
