"""Week 1: reproducible preprocessing and EDA using scikit-learn breast-cancer data."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent
data = load_breast_cancer(as_frame=True)
df = data.data.copy()
df["diagnosis"] = pd.Series(data.target, index=df.index).map({0: "malignant", 1: "benign"})
# Reproducible missingness for demonstrating a real preprocessing workflow.
rng = np.random.default_rng(42)
for col in ["mean radius", "mean texture", "mean area"]:
    idx = rng.choice(df.index.to_numpy(), size=12, replace=False)
    df.loc[idx, col] = np.nan
df["radius_group"] = pd.cut(df["mean radius"], bins=[-np.inf, 12, 18, np.inf],
                            labels=["small", "medium", "large"])
missing_before = df.isna().sum().to_dict()
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = ["radius_group"]
df[numeric_cols] = SimpleImputer(strategy="median").fit_transform(df[numeric_cols])
df[categorical_cols] = df[categorical_cols].astype("object").fillna("unknown")
encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
X = encoded.drop(columns=["diagnosis"])
y = (encoded["diagnosis"] == "malignant").astype(int)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
selector = SelectKBest(score_func=f_classif, k=10)
selector.fit(X_scaled, y)
selected = X.columns[selector.get_support()].tolist()
summary = {
    "dataset": "scikit-learn Wisconsin Diagnostic Breast Cancer dataset",
    "rows": int(df.shape[0]),
    "original_numeric_features": int(data.data.shape[1]),
    "features_after_encoding": int(X.shape[1]),
    "missing_values_injected": int(sum(missing_before.values())),
    "missing_values_after_cleaning": int(df.isna().sum().sum()),
    "target_counts": {str(k): int(v) for k, v in df["diagnosis"].value_counts().items()},
    "selected_top_10_features": selected,
    "scaled_feature_mean_abs_max": float(X_scaled.mean().abs().max()),
    "scaled_feature_std_mean": float(X_scaled.std(ddof=0).mean()),
    "sample_rows": df[["mean radius", "mean texture", "mean area", "diagnosis", "radius_group"]].head(5).fillna("").to_dict(orient="records")
}
df.to_csv(OUT / "cleaned_breast_cancer.csv", index=False)
(OUT / "results.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
