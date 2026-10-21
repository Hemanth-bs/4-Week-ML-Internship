"""Week 3: clustering, PCA visualization coordinates, cross-validation and tuning."""
import json
from pathlib import Path
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

OUT = Path(__file__).resolve().parent
iris = load_iris()
X = StandardScaler().fit_transform(iris.data)
cluster_results = []
for name, model in [
    ("K-Means (k=3)", KMeans(n_clusters=3, random_state=42, n_init=20)),
    ("Agglomerative (k=3)", AgglomerativeClustering(n_clusters=3, linkage="ward"))
]:
    labels = model.fit_predict(X)
    cluster_results.append({
        "model": name,
        "silhouette_score": float(silhouette_score(X, labels)),
        "adjusted_rand_index_vs_known_species_for_analysis_only": float(adjusted_rand_score(iris.target, labels)),
        "cluster_sizes": {str(int(k)): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
    })
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X)
pca_info = {"explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
            "total_variance_explained_2_components": float(pca.explained_variance_ratio_.sum())}
# Classification evaluation and small hyperparameter search on breast-cancer data.
c = load_breast_cancer()
Xtr, Xte, ytr, yte = train_test_split(c.data, c.target, test_size=0.2, stratify=c.target, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
base = RandomForestClassifier(random_state=42, n_jobs=-1)
grid = GridSearchCV(base, {"n_estimators": [100, 200], "max_depth": [None, 8]},
                    cv=cv, scoring="f1", n_jobs=-1)
grid.fit(Xtr, ytr)
best = grid.best_estimator_
pred = best.predict(Xte)
prob = best.predict_proba(Xte)[:, 1]
result = {
    "clustering": cluster_results, "pca": pca_info,
    "classification": {
        "best_params": grid.best_params_,
        "best_mean_cv_f1": float(grid.best_score_),
        "test_accuracy": float(accuracy_score(yte, pred)),
        "test_precision": float(precision_score(yte, pred)),
        "test_recall": float(recall_score(yte, pred)),
        "test_f1": float(f1_score(yte, pred)),
        "test_roc_auc": float(roc_auc_score(yte, prob)),
        "confusion_matrix_rows_true_cols_pred": confusion_matrix(yte, pred).tolist(),
        "classification_report": classification_report(yte, pred, target_names=c.target_names, output_dict=True)
    },
    "protocol": "Iris clustering scaled; breast-cancer classifier uses stratified 80/20 split and 5-fold CV on training data."
}
(OUT / "results.json").write_text(json.dumps(result, indent=2))
# Save PCA coordinates and cluster labels for reproducible plotting.
import pandas as pd
pd.DataFrame({"PC1": coords[:,0], "PC2": coords[:,1], "species": [iris.target_names[i] for i in iris.target]}).to_csv(OUT / "iris_pca_coordinates.csv", index=False)
print(json.dumps(result, indent=2))
