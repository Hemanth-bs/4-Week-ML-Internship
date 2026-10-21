# Four-Week Machine Learning Internship Project

This repository contains reproducible implementations and reports for:
- Week 1: Python, NumPy, Pandas, EDA and preprocessing
- Week 2: Regression and classification model comparison
- Week 3: Clustering, PCA, cross-validation and hyperparameter tuning
- Week 4: Serialized ML model and FastAPI prediction service

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run scripts from the repository root:
```bash
python week1/preprocess.py
python week2/train_models.py
python week3/evaluate_clustering.py
python week4/train_and_api.py
cd week4
uvicorn main:app --reload
```
Open `http://127.0.0.1:8000/docs` for interactive API documentation.

## Dataset and limitations
The examples use built-in datasets from scikit-learn so the project does not depend on external downloads. Week 1 deliberately injects missing values to demonstrate imputation; this is documented and is not a claim about missingness in the original dataset. Results are educational benchmark results, not clinical guidance. For production deployment, add authentication, rate limiting, logging, monitoring, dependency pinning and a formal data/privacy review.
