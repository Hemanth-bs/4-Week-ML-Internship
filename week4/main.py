"""FastAPI prediction API for the trained model.
Start from this directory with: uvicorn main:app --reload
"""
from pathlib import Path
from typing import List
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ARTIFACT = Path(__file__).resolve().parent / "model.joblib"
if not ARTIFACT.exists():
    raise RuntimeError("model.joblib missing. Run python train_and_api.py first.")
bundle = joblib.load(ARTIFACT)
model = bundle["model"]
feature_names = bundle["feature_names"]
target_names = bundle["target_names"]
app = FastAPI(title="Breast Cancer Classification API", version="1.0.0",
              description="Educational demo using the scikit-learn Wisconsin Diagnostic Breast Cancer dataset.")

class PredictionRequest(BaseModel):
    features: List[float] = Field(..., description="Exactly 30 numeric features in the order returned by /features")

@app.get("/")
def root():
    return {"message": "ML prediction API is running", "docs": "/docs", "features_endpoint": "/features"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "feature_count": len(feature_names)}

@app.get("/features")
def features():
    return {"count": len(feature_names), "features": feature_names, "target_classes": target_names}

@app.post("/predict")
def predict(request: PredictionRequest):
    if len(request.features) != len(feature_names):
        raise HTTPException(status_code=422, detail=f"Expected {len(feature_names)} features; received {len(request.features)}.")
    if not np.isfinite(np.asarray(request.features, dtype=float)).all():
        raise HTTPException(status_code=422, detail="All feature values must be finite numbers.")
    arr = np.asarray(request.features, dtype=float).reshape(1, -1)
    idx = int(model.predict(arr)[0])
    probabilities = model.predict_proba(arr)[0].tolist()
    return {"prediction_index": idx, "prediction": target_names[idx],
            "probabilities_by_class_index": probabilities,
            "note": "Educational model output, not a medical diagnosis."}
