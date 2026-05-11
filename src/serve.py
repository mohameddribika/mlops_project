"""FastAPI service that serves the production model from the MLflow registry.

Two prediction endpoints:
    POST /predict         — single record (real-time)
    POST /predict-batch   — array of records

The model is loaded once on startup via the alias ``models:/<name>@production``,
so promoting a new version in the registry only requires a restart.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME


class CustomerRecord(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class BatchRequest(BaseModel):
    records: list[CustomerRecord] = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    model_version: str
    model_uri: str


_STATE: dict[str, Any] = {"model": None, "model_uri": None, "version": None}


def _load_production_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@production"
    model = mlflow.pyfunc.load_model(model_uri)
    # Resolve the actual version behind the alias for traceability in responses.
    client = mlflow.MlflowClient()
    try:
        mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "production")
        version = mv.version
    except Exception:
        version = "unknown"
    _STATE["model"] = model
    _STATE["model_uri"] = model_uri
    _STATE["version"] = version
    print(f"Loaded {model_uri} (version {version})")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load_production_model()
    yield


app = FastAPI(
    title="Telco Churn Classifier",
    version="1.0",
    description="Serves the production model registered in MLflow.",
    lifespan=lifespan,
)


def _predict(df: pd.DataFrame) -> list[PredictionResponse]:
    model = _STATE["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    # The underlying sklearn pipeline exposes predict_proba, but mlflow.pyfunc
    # only guarantees `predict`. We call the unwrapped sklearn model for proba.
    try:
        sk_model = model.unwrap_python_model() if hasattr(model, "unwrap_python_model") else None
    except Exception:
        sk_model = None
    if sk_model is None:
        # Direct sklearn flavor — load via sklearn to get predict_proba.
        import mlflow.sklearn  # local import to avoid startup cost when not needed

        sk_model = mlflow.sklearn.load_model(_STATE["model_uri"])
        _STATE["model"] = sk_model  # cache it so future calls skip the reload

    proba = sk_model.predict_proba(df)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return [
        PredictionResponse(churn_prediction=int(p), churn_probability=float(prob))
        for p, prob in zip(preds, proba)
    ]


@app.get("/health")
def health():
    return {
        "status": "ok" if _STATE["model"] is not None else "loading",
        "model_uri": _STATE["model_uri"],
        "version": _STATE["version"],
    }


@app.post("/predict", response_model=BatchResponse)
def predict(record: CustomerRecord):
    df = pd.DataFrame([record.model_dump()])
    preds = _predict(df)
    return BatchResponse(
        predictions=preds,
        model_version=str(_STATE["version"]),
        model_uri=str(_STATE["model_uri"]),
    )


@app.post("/predict-batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    df = pd.DataFrame([r.model_dump() for r in req.records])
    preds = _predict(df)
    return BatchResponse(
        predictions=preds,
        model_version=str(_STATE["version"]),
        model_uri=str(_STATE["model_uri"]),
    )
