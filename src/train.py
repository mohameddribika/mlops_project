"""Train baseline churn classifiers and log to MLflow.

Three model families are trained with default-ish parameters:
    - LogisticRegression
    - RandomForestClassifier
    - GradientBoostingClassifier

Each run logs params, metrics on train/val/test, and the fitted pipeline
as an MLflow model artifact (with a signature inferred from the training data).
"""
from __future__ import annotations

import argparse
import json
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve

from src.config import ARTIFACTS_DIR, RANDOM_STATE
from src.data_loader import prepare
from src.evaluation import compute_metrics
from src.mlflow_utils import init_mlflow
from src.pipeline import build_pipeline


def _model_factory(name: str, params: dict[str, Any] | None = None):
    params = params or {}
    if name == "logreg":
        return LogisticRegression(
            max_iter=params.get("max_iter", 1000),
            C=params.get("C", 1.0),
            penalty=params.get("penalty", "l2"),
            solver=params.get("solver", "lbfgs"),
            random_state=RANDOM_STATE,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth"),
            min_samples_split=params.get("min_samples_split", 2),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if name == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=params.get("n_estimators", 200),
            learning_rate=params.get("learning_rate", 0.1),
            max_depth=params.get("max_depth", 3),
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unknown model: {name}")


def _log_diagnostic_plots(y_test, y_pred, y_proba, prefix: str) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    cm_path = ARTIFACTS_DIR / f"{prefix}_confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, colorbar=False)
    fig.tight_layout()
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)
    mlflow.log_artifact(str(cm_path), artifact_path="plots")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_path = ARTIFACTS_DIR / f"{prefix}_roc_curve.png"
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label="model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(roc_path, dpi=120)
    plt.close(fig)
    mlflow.log_artifact(str(roc_path), artifact_path="plots")


def train_one(
    model_name: str,
    params: dict[str, Any] | None = None,
    run_name: str | None = None,
    register_as: str | None = None,
) -> dict[str, float]:
    """Train one model family, log everything to MLflow, return val/test metrics."""
    X_train, X_val, X_test, y_train, y_val, y_test = prepare(persist=False)
    estimator = _model_factory(model_name, params)
    pipe = build_pipeline(estimator)

    with mlflow.start_run(run_name=run_name or f"baseline_{model_name}") as run:
        mlflow.set_tag("model_family", model_name)
        mlflow.set_tag("stage", "baseline")
        mlflow.log_params(
            {f"model__{k}": v for k, v in (params or {}).items()}
        )
        mlflow.log_param("model_name", model_name)

        pipe.fit(X_train, y_train)

        # Predict + score on all three splits — log metrics with prefix.
        for split_name, X_split, y_split in [
            ("train", X_train, y_train),
            ("val", X_val, y_val),
            ("test", X_test, y_test),
        ]:
            preds = pipe.predict(X_split)
            proba = pipe.predict_proba(X_split)[:, 1]
            m = compute_metrics(np.asarray(y_split), preds, proba)
            mlflow.log_metrics(m.as_dict(prefix=f"{split_name}_"))

        # Plots from the final test split.
        test_pred = pipe.predict(X_test)
        test_proba = pipe.predict_proba(X_test)[:, 1]
        _log_diagnostic_plots(y_test, test_pred, test_proba, prefix=model_name)

        # Log model with signature + a small input example.
        signature = infer_signature(X_train.head(5), pipe.predict(X_train.head(5)))
        mlflow.sklearn.log_model(
            sk_model=pipe,
            name="model",
            signature=signature,
            input_example=X_train.head(3),
            registered_model_name=register_as,
        )

        test_metrics = compute_metrics(
            np.asarray(y_test), test_pred, test_proba
        ).as_dict(prefix="test_")
        print(
            f"[{model_name}] run_id={run.info.run_id}  "
            f"test_roc_auc={test_metrics['test_roc_auc']:.4f}"
        )
        return {"run_id": run.info.run_id, **test_metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logreg", "random_forest", "gradient_boosting"],
        help="Which baseline models to train.",
    )
    args = parser.parse_args()

    init_mlflow()
    results = []
    for name in args.models:
        results.append(train_one(name))
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
