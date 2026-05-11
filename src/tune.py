"""Hyperopt-based hyperparameter tuning with nested MLflow runs.

Each trial becomes a child run under a parent "search" run; the parent run
stores the best params + best validation metric so the search can be reviewed
at a glance in the MLflow UI.
"""
from __future__ import annotations

import argparse
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src.config import RANDOM_STATE
from src.data_loader import prepare
from src.evaluation import compute_metrics
from src.mlflow_utils import init_mlflow
from src.pipeline import build_pipeline


SEARCH_SPACES = {
    "gradient_boosting": {
        "n_estimators": scope.int(hp.quniform("n_estimators", 50, 400, 25)),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
        "max_depth": scope.int(hp.quniform("max_depth", 2, 6, 1)),
        "min_samples_split": scope.int(hp.quniform("min_samples_split", 2, 20, 1)),
        "min_samples_leaf": scope.int(hp.quniform("min_samples_leaf", 1, 20, 1)),
        "subsample": hp.uniform("subsample", 0.6, 1.0),
    },
    "logreg": {
        "C": hp.loguniform("C", np.log(0.001), np.log(10.0)),
        "penalty": hp.choice("penalty", ["l1", "l2"]),
        # solver chosen below based on penalty
    },
}


def _build_estimator(model_name: str, params: dict[str, Any]):
    if model_name == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            min_samples_split=int(params["min_samples_split"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            subsample=float(params["subsample"]),
            random_state=RANDOM_STATE,
        )
    if model_name == "logreg":
        penalty = params["penalty"]
        if not isinstance(penalty, str):
            penalty = ["l1", "l2"][penalty]
        solver = "liblinear" if penalty == "l1" else "lbfgs"
        return LogisticRegression(
            C=float(params["C"]),
            penalty=penalty,
            solver=solver,
            max_iter=2000,
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unknown model: {model_name}")


def run_search(model_name: str, max_evals: int) -> dict[str, Any]:
    X_train, X_val, X_test, y_train, y_val, y_test = prepare(persist=False)
    space = SEARCH_SPACES[model_name]

    with mlflow.start_run(run_name=f"hyperopt_{model_name}") as parent:
        mlflow.set_tag("stage", "tuning")
        mlflow.set_tag("model_family", model_name)
        mlflow.log_param("search_algo", "tpe")
        mlflow.log_param("max_evals", max_evals)

        def objective(params: dict[str, Any]) -> dict[str, Any]:
            estimator = _build_estimator(model_name, params)
            pipe = build_pipeline(estimator)
            with mlflow.start_run(nested=True, run_name=f"trial_{model_name}"):
                mlflow.log_params({f"model__{k}": v for k, v in params.items()})
                pipe.fit(X_train, y_train)
                val_proba = pipe.predict_proba(X_val)[:, 1]
                val_pred = pipe.predict(X_val)
                m = compute_metrics(np.asarray(y_val), val_pred, val_proba)
                mlflow.log_metrics(m.as_dict(prefix="val_"))
                # Hyperopt minimizes — flip ROC-AUC.
                return {"loss": -m.roc_auc, "status": STATUS_OK, "metrics": m}

        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            rstate=np.random.default_rng(RANDOM_STATE),
            show_progressbar=False,
        )

        # Hyperopt returns indices for hp.choice — resolve back to the actual params used.
        best_trial = min(trials.results, key=lambda r: r["loss"])
        best_val_metrics = best_trial["metrics"]

        best_params_resolved = {k: best[k] for k in best}
        if model_name == "logreg" and isinstance(best_params_resolved.get("penalty"), int):
            best_params_resolved["penalty"] = ["l1", "l2"][best_params_resolved["penalty"]]

        mlflow.log_params({f"best__{k}": v for k, v in best_params_resolved.items()})
        mlflow.log_metrics(best_val_metrics.as_dict(prefix="best_val_"))

        # Refit best on train, evaluate on test, log the final model under the parent run.
        final_estimator = _build_estimator(model_name, best_params_resolved)
        final_pipe = build_pipeline(final_estimator)
        final_pipe.fit(X_train, y_train)
        test_pred = final_pipe.predict(X_test)
        test_proba = final_pipe.predict_proba(X_test)[:, 1]
        test_m = compute_metrics(np.asarray(y_test), test_pred, test_proba)
        mlflow.log_metrics(test_m.as_dict(prefix="best_test_"))

        signature = infer_signature(X_train.head(5), final_pipe.predict(X_train.head(5)))
        mlflow.sklearn.log_model(
            sk_model=final_pipe,
            name="model",
            signature=signature,
            input_example=X_train.head(3),
        )
        mlflow.set_tag("best_test_roc_auc", f"{test_m.roc_auc:.4f}")

        print(
            f"[{model_name}] best val_roc_auc={best_val_metrics.roc_auc:.4f} "
            f"test_roc_auc={test_m.roc_auc:.4f}"
        )
        print(f"best params: {best_params_resolved}")
        return {
            "parent_run_id": parent.info.run_id,
            "best_params": best_params_resolved,
            "best_val_roc_auc": best_val_metrics.roc_auc,
            "best_test_roc_auc": test_m.roc_auc,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gradient_boosting", choices=list(SEARCH_SPACES))
    parser.add_argument("--max-evals", type=int, default=20)
    args = parser.parse_args()

    init_mlflow()
    result = run_search(args.model, args.max_evals)
    print(result)


if __name__ == "__main__":
    main()
