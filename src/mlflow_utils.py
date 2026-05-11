"""MLflow setup helpers."""
from __future__ import annotations

import mlflow

from src.config import EXPERIMENT_NAME, MLFLOW_TRACKING_URI


def init_mlflow(experiment: str = EXPERIMENT_NAME) -> str:
    """Point MLflow at the SQLite tracking store and ensure the experiment exists.

    Returns the experiment ID.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        exp_id = mlflow.create_experiment(experiment)
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(experiment)
    return exp_id
