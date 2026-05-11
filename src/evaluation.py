"""Evaluation metric helpers — kept separate so train/tune/monitor can share them."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    log_loss: float

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        return {f"{prefix}{k}": v for k, v in asdict(self).items()}


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> ClassificationMetrics:
    return ClassificationMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_true, y_proba),
        pr_auc=average_precision_score(y_true, y_proba),
        log_loss=log_loss(y_true, y_proba, labels=[0, 1]),
    )
