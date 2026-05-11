"""Production-data drift monitoring.

Reference data is the training set; "incoming" data is simulated by sampling
the test set and applying perturbations (feature shifts, label flips). Each
simulated batch is logged as its own MLflow run inside the ``telco_churn_monitoring``
experiment so drift can be reviewed as a time series in the UI.

Per-batch metrics logged:
    - PSI for each numeric feature (and overall drift_score = max PSI)
    - KS statistic + p-value for each numeric feature
    - model performance (accuracy / ROC-AUC / F1) when labels are available
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy import stats

from src.config import (
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
    REPORTS_DIR,
)
from src.data_loader import NUMERIC_COLS, prepare
from src.evaluation import compute_metrics
from src.mlflow_utils import init_mlflow


MONITORING_EXPERIMENT = "telco_churn_monitoring"
PSI_BUCKETS = 10
# Common thresholds: <0.1 = no drift, 0.1-0.25 = moderate, >0.25 = significant.
PSI_ALERT_THRESHOLD = 0.25


def population_stability_index(reference: np.ndarray, current: np.ndarray) -> float:
    """PSI between two 1-D arrays using equal-frequency bins from reference."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    quantiles = np.linspace(0, 1, PSI_BUCKETS + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    # Smooth tiny bins to avoid log(0).
    eps = 1e-6
    ref_pct = ref_counts / max(ref_counts.sum(), 1) + eps
    cur_pct = cur_counts / max(cur_counts.sum(), 1) + eps
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def simulate_batch(test: pd.DataFrame, labels: pd.Series, mode: str, rng: np.random.Generator):
    """Return a (features, labels) tuple that mimics an incoming production batch."""
    n = len(test)
    idx = rng.choice(n, size=min(500, n), replace=False)
    batch = test.iloc[idx].copy()
    y = labels.iloc[idx].copy()

    if mode == "clean":
        return batch, y, "no perturbation — should show low PSI"

    if mode == "feature_drift":
        # Simulate a billing-system change: monthly charges shifted up ~20%,
        # contracts skewed toward Month-to-month, fiber adoption increased.
        batch["MonthlyCharges"] = batch["MonthlyCharges"] * rng.uniform(1.15, 1.35, size=len(batch))
        batch["TotalCharges"] = batch["TotalCharges"] * rng.uniform(1.05, 1.20, size=len(batch))
        contract_mask = rng.random(len(batch)) < 0.4
        batch.loc[contract_mask, "Contract"] = "Month-to-month"
        return batch, y, "monthly charges +20-35%, contracts skewed toward Month-to-month"

    if mode == "concept_drift":
        # Same feature distribution; labels flipped for ~25% of records — the
        # relationship between features and the target has changed.
        flip_mask = rng.random(len(batch)) < 0.25
        y = y.copy()
        y[flip_mask] = 1 - y[flip_mask]
        return batch, y, "25% of labels flipped — relationship between X and Y has shifted"

    raise ValueError(f"Unknown mode: {mode}")


def _load_production_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}@production")


def _try_evidently_report(reference: pd.DataFrame, current: pd.DataFrame, path: Path) -> bool:
    """Best-effort: render an Evidently HTML report. Returns False if the library
    can't produce one in this environment (we still log the manual metrics)."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        report = Report([DataDriftPreset()])
        snapshot = report.run(reference_data=reference, current_data=current)
        snapshot.save_html(str(path))
        return True
    except Exception as exc:  # noqa: BLE001 — Evidently API drift between minor versions
        warnings.warn(f"Evidently report failed ({exc}); skipping HTML artifact.")
        return False


def run_monitoring(seed: int = 7) -> list[dict]:
    init_mlflow(MONITORING_EXPERIMENT)
    rng = np.random.default_rng(seed)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, _, X_test, y_train, _, y_test = prepare(persist=False)
    model = _load_production_model()
    summaries = []

    for mode in ("clean", "feature_drift", "concept_drift"):
        current_X, current_y, description = simulate_batch(X_test, y_test, mode, rng)

        with mlflow.start_run(run_name=f"monitoring_{mode}") as run:
            mlflow.set_tag("stage", "monitoring")
            mlflow.set_tag("scenario", mode)
            mlflow.set_tag("description", description)
            mlflow.log_param("batch_size", len(current_X))
            mlflow.log_param("registered_model", f"{REGISTERED_MODEL_NAME}@production")

            psi_scores = {}
            ks_pvalues = {}
            for col in NUMERIC_COLS:
                ref = X_train[col].astype(float).to_numpy()
                cur = current_X[col].astype(float).to_numpy()
                psi = population_stability_index(ref, cur)
                ks_stat, ks_p = stats.ks_2samp(ref, cur)
                psi_scores[col] = psi
                ks_pvalues[col] = ks_p
                mlflow.log_metric(f"psi_{col}", psi)
                mlflow.log_metric(f"ks_pvalue_{col}", ks_p)
                mlflow.log_metric(f"ks_stat_{col}", ks_stat)

            max_psi = max(psi_scores.values())
            drift_alert = int(max_psi > PSI_ALERT_THRESHOLD)
            mlflow.log_metric("max_psi", max_psi)
            mlflow.log_metric("drift_alert", drift_alert)

            preds = model.predict(current_X)
            proba = model.predict_proba(current_X)[:, 1]
            perf = compute_metrics(np.asarray(current_y), preds, proba)
            mlflow.log_metrics(perf.as_dict(prefix="prod_"))

            html_path = REPORTS_DIR / f"drift_report_{mode}.html"
            if _try_evidently_report(X_train, current_X, html_path):
                mlflow.log_artifact(str(html_path), artifact_path="drift_reports")

            print(
                f"[{mode}] max_psi={max_psi:.3f} alert={drift_alert} "
                f"prod_roc_auc={perf.roc_auc:.4f} prod_f1={perf.f1:.4f}"
            )
            summaries.append(
                {
                    "mode": mode,
                    "run_id": run.info.run_id,
                    "max_psi": max_psi,
                    "drift_alert": bool(drift_alert),
                    "prod_roc_auc": perf.roc_auc,
                    "prod_f1": perf.f1,
                }
            )

    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    out = run_monitoring(seed=args.seed)
    import json

    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
