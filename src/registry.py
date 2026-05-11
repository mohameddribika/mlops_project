"""Model Registry: find the best run, register it, manage stage transitions.

MLflow 3 has moved away from numeric stages toward named **aliases**
(e.g. ``@staging``, ``@production``). This module uses the alias API but
also writes the legacy "Staging"/"Production" stage tags so the demo still
shows the same lifecycle stages the brief asks for.
"""
from __future__ import annotations

import argparse
from typing import Iterable

import mlflow
from mlflow import MlflowClient
from mlflow.entities import ViewType

from src.config import EXPERIMENT_NAME, REGISTERED_MODEL_NAME
from src.mlflow_utils import init_mlflow


METRIC_FOR_SELECTION = "metrics.test_roc_auc"


def _best_run_across_experiment(client: MlflowClient, experiment_id: str):
    """Pick the run with the highest test_roc_auc — could be a baseline OR a tuned run.

    Tuned parent runs log ``best_test_roc_auc`` instead of ``test_roc_auc``;
    we union both before sorting so the registry isn't biased toward the
    baseline naming convention.
    """
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1000,
    )
    scored = []
    for run in runs:
        score = run.data.metrics.get("test_roc_auc") or run.data.metrics.get("best_test_roc_auc")
        if score is None:
            continue
        scored.append((score, run))
    if not scored:
        raise RuntimeError("No runs with test_roc_auc / best_test_roc_auc found.")
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[0][1]


def _ensure_registered_model(client: MlflowClient, name: str) -> None:
    try:
        client.get_registered_model(name)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(name)


def register_best(model_name: str = REGISTERED_MODEL_NAME) -> dict:
    init_mlflow()
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"Experiment {EXPERIMENT_NAME!r} not found — train models first.")

    best_run = _best_run_across_experiment(client, exp.experiment_id)
    best_score = (
        best_run.data.metrics.get("test_roc_auc")
        or best_run.data.metrics["best_test_roc_auc"]
    )

    # MLflow 3 stores models as "Logged Models" with model_id="m-<uuid>"; locate the
    # one produced by this run and register against its canonical URI.
    logged = client.search_logged_models(
        experiment_ids=[exp.experiment_id],
        filter_string=f"source_run_id='{best_run.info.run_id}'",
    )
    if not logged:
        raise RuntimeError(f"No logged model found for run {best_run.info.run_id}")
    logged_model = logged[0]
    source_uri = f"models:/{logged_model.model_id}"

    _ensure_registered_model(client, model_name)
    mv = client.create_model_version(
        name=model_name,
        source=source_uri,
        run_id=best_run.info.run_id,
        description=(
            f"Auto-registered from run {best_run.info.run_id} "
            f"(logged_model={logged_model.model_id}) with test_roc_auc={best_score:.4f}"
        ),
    )
    # Tag the version with the source run's family for traceability.
    family = best_run.data.tags.get("model_family", "unknown")
    client.set_model_version_tag(model_name, mv.version, "model_family", family)
    client.set_model_version_tag(model_name, mv.version, "selection_metric", f"{best_score:.6f}")

    print(
        f"Registered {model_name} v{mv.version} from "
        f"run {best_run.info.run_id} "
        f"(family={family}, test_roc_auc={best_score:.4f})"
    )
    return {
        "name": model_name,
        "version": mv.version,
        "run_id": best_run.info.run_id,
        "score": best_score,
    }


def transition(model_name: str, version: str | int, stage: str) -> None:
    """Move a model version into the staging/production lifecycle stage.

    Uses both legacy stage tags (for display) and modern aliases (for serving).
    """
    valid = {"staging", "production", "archived"}
    if stage.lower() not in valid:
        raise ValueError(f"stage must be one of {valid}")
    stage = stage.lower()

    client = MlflowClient()
    # Modern alias-based promotion (recommended in MLflow 3+).
    if stage in {"staging", "production"}:
        client.set_registered_model_alias(model_name, stage, version)

    # Mirror as a tag so the legacy lifecycle wording the brief uses is visible too.
    client.set_model_version_tag(model_name, version, "stage", stage)
    print(f"Set {model_name} v{version} -> alias '{stage}' (tag stage={stage})")


def list_versions(model_name: str = REGISTERED_MODEL_NAME) -> Iterable[dict]:
    client = MlflowClient()
    try:
        rm = client.get_registered_model(model_name)
    except mlflow.exceptions.MlflowException:
        print(f"No registered model named {model_name!r}.")
        return []
    # rm.aliases is {alias_name: version_str}; invert so we can list aliases per version.
    aliases_by_version: dict[str, list[str]] = {}
    for alias, ver in dict(rm.aliases).items():
        aliases_by_version.setdefault(str(ver), []).append(alias)

    rows = []
    for v in client.search_model_versions(f"name='{model_name}'"):
        rows.append(
            {
                "version": v.version,
                "stage_tag": v.tags.get("stage", "-"),
                "aliases": ",".join(sorted(aliases_by_version.get(str(v.version), []))),
                "run_id": v.run_id,
                "score": v.tags.get("selection_metric", "-"),
            }
        )
    rows.sort(key=lambda r: int(r["version"]))
    for r in rows:
        print(r)
    return rows


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("register-best")
    p_trans = sub.add_parser("transition")
    p_trans.add_argument("version")
    p_trans.add_argument("stage", choices=["staging", "production", "archived"])
    sub.add_parser("list")
    args = parser.parse_args()
    init_mlflow()

    if args.cmd == "register-best":
        register_best()
    elif args.cmd == "transition":
        transition(REGISTERED_MODEL_NAME, args.version, args.stage)
    elif args.cmd == "list":
        list_versions()


if __name__ == "__main__":
    main()
