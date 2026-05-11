#!/usr/bin/env bash
# Serve the production model using MLflow's built-in scoring server.
#
# This is the path the term project's solution guide refers to as
# "MLflow's model serving capabilities". The FastAPI service in src/serve.py
# is an alternative that exposes the same model behind a typed REST API.
#
# Endpoints exposed here:
#   GET  /ping          — health check
#   POST /invocations   — accepts the standard MLflow inference payload, e.g.
#                         {"dataframe_split": {"columns": [...], "data": [...]}}
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Activate the venv so the subprocesses spawned by `mlflow models serve`
# (gunicorn / uvicorn) are on PATH.
# shellcheck disable=SC1091
source .venv/bin/activate

export MLFLOW_TRACKING_URI="sqlite:///${PROJECT_ROOT}/mlflow.db"

exec mlflow models serve \
    --model-uri "models:/telco-churn-classifier@production" \
    --host 127.0.0.1 \
    --port 5001 \
    --no-conda
