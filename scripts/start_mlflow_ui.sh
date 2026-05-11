#!/usr/bin/env bash
# Start the MLflow tracking UI for this project.
# Uses a SQLite backend store and a local filesystem artifact root.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

exec ./.venv/bin/mlflow server \
    --backend-store-uri "sqlite:///${PROJECT_ROOT}/mlflow.db" \
    --default-artifact-root "${PROJECT_ROOT}/mlartifacts" \
    --host 127.0.0.1 \
    --port 5000
