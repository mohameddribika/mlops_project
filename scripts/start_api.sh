#!/usr/bin/env bash
# Launch the FastAPI prediction service.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
exec ./.venv/bin/uvicorn src.serve:app --host 127.0.0.1 --port 8000 --reload
