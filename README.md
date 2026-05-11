# Telco Churn — End-to-End MLflow Lifecycle Management

AIN-3009 MLOps term project. Predicts customer churn for a telecommunications
provider using the Telco Customer Churn dataset, with every stage of the ML
lifecycle managed by **MLflow**:

| Stage | Module | What it does |
| --- | --- | --- |
| Experiment tracking | `src.train` | Trains 3 baseline classifiers, logs params/metrics/artifacts |
| Hyperparameter tuning | `src.tune` | Hyperopt TPE search with nested MLflow runs |
| Model Registry | `src.registry` | Registers the best run, transitions through `staging` → `production` aliases |
| Real-time serving | `src.serve` | FastAPI app loading the `@production` model |
| Drift monitoring | `src.monitor` | Simulates production batches, logs PSI + KS + performance to MLflow |

## Quick start

```bash
# 1. install
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. train baselines
.venv/bin/python -m src.train

# 3. Hyperopt search (20 trials, nested runs)
.venv/bin/python -m src.tune --model gradient_boosting --max-evals 20

# 4. register best run, transition through stages
.venv/bin/python -m src.registry register-best
.venv/bin/python -m src.registry transition 1 staging
.venv/bin/python -m src.registry transition 1 production
.venv/bin/python -m src.registry list

# 5. serve — pick one
./scripts/start_mlflow_serve.sh              # MLflow built-in scoring server (port 5001)
                                             #   POST /invocations expects:
                                             #   {"dataframe_split": {"columns": [...], "data": [...]}}
./scripts/start_api.sh                       # FastAPI alternative (port 8000)
                                             #   POST /predict expects a typed JSON record

# 6. drift monitoring (3 simulated batches: clean / feature_drift / concept_drift)
.venv/bin/python -m src.monitor

# 7. browse runs in the MLflow UI
./scripts/start_mlflow_ui.sh                 # http://127.0.0.1:5000
```

## Project layout

```
mlops_project/
├── data/
│   ├── raw/telco_churn.csv          # downloaded once
│   └── processed/                   # parquet splits (gitignored)
├── src/
│   ├── config.py                    # paths + MLflow URIs + constants
│   ├── data_loader.py               # load, clean, split
│   ├── pipeline.py                  # sklearn preprocessor + estimator
│   ├── evaluation.py                # shared metric helpers
│   ├── mlflow_utils.py              # set tracking URI + experiment
│   ├── train.py                     # baseline runs
│   ├── tune.py                      # Hyperopt + nested runs
│   ├── registry.py                  # register / transition / list
│   ├── serve.py                     # FastAPI inference service
│   └── monitor.py                   # drift + perf monitoring runs
├── scripts/
│   ├── start_mlflow_ui.sh           # SQLite-backed tracking server, port 5000
│   ├── start_mlflow_serve.sh        # MLflow built-in scoring server, port 5001
│   └── start_api.sh                 # uvicorn FastAPI server, port 8000
├── reports/
│   ├── project_report.md            # written deliverable
│   ├── presentation_outline.md      # 5-minute presentation slide-by-slide
│   └── drift_report_*.html          # Evidently reports (one per scenario)
├── artifacts/                       # confusion matrices, ROC curves
├── mlflow.db                        # SQLite tracking store
├── mlruns/                          # run + logged-model artifacts
└── requirements.txt
```

## Tracking store

- **Backend store**: SQLite (`sqlite:///mlflow.db`) — keeps everything in one file.
- **Artifact root**: `./mlruns/` (local filesystem). For a real deployment this
  would point at S3 / GCS / Azure Blob.
- **Experiments**: `telco_churn` (training + tuning) and `telco_churn_monitoring`
  (drift batches).
