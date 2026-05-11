# Telco Churn — End-to-End MLflow Lifecycle Management

AIN-3009 MLOps term project. Predicts customer churn for a telecommunications
provider using the Telco Customer Churn dataset, with every stage of the ML
lifecycle managed by **MLflow**:

| Stage | Module | What it does |
| --- | --- | --- |
| Experiment tracking | `src.train` | Trains 3 baseline classifiers, logs params/metrics/artifacts |
| Hyperparameter tuning | `src.tune` | Hyperopt TPE search with nested MLflow runs |
| Model Registry | `src.registry` | Registers the best run, transitions through `staging` → `production` aliases |
| Model serving | `mlflow models serve` | Loads the `@production` model and exposes `/invocations` |
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

# 5. serve the production model (MLflow's built-in scoring server, port 5001)
./scripts/start_mlflow_serve.sh
# POST /invocations expects:
#   {"dataframe_split": {"columns": [...], "data": [...]}}

# 6. drift monitoring (3 simulated batches: clean / feature_drift / concept_drift)
.venv/bin/python -m src.monitor

# 7. browse runs in the MLflow UI
./scripts/start_mlflow_ui.sh                 # http://127.0.0.1:5000
```

## Project layout

```
mlops_project/
├── data/raw/telco_churn.csv         # source dataset
├── src/
│   ├── config.py                    # paths + MLflow URIs + constants
│   ├── data_loader.py               # load, clean, split
│   ├── pipeline.py                  # sklearn preprocessor + estimator
│   ├── evaluation.py                # shared metric helpers
│   ├── mlflow_utils.py              # set tracking URI + experiment
│   ├── train.py                     # baseline runs
│   ├── tune.py                      # Hyperopt + nested runs
│   ├── registry.py                  # register / transition / list
│   └── monitor.py                   # drift + perf monitoring runs
├── scripts/
│   ├── start_mlflow_ui.sh           # SQLite-backed tracking server, port 5000
│   └── start_mlflow_serve.sh        # MLflow built-in scoring server, port 5001
├── reports/
│   ├── project_report.pdf           # written deliverable (PDF)
│   ├── project_report.docx          # same content as DOCX
│   ├── presentation.pptx            # presentation deliverable
│   └── drift_report_*.html          # Evidently reports (one per scenario)
├── artifacts/                       # confusion matrices, ROC curves
└── requirements.txt
```

## Tracking store

- **Backend store**: SQLite (`sqlite:///mlflow.db`) — keeps everything in one file.
- **Artifact root**: `./mlruns/` (local filesystem).
- **Experiments**: `telco_churn` (training + tuning) and `telco_churn_monitoring`
  (drift batches).
