# Presentation Outline — 8 minutes (5-minute cuts marked ✂)

> Domain: **Telco customer churn**. Tool: **MLflow**. End-to-end lifecycle:
> tracking → tuning → registry → serving → monitoring.

---

## Slide 1 — Title (0:00 – 0:20)

- Development and Evaluation of an MLflow-based ML Lifecycle Management System
- Telco customer churn prediction
- Name / student number / course

## Slide 2 — Problem and dataset (0:20 – 1:00)

- Why churn: retention is cheaper than acquisition; predict before cancellation.
- Dataset: IBM Telco Customer Churn — 7,043 customers, 20 features
  (15 categorical, 4 numeric), 26.5 % positive class.
- Split: 70 / 10 / 20 stratified.

## Slide 3 — System diagram (1:00 – 1:40)

Show the diagram from the report. Six MLflow-managed stages:
data loader → training → tuning → registry → serving → monitoring.
Backend store = SQLite; artifact root = local FS.

## Slide 4 — Experiment tracking (1:40 – 2:20)

- Three baselines: LogReg, RandomForest, GradientBoosting.
- Each run logs: params, metrics on train/val/test, ROC curve, confusion
  matrix, fitted pipeline with inferred signature and input example.
- Live demo (~15 s): switch to MLflow UI, show one baseline run page.

| Model | Test ROC-AUC | Test F1 |
| --- | --- | --- |
| LogisticRegression | **0.8426** | 0.605 |
| GradientBoosting   | 0.8390     | 0.571 |
| RandomForest       | 0.8223     | 0.539 |

*Aside: simple linear model wins out of the box — common for small tabular data.* ✂

## Slide 5 — Hyperparameter tuning (2:20 – 3:10)

- Hyperopt TPE, 20 trials over Gradient Boosting.
- **Parent run / child runs** pattern: search history is one parent, each
  trial is a nested run.
- Best test ROC-AUC = **0.8457** — tuned GB now beats LogReg.
- Show parent run page in UI with the trial table sorted by `val_roc_auc`. ✂ if tight

## Slide 6 — Model Registry (3:10 – 4:00)

- Best run promoted to a registered model `telco-churn-classifier`.
- Lifecycle: `None → staging → production` via MLflow aliases
  (`@staging`, `@production`), with legacy stage tags mirrored.
- CLI: `python -m src.registry register-best | transition 2 production | list`.
- Demo: show the Models tab in the UI with the green "production" alias.

## Slide 7 — Deployment (4:00 – 5:00)

- **`mlflow models serve`** on port 5001 — POST `/invocations` with the
  standard `dataframe_split` payload. Zero serving code; MLflow
  reconstructs the full sklearn Pipeline from the registered artifact.
- Live demo: `curl -X POST http://127.0.0.1:5001/invocations` with a
  3-row batch from the test split → returns `{"predictions":[0,1,0]}`.

## Slide 8 — Drift monitoring (5:00 – 6:30)

This is the headline slide. Show the table:

| Scenario | max_psi | drift_alert | prod_roc_auc |
| --- | --- | --- | --- |
| clean | 0.02 | ✗ | 0.834 |
| feature_drift | **1.49** | ✓ | 0.843 |
| concept_drift | 0.02 | ✗ | **0.654** |

Three takeaways:
1. PSI fires correctly on feature drift.
2. Performance stays roughly stable even with shifted features — the model
   generalizes through the perturbation.
3. **Concept drift is invisible to PSI** — inputs unchanged, but performance
   collapsed. → You need both input-distribution checks **and** performance
   metrics on labelled samples.

Show one Evidently HTML drift report briefly. ✂ if tight

## Slide 9 — What MLflow gave us (6:30 – 7:15)

- Single source of truth for params/metrics/artifacts → no ad-hoc spreadsheets.
- Registry decouples training from serving: redeploy by changing an alias.
- Pipeline-as-artifact eliminates training/serving skew.
- Same client API used for training, tuning, registry, monitoring → small
  surface area to learn.

## Slide 10 — Limitations + closing (7:15 – 8:00)

- Threshold not tuned (using 0.5); production needs cost-aware threshold.
- No automatic rollback on drift alert — natural next step.
- Local SQLite backend; would be Postgres + S3 in real deployment.
- Q&A.

---

## Demo plan (run before the talk, leave open in tabs)

1. Terminal with `./scripts/start_mlflow_ui.sh` running — open
   `http://127.0.0.1:5000` to **Experiments → telco_churn**.
2. Terminal with `./scripts/start_api.sh` running — `curl /health` ready in
   shell history.
3. Browser tab with one Evidently HTML report (`reports/drift_report_concept_drift.html`).

## Backup answers

- **"Why not XGBoost?"** Sklearn ships GB; an extra dependency wasn't
  justified given GB closed the gap to within 0.001 ROC-AUC after tuning.
- **"How would you retrain automatically?"** Schedule `train.py` on new data,
  the registry's `register-best` step + a performance gate would decide
  whether to flip the `@production` alias.
- **"Is SQLite production-ready?"** No — it's fine for a single-user demo
  and writes are file-locked. Postgres for any multi-writer deployment.
