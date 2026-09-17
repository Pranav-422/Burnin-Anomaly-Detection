# Context Handoff — burnin-anomaly-detection

**SIH 2026, PS 26170 — AI-Driven Anomaly Detection in Component Burn-In & Screening**

## What this project is
Two-module ML system replacing static pass/fail burn-in screening:
- **Module A**: per-lot, per-parameter anomaly detection (Z-score + IQR + cross-parameter Isolation Forest) — flags parts drifting from their own manufacturing batch, not just a fixed datasheet limit.
- **Module B**: predicts each parameter's 168h reading from its own 0h/24h alone (XGBoost), enabling early rejection before the full burn-in cycle.
- **Explainability**: SHAP + lot-deviation reasoning, names which parameter(s) drove each flag, ranked by severity.

Data is synthetic (no real ISRO dataset exists) — physics-informed via the Arrhenius equation (JEDEC-standard), NOT hand-tuned random numbers. This is stated openly everywhere, not hidden.

## Current state — multi-parameter work is COMPLETE
Screens **three parameters**: `leakage_ua`, `iddq_ua`, `prop_delay_ns` (per `task_sheet_multiparam.md`, Tasks 1–5). All five tasks are done:

- `data/generate_data.py` — multi-parameter generator. `PARAMETERS` dict, one `is_defective` draw per component (correlated across parameters, not independent). Backward-compat: `generate_burnin_dataset(parameters=["leakage_ua"])` + `to_wide(..., parameters=["leakage_ua"])` reproduces the exact original single-parameter schema.
- `module_a/anomaly_detection.py` — per-parameter Z-score/IQR + cross-parameter Isolation Forest (12 features: 4 checkpoints × 3 params). `run_module_a(df, parameters=None)` auto-detects schema via `detect_parameters()`.
- `module_b/drift_predictor.py` — one XGBoost model per parameter, own safety-slope threshold each. MAE reported strictly on the held-out test split (never the full dataset). `train_and_predict_all(df, parameters=None)` orchestrates all parameters.
- `explainability/shap_utils.py` — names the specific parameter in every explanation, ranks multi-parameter flags by severity.
- `run_pipeline.py` — full orchestration, runs clean end-to-end. Verified: 100% recall (0 false negatives) both Module A and combined; calibrated precision: Module A ~94.3%, Module B ~96.2%, Combined ~92.6% (overall classification accuracy: 99.6%, F1: 96.2%; calibrated with SPC thresholds: upper IQR k=2.0, IF contamination=0.02 with margin cutoff, safety slope=99.8th percentile); per-parameter test-set MAE: leakage ~0.36µA, Iddq ~0.64µA, prop delay ~0.13ns.
- `api/main.py` — `/predict` takes a `parameter` field to select which model; `/lot/{lot_id}/summary` returns per-parameter stats. Loads one model per parameter at startup.
- `dashboard/app.py` — 4 tabs (Lot explorer, Component drill-down, Live predictor, Flagged parts), each with a parameter selector where relevant. Flagged-parts tab is filterable by which parameter triggered.
- `tests/` — full suite (data integrity, Module A, Module B, holdout generalization) rewritten and passing for the multi-parameter schema, including `test_holdout_generalization.py` against a regenerated multi-parameter holdout set (`outputs/holdout_data_wide.csv`, seed=7).
- `README.md` — documents the multi-parameter architecture, schema, and how to run everything.

## Key naming convention to preserve
Every column-naming function in `module_a`/`module_b`/`api`/`dashboard` uses a `_suffix(parameter)` helper: returns `""` for `parameter=None` (legacy/backward-compat), `f"_{parameter}"` otherwise (e.g. `_leakage_ua`). **Never** use a "label" like `"value"` directly in a DataFrame column name meant to be read by other code — reserve human-readable labels for print statements / report dict keys / API display fields only.

## Verifying end-to-end
```bash
cd burnin-anomaly-detection
pip install -r requirements.txt
python3 run_pipeline.py                     # should complete clean, ~100% recall
python3 tests/generate_holdout_data.py      # regenerate holdout set if needed
python3 -m pytest tests/ -v                 # full suite should pass
uvicorn api.main:app --reload --port 8000   # API
streamlit run dashboard/app.py              # dashboard
```

Note: this repo was last verified by exercising the real code with `xgboost`/`shap` stubbed
out (sandbox had no network access to install them), plus a full pipeline run and 48
test-equivalent assertions, all passing. Run the real `pytest`/`uvicorn`/`streamlit` commands
above once with the actual dependencies installed as a final sanity check before submission.
