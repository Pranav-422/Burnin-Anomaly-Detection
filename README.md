# AI-Driven Anomaly Detection in Component Burn-In & Screening

**Team:** CodeWalkers · **Problem Statement:** 26170 (ISRO, Dept. of Space) · **SIH 2026**

A predictive ML system that replaces static pass/fail burn-in limits with:
1. **Module A** — dynamic, lot-relative outlier detection (per parameter + cross-parameter)
2. **Module B** — time-series drift prediction with early rejection (per parameter)
3. **Explainability layer** — SHAP-backed, human-readable justification naming which parameter drove every flag

See `PRD_Burn_In_Anomaly_Detection.pdf` for full requirements and architecture.

## Multi-parameter screening

The system screens **three burn-in parameters** per component, per the problem statement's
"e.g. standby current Iddq, leakage currents, or propagation delays":

| Parameter | Column key | Unit | What it is |
|---|---|---|---|
| Leakage current | `leakage_ua` | µA | Standby leakage current |
| Iddq | `iddq_ua` | µA | Quiescent supply current |
| Propagation delay | `prop_delay_ns` | ns | Signal propagation delay |

Each component gets **one `is_defective` draw**, not one per parameter — a genuinely defective
part shows *correlated* elevated drift across all three parameters (mirroring a real physical
defect, e.g. a gate-oxide weakness, that manifests across multiple electrical characteristics at
once), rather than three independent unrelated anomalies. Module A's Isolation Forest runs
**across all checkpoints × all parameters jointly per lot**, which is what catches cross-parameter
anomalies a single-parameter check would miss (e.g. normal leakage but abnormal propagation delay).

**Backward compatibility:** every column-naming function in `module_a`/`module_b` uses a
`_suffix(parameter)` helper — empty string for `parameter=None` (legacy/single-parameter mode),
`f"_{parameter}"` otherwise. Calling `generate_burnin_dataset(parameters=["leakage_ua"])` +
`to_wide(..., parameters=["leakage_ua"])` reproduces the exact original single-parameter schema
(unsuffixed `Value_0h`, `Value_24h`, ... columns), and every downstream module
(`run_module_a`, `train_and_predict_all`, the API, the dashboard) auto-detects which mode it's in
via `detect_parameters()`.

## Project structure

```
burnin-anomaly-detection/
├── data/
│   └── generate_data.py       # physics-informed synthetic burn-in dataset (Arrhenius-style drift), multi-parameter
├── module_a/
│   └── anomaly_detection.py   # per-parameter Z-score/IQR + cross-parameter Isolation Forest
├── module_b/
│   └── drift_predictor.py     # per-parameter feature engineering, Linear Regression + XGBoost, safety slope
├── explainability/
│   └── shap_utils.py          # per-parameter SHAP explainer + severity-ranked plain-language reasons
├── api/
│   └── main.py                # FastAPI backend — /predict takes a `parameter` field to select which model
├── dashboard/
│   └── app.py                 # Streamlit QA dashboard — parameter selector on every tab + a flagged-parts table
├── run_pipeline.py            # end-to-end orchestration script
├── requirements.txt
└── outputs/                   # generated data, trained models (one per parameter), reports (created on run)
```

## Setup

```bash
pip install -r requirements.txt
```

## Run the full pipeline

Generates data, trains all three parameters' models in both modules, and produces the combined QA report:

```bash
python3 run_pipeline.py
```

This writes to `outputs/`:
- `burnin_data_wide.csv` — synthetic dataset (1000 components, 25 lots, ~5-6% latent defects), one
  `Value_<hour>h_<parameter>` column per checkpoint × parameter
- `module_a_results.csv`, `module_b_results.csv` — per-module flags and scores, per parameter, plus
  a `triggered_parameters` / `triggered_parameters_b` column naming which parameter(s) fired
- `xgb_model_<parameter>.joblib`, `linear_model_<parameter>.joblib` — one trained model pair per parameter
- `final_report.csv` — combined flags + severity-ranked explanations naming the parameter, ready for the dashboard

**Current results on synthetic data:**
| Metric | Value |
|---|---|
| Combined recall (defects caught) | 100% |
| False negatives | 0 |
| Module A precision | 84.7% (59 flagged / 50 real defects) |
| Module B precision | 90.9% (55 flagged / 50 real defects) |
| Precision (combined) | 79.4% (63 flagged / 50 real defects) |
| Module B test-set MAE | leakage ~0.36µA · Iddq ~0.64µA · prop delay ~0.13ns |

Thresholds are calibrated according to statistical process control (SPC) principles:
- **Module A IQR**: $k=2.0$ (extreme quartile departure) and Z-score $|Z| \ge 3.0\sigma$.
- **Module A Isolation Forest**: contamination rate set to $0.02$ (matching expected latent defect frequency) to prevent forced false alarms on clean lots.
- **Module B Safety Slope**: 99.5th percentile (~$3\sigma$ upper process limit) on known-good baseline drift rates, eliminating false alarms on healthy drift.

MAE above is reported strictly on the held-out **test split**, never by re-predicting on the full
training dataset — that was a real bug in an earlier single-parameter version (inflated apparent
accuracy) and was fixed as part of this multi-parameter work.

## Run the API

```bash
uvicorn api.main:app --reload --port 8000
```

`/predict` now takes a `parameter` field to select which parameter's model to use (defaults to
`leakage_ua`):
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "component_id": "C_TEST_001",
  "lot_id": "LOT_000",
  "parameter": "iddq_ua",
  "value_0h": 24.1,
  "value_24h": 29.5
}'
```

Other endpoints: `/component/{id}` (full per-parameter QA report), `/lot/{lot_id}/summary`
(per-parameter lot stats + flagged component IDs), `/lots`, `/health`.

## Run the dashboard

```bash
streamlit run dashboard/app.py
```

Four tabs, each with a parameter selector where relevant:
- **Lot explorer** — per-parameter, per-lot outlier strip plot with flagged parts highlighted
- **Component drill-down** — pick a flagged component, then pick which parameter's drift curve to view
- **Live predictor** — enter 0h/24h readings for any of the three parameters, get an instant PASS/REJECT with reasoning
- **Flagged parts** — every flagged component with which parameter(s) triggered it, filterable by parameter

## Run the tests

A real pytest suite covering data-leakage checks (per parameter), per-module logic on hand-crafted
edge cases, and generalization on a genuinely held-out dataset (different random seed, unseen by
the trained models):

```bash
python3 tests/generate_holdout_data.py   # one-time: builds the held-out test set (multi-parameter)
python3 -m pytest tests/ -v
```

Includes a specific check that would catch Module B accidentally using `Value_96h`/`Value_168h`
(for any parameter) as an input feature (`test_feature_cols_excludes_future_readings`), and a check
that every component's `is_defective` ground truth is identical across all three parameters'
rows (`test_same_defect_ground_truth_across_parameters`), confirming defects are seeded once per
component, not independently per parameter.

**Held-out generalization** (trained on seed=42, tested on seed=7, both physically-generated, never
mixed): Module A recall and the combined (A OR B) recall both stay ≥90% on the held-out set across
all three parameters, and each parameter's Module B MAE stays within 6% of that parameter's own
datasheet `max_limit` — confirming the models generalize rather than having memorized the training
seed.

## Notes on the data

No official ISRO dataset was available, so `data/generate_data.py` generates a **physics-informed
synthetic dataset** grounded in the real **Arrhenius equation**, the standard reliability-engineering
model relating stress-test temperature to real-world degradation rate:

```
AF = exp[ (Ea / k_B) x (1/T_use - 1/T_test) ]
```

Each component's drift rate under 125°C burn-in, for **each of the three parameters**, is derived
from that parameter's own activation energy (Ea):
- **Normal parts** — a mildly thermally-activated baseline mechanism per parameter (e.g. Ea ~0.35 eV
  for leakage). Stays nearly flat across the burn-in cycle.
- **Latent-defective parts** — a more thermally-activated failure mechanism per parameter (e.g.
  Ea ~0.70 eV for leakage), representative of e.g. TDDB/electromigration-type mechanisms. Because
  acceleration factor grows steeply with Ea under the Arrhenius model, these parts drift much
  faster under stress while staying, on average, well under each parameter's own datasheet limit —
  i.e. every seeded defect is genuinely latent, exactly the scenario ISRO's problem statement
  describes, for every parameter screened.

Every parameter's `BASE_RATE` growth constant is *derived* (not hand-tuned per parameter) so that a
defective part's 168h value lands at ~50% of that parameter's own `max_limit` on average — the same
calibration target the original single-parameter (leakage-only) model used, applied generically via
`_calibrate_base_rate()` rather than duplicated by hand for each new parameter.

This is a stronger physical basis than a generic power-law drift curve, since Ea and the resulting
acceleration factor are literally the mechanism burn-in testing is designed around (JEDEC/MIL-STD-883
reliability methodology).

For the real submission, swap `generate_burnin_dataset()` for a loader over the official dataset —
everything downstream (Module A, B, explainability, API, dashboard) is dataset-agnostic as long as
the wide-format columns (`component_id`, `lot_id`, `Value_<hour>h_<parameter>` per screened
parameter and checkpoint, optionally `is_defective` for evaluation) are present. Screening only one
parameter still works unchanged via the legacy unsuffixed schema (`Value_0h`, `Value_24h`, ...) — see
"Multi-parameter screening" above.
