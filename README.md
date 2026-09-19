# ASTROLAB — AI-Driven Anomaly Detection in Component Burn-In & Screening

**Team:** CodeWalkers · **Problem Statement:** 26170 (ISRO, Dept. of Space) · **Smart India Hackathon (SIH 2026)**

ASTROLAB is an intelligent, physics-informed aerospace quality screening system designed for mission-critical space-grade electronics. It replaces blunt, static datasheet limits with:
1. **Module A** — dynamic, lot-relative outlier detection (per-parameter directional IQR + cross-parameter Isolation Forest).
2. **Module B** — time-series drift prediction forecasting 168h endpoints from early readings (0h/24h) with calibrated early rejection.
3. **Explainability Layer** — SHAP-backed, human-readable justification naming exact electrical parameter and feature drivers for every flagged component.
4. **ASTROLAB Web Suite** — a full-featured, responsive, ISRO-themed web console with operator authentication, planetary landing, project dossier, and real-time telemetry.

See `PRD_Burn_In_Anomaly_Detection.pdf` for full requirements and system architecture.

---

## The Aerospace Challenge

Spacecraft components cannot be repaired or replaced in orbit. A single latent defect escaping quality control can jeopardize a multi-crore satellite mission.
- **The Failure of Static Limits:** A component that drifts dangerously from 10µA to 48µA still passes a static 50µA datasheet threshold despite suffering severe gate-oxide breakdown. Conversely, healthy chips from high-baseline manufacturing lots get wrongly scrapped.
- **Chamber Bottlenecks & Energy Cost:** Standard 168-hour High-Temperature Operating Life (HTOL at 125°C) tests create massive queue delays and energy consumption.
- **The ASTROLAB Solution:** Evaluates parts relative to their specific manufacturing lot (Module A), forecasts 168h values at $T+24\text{h}$ with a $99.8^{\text{th}}$ percentile safety threshold to enable early rejection (Module B, saving ~85% chamber time), and provides full SHAP auditability.

---

## Multi-Parameter Electrical Screening

The system screens **three burn-in parameters** per component:

| Parameter | Column Key | Unit | Physics & Screening Significance |
|---|---|---|---|
| **Leakage Current** | `leakage_ua` | µA | Reverse-bias dielectric leakage. Exponential thermal drift indicates gate-oxide defect. |
| **Quiescent Supply Current (Iddq)** | `iddq_ua` | µA | Standby supply current. Sensitive to bridging faults, gate punch-through, and silicon micro-cracks. |
| **Propagation Delay** | `prop_delay_ns` | ns | Gate switching transition time. Captures NBTI and hot-carrier degradation across logic paths. |

Each component receives **one unified defect status** across all three parameters, mirroring real physical failure mechanisms (e.g. gate-oxide defects) that manifest across multiple electrical characteristics simultaneously.

---

## ASTROLAB Web Portal Architecture

ASTROLAB provides a unified, space-grade web experience served directly via FastAPI (`http://127.0.0.1:8000`):

| Page | URL Route | Description |
|---|---|---|
| **Planetary Home** | `/home` or `/dashboard/home.html` | ISRO-themed landing page with Earth imagery, project overview, and one-click access to the screening console. |
| **About Us Dossier** | `/about` or `/dashboard/about.html` | Comprehensive project dossier detailing the aerospace challenge, the 3 core pillars, screened electrical parameters, MIL-STD-883K compliance, and Team CodeWalkers. |
| **Operator Login** | `/login` or `/dashboard/login.html` | Hardened operator authentication portal featuring 5-attempt brute-force lockout (2-minute cooldown), input sanitization, password strength indicator, and session token management. |
| **Screening Console** | `/dashboard/` or `/dashboard/index.html` | Full interactive QA console protected by session auth guard: Executive Telemetry, Multi-parameter Lot Explorer, Enhanced Component Drill-Down, and Live Predictor. |

### Enhanced Component Drill-Down Features:
- **Changeable Lot Selector:** Filter and switch between lots directly inside the drill-down view, dynamically populating parts belonging to that lot with clear `⚠️ Flagged` and `Normal` indicators.
- **Accessible Lot Navigation:** One-click **"View Lot"** action button and clickable **Lot Sequence ID** linking directly into that lot's distribution inside the Lot Explorer.
- **Cleaned Telemetry:** Obsolete static placeholder strings removed in favor of dynamic, live-calculated diagnostics.

---

## Project Structure

```
burnin-anomaly-detection/
├── data/
│   └── generate_data.py          # physics-informed Arrhenius drift generator (multi-parameter)
├── module_a/
│   └── anomaly_detection.py      # directional IQR + lot Z-scores + joint Isolation Forest
├── module_b/
│   └── drift_predictor.py        # feature engineering, XGBoost 168h predictor, 99.8% safety slope
├── explainability/
│   └── shap_utils.py             # SHAP TreeExplainer + plain-language diagnostic generation
├── api/
│   └── main.py                   # FastAPI application + static web dashboard mount + REST endpoints
├── dashboard/
│   └── static/                   # ASTROLAB Web Portal (HTML5 / Tailwind CSS / Vanilla JS)
│       ├── home.html             # Planetary Landing Page
│       ├── about.html            # Mission Dossier & Architecture Page
│       ├── login.html            # Operator Authentication Screen (Hardened)
│       ├── index.html            # Main Operator Screening Console
│       └── earth_bg.jpg          # Earth visual asset
├── run_pipeline.py               # End-to-end orchestration pipeline
├── requirements.txt              # Project dependencies
├── antigravity_changes_log.md    # Audit log, re-engineering history, and performance verification
└── outputs/                      # Models, final QA reports, wide datasets
```

---

## Setup & Execution

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Run the Full ML Pipeline

Generates synthetic Arrhenius drift data, fits models across all 3 parameters, evaluates Module A & B, and outputs final QA reports:

```bash
python run_pipeline.py
```

Outputs created in `outputs/`:
- `burnin_data_wide.csv` — multi-parameter burn-in dataset (1000 components, 25 lots).
- `module_a_results.csv`, `module_b_results.csv` — per-module anomaly flags and scores.
- `xgb_model_<parameter>.joblib` — trained XGBoost models per parameter.
- `final_report.csv` — fused multi-parameter screening verdict with SHAP plain-language explanations.

---

## Running the ASTROLAB Web Portal

```bash
uvicorn api.main:app --reload --port 8000
```
- **Landing Page:** [http://127.0.0.1:8000/home](http://127.0.0.1:8000/home)
- **About Us Dossier:** [http://127.0.0.1:8000/about](http://127.0.0.1:8000/about)
- **Operator Login:** [http://127.0.0.1:8000/login](http://127.0.0.1:8000/login)
- **Screening Console:** [http://127.0.0.1:8000/dashboard/](http://127.0.0.1:8000/dashboard/)
- **Interactive Swagger API Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

*Test Operator Credentials:*
- **Operator ID:** `admin@astrolab.isro`
- **Password:** `isro@2026`

---

## API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System health check and component count. |
| `GET` | `/home` | Redirect to ASTROLAB Planetary Landing Page. |
| `GET` | `/about` | Redirect to Project Dossier & About Us Page. |
| `GET` | `/login` | Redirect to Operator Login Screen. |
| `GET` | `/api/overview` | Executive telemetry: total tested, flagged, yield, and parameters. |
| `GET` | `/api/lots` | Metadata and drift percentages for all manufacturing lots. |
| `GET` | `/api/lot/{lot_id}/scatter` | 168h scatter plot values, anomaly scores, and flags for a lot. |
| `GET` | `/api/lot/{lot_id}/components` | Paginated checkpoint readings for all components in a lot. |
| `GET` | `/api/component/{id}` | Full drill-down: all checkpoints, AI predictions, and SHAP rationale. |
| `GET` | `/api/flagged` | Global list of all flagged components across lots. |
| `GET` | `/api/export/report` | Download the full screening report as CSV. |
| `GET` | `/api/workflow-image` | Serves the screening architecture workflow diagram (PNG). |
| `POST` | `/predict` | Real-time what-if 168h inference from early 0h/24h readings. |

---

## Testing & Verification

A comprehensive 40-test pytest suite verifies data integrity, feature leakage prevention, and model generalization:

```bash
python -m pytest tests/ -v
```

- **Data Integrity Tests:** Confirms no feature leakage of future readings (`Value_96h`, `Value_168h`), ensures non-negative drift, and verifies uniform defect seeding.
- **Holdout Generalization:** Evaluates on unseen data (`seed=7` holdout) with zero data leakage.
- **Results:** **40 / 40 passed (100% pass rate)**.

---

## Standards Compliance

- **MIL-STD-883K Method 1015 Condition D:** High Temperature Operating Life (HTOL) screening protocol at 125°C under continuous dynamic excitation.
- **Zero-Defect Mandate:** Calibrated to prioritize 100% defect recall ($FN = 0$) for mission-critical flight hardware.
