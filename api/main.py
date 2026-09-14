"""
FastAPI backend -- multi-parameter.

Exposes the trained per-parameter models + explainability layer over
HTTP so the QA dashboard (or any other client) can request a risk
assessment for a component given its early burn-in readings, for
whichever screened parameter (leakage current, Iddq, or propagation
delay) the caller is testing.

Run with:
    uvicorn api.main:app --reload --port 8000
"""

import os
import sys
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate_data import ALL_PARAMETERS, CHECKPOINTS, detect_parameters, value_col
from module_b.drift_predictor import (
    engineer_features, feature_cols_for, compute_safety_slope, _param_label,
)
from explainability.shap_utils import (
    build_shap_explainers, explain_module_b_for_parameter, _display_name, UNITS,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUTS_DIR = os.path.join(OUT_DIR, "outputs")
DASHBOARD_DIR = os.path.join(OUT_DIR, "dashboard", "static")

app = FastAPI(
    title="Burn-In Anomaly Detection API",
    description="AI-driven anomaly detection for ISRO component burn-in screening (SIH 2026, PS 26170)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Loaded once at startup -----
_state = {}


@app.on_event("startup")
def load_artifacts():
    wide_df = pd.read_csv(f"{OUTPUTS_DIR}/burnin_data_wide.csv")
    parameters = detect_parameters(wide_df)  # e.g. ["iddq_ua","leakage_ua","prop_delay_ns"], or [None] legacy

    xgb_models = {}
    safety_slopes = {}
    param_by_label = {}
    for p in parameters:
        label = _param_label(p)
        xgb_models[label] = joblib.load(f"{OUTPUTS_DIR}/xgb_model_{label}.joblib")
        safety_slopes[label] = compute_safety_slope(wide_df, parameter=p, percentile=99.5)
        param_by_label[label] = p

    _state["wide_df"] = wide_df
    _state["parameters"] = parameters
    _state["param_by_label"] = param_by_label
    _state["xgb_models"] = xgb_models
    _state["safety_slopes"] = safety_slopes
    _state["explainers"] = build_shap_explainers(xgb_models, wide_df)
    _state["final_report"] = pd.read_csv(f"{OUTPUTS_DIR}/final_report.csv")

    print("Artifacts loaded. Parameters:", list(param_by_label.keys()))
    print("Safety slopes:", safety_slopes)

    # Mount static dashboard files (after startup so dir can exist)
    if os.path.isdir(DASHBOARD_DIR):
        app.mount("/dashboard", StaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard")
        print(f"Dashboard mounted at /dashboard from {DASHBOARD_DIR}")


class ComponentReading(BaseModel):
    component_id: str = Field(..., example="C_TEST_001")
    lot_id: str = Field(..., example="LOT_000")
    parameter: str = Field(
        "leakage_ua", example="leakage_ua",
        description=f"Which screened parameter this reading is for. One of: {ALL_PARAMETERS}",
    )
    value_0h: float = Field(..., example=10.2)
    value_24h: float = Field(..., example=13.8)


class PredictionResponse(BaseModel):
    component_id: str
    parameter: str
    predicted_168h: float
    predicted_slope: float
    safety_slope_threshold: float
    recommendation: str
    explanation: str
    feature_contributions: dict


@app.get("/")
def root():
    return {
        "service": "Burn-In Anomaly Detection API",
        "problem_statement": "SIH 2026 - PS 26170",
        "parameters_screened": list(_state.get("param_by_label", {}).keys()) or ALL_PARAMETERS,
        "endpoints": ["/predict", "/component/{component_id}", "/lot/{lot_id}/summary", "/lots", "/health"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "components_loaded": len(_state.get("wide_df", []))}


# ========================== Dashboard API ==========================

@app.get("/api/overview")
def api_overview():
    """Executive telemetry: KPIs, yield breakdown, and summary stats
    for the Overview Dashboard screen."""
    report = _state["final_report"]
    n_total = len(report)
    n_flagged = int(report["final_flag"].sum())
    tp = int(((report["final_flag"]) & (report["is_defective"])).sum())
    fn = int(((~report["final_flag"]) & (report["is_defective"])).sum())
    fp = int(((report["final_flag"]) & (~report["is_defective"])).sum())
    recall = round(tp / (tp + fn), 4) if (tp + fn) else 0
    precision = round(tp / (tp + fp), 4) if (tp + fp) else 0
    n_passed = n_total - n_flagged
    yield_pct = round(n_passed / n_total * 100, 1) if n_total else 0
    drift_rate = round(n_flagged / n_total * 100, 2) if n_total else 0

    # Per-parameter stats
    param_stats = {}
    for p in _state["parameters"]:
        label = _param_label(p)
        col168 = value_col(168, p)
        unit = UNITS.get(p, "")
        param_stats[label] = {
            "display_name": _display_name(p),
            "unit": unit,
            "mean_168h": round(float(report[col168].mean()), 3),
            "std_168h": round(float(report[col168].std()), 3),
        }

    return {
        "total_tested": n_total,
        "flagged_outliers": n_flagged,
        "recall_rate": recall,
        "precision": precision,
        "false_negatives": fn,
        "false_positives": fp,
        "yield_pct": yield_pct,
        "passed": n_passed,
        "drift_rate_pct": drift_rate,
        "parameters": param_stats,
        "n_lots": report["lot_id"].nunique(),
    }


@app.get("/api/workflow-image")
def api_workflow_image():
    """Serves the screening architecture workflow diagram."""
    img_path = os.path.join(DASHBOARD_DIR, "workflow.png")
    if os.path.isfile(img_path):
        return FileResponse(img_path, media_type="image/png")
    fallback = os.path.join(OUT_DIR, "dashboard", "screenshot_overview.png")
    if os.path.isfile(fallback):
        return FileResponse(fallback, media_type="image/png")
    raise HTTPException(status_code=404, detail="Workflow image not found")


@app.get("/api/lots")
def api_lots():
    """Returns all lots with summary metadata for the lots table."""
    report = _state["final_report"]
    lots = sorted(report["lot_id"].unique())
    result = []

    # Use the first parameter for the sparkline / drift display
    first_param = _state["parameters"][0]
    col168 = value_col(168, first_param)

    for lot_id in lots:
        lot_df = report[report["lot_id"] == lot_id]
        n = len(lot_df)
        n_flagged = int(lot_df["final_flag"].sum())
        drift_pct = round(n_flagged / n * 100, 2) if n else 0
        mean_168 = round(float(lot_df[col168].mean()), 2)

        # Generate sparkline data (min-max normalized checkpoint means)
        sparkline = []
        for h in CHECKPOINTS:
            col = value_col(h, first_param)
            if col in lot_df.columns:
                sparkline.append(round(float(lot_df[col].mean()), 3))

        status = "High Drift" if drift_pct > 3.0 else "Nominal"

        result.append({
            "lot_id": lot_id,
            "n_components": n,
            "n_flagged": n_flagged,
            "drift_pct": drift_pct,
            "status": status,
            "mean_168h": mean_168,
            "sparkline": sparkline,
        })

    return result


@app.get("/api/lot/{lot_id}/scatter")
def api_lot_scatter(lot_id: str, parameter: str = "leakage_ua"):
    """Returns scatter-plot data for a lot: each component's 168h value,
    flag status, and anomaly score."""
    report = _state["final_report"]
    lot_df = report[report["lot_id"] == lot_id]
    if lot_df.empty:
        raise HTTPException(status_code=404, detail="Lot not found")

    param_by_label = _state["param_by_label"]
    if parameter not in param_by_label:
        raise HTTPException(status_code=400, detail=f"Unknown parameter. Known: {list(param_by_label.keys())}")
    p = param_by_label[parameter]
    col168 = value_col(168, p)
    unit = UNITS.get(p, "")

    points = []
    for idx, row in lot_df.iterrows():
        points.append({
            "component_id": row["component_id"],
            "value_168h": round(float(row[col168]), 3),
            "flagged": bool(row["final_flag"]),
            "anomaly_score": round(float(row["anomaly_score"]), 1) if pd.notna(row.get("anomaly_score")) else None,
        })

    lot_mean = round(float(lot_df[col168].mean()), 3)
    lot_std = round(float(lot_df[col168].std()), 3)

    return {
        "lot_id": lot_id,
        "parameter": parameter,
        "display_name": _display_name(p),
        "unit": unit,
        "lot_mean": lot_mean,
        "lot_std": lot_std,
        "points": points,
    }


@app.get("/api/lot/{lot_id}/components")
def api_lot_components(
    lot_id: str,
    parameter: str = "leakage_ua",
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    flagged_only: bool = False,
    search: str = "",
):
    """Returns paginated component table for a lot with all checkpoint readings."""
    report = _state["final_report"]
    lot_df = report[report["lot_id"] == lot_id].copy()
    if lot_df.empty:
        raise HTTPException(status_code=404, detail="Lot not found")

    param_by_label = _state["param_by_label"]
    if parameter not in param_by_label:
        raise HTTPException(status_code=400, detail=f"Unknown parameter. Known: {list(param_by_label.keys())}")
    p = param_by_label[parameter]
    unit = UNITS.get(p, "")

    if flagged_only:
        lot_df = lot_df[lot_df["final_flag"]]
    if search:
        lot_df = lot_df[lot_df["component_id"].str.contains(search, case=False, na=False)]

    lot_df = lot_df.sort_values("anomaly_score", ascending=False)
    total = len(lot_df)
    total_pages = max(1, (total + page_size - 1) // page_size)
    start = (page - 1) * page_size
    page_df = lot_df.iloc[start:start + page_size]

    rows = []
    for _, row in page_df.iterrows():
        readings = {}
        for h in CHECKPOINTS:
            col = value_col(h, p)
            if col in row.index:
                readings[f"{h}h"] = round(float(row[col]), 3) if pd.notna(row[col]) else None
        rows.append({
            "component_id": row["component_id"],
            "readings": readings,
            "anomaly_score": round(float(row["anomaly_score"]), 1) if pd.notna(row.get("anomaly_score")) else None,
            "flagged": bool(row["final_flag"]),
            "is_defective": bool(row["is_defective"]),
            "status": "Flagged" if row["final_flag"] else "Normal",
        })

    return {
        "lot_id": lot_id,
        "parameter": parameter,
        "display_name": _display_name(p),
        "unit": unit,
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
        "components": rows,
    }


@app.get("/api/component/{component_id}")
def api_component_detail(component_id: str):
    """Full component drill-down: risk score, all checkpoint values per
    parameter, predictions, and explanation."""
    report = _state["final_report"]
    match = report[report["component_id"] == component_id]
    if match.empty:
        raise HTTPException(status_code=404, detail="Component not found")
    row = match.iloc[0]

    # Build per-parameter readings + predictions
    param_data = {}
    for p in _state["parameters"]:
        label = _param_label(p)
        suf = "" if p is None else f"_{p}"
        unit = UNITS.get(p, "")
        readings = {}
        for h in CHECKPOINTS:
            col = value_col(h, p)
            if col in row.index:
                readings[f"{h}h"] = round(float(row[col]), 3) if pd.notna(row[col]) else None

        predicted_col = f"predicted_168h{suf}"
        predicted_168h = round(float(row[predicted_col]), 3) if predicted_col in row.index and pd.notna(row.get(predicted_col)) else None

        param_data[label] = {
            "display_name": _display_name(p),
            "unit": unit,
            "readings": readings,
            "predicted_168h": predicted_168h,
        }

    anomaly_score = round(float(row["anomaly_score"]), 1) if pd.notna(row.get("anomaly_score")) else 0
    explanation = row.get("explanation", "")
    triggered = row.get("triggered_parameters_all", "")

    # Compute SHAP contributions for the top parameter
    top_contributions = {}
    for p in _state["parameters"]:
        label = _param_label(p)
        if label in _state["explainers"]:
            try:
                _, _, contribs = explain_module_b_for_parameter(
                    _state["explainers"][label],
                    match,
                    parameter=p,
                )
                top_contributions[label] = {k: round(float(v), 4) for k, v in contribs.items()}
            except Exception:
                top_contributions[label] = {}

    return {
        "component_id": component_id,
        "lot_id": row["lot_id"],
        "anomaly_score": anomaly_score,
        "flagged": bool(row["final_flag"]),
        "is_defective": bool(row["is_defective"]),
        "explanation": str(explanation) if pd.notna(explanation) else "",
        "triggered_parameters": str(triggered) if pd.notna(triggered) else "",
        "flag_module_a": bool(row.get("flag_module_a", False)),
        "flag_module_b": bool(row.get("flag_module_b", False)),
        "parameters": param_data,
        "shap_contributions": top_contributions,
    }


@app.get("/api/flagged")
def api_flagged_components(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """Returns all flagged components across all lots."""
    report = _state["final_report"]
    flagged = report[report["final_flag"]].sort_values("anomaly_score", ascending=False)
    total = len(flagged)
    total_pages = max(1, (total + page_size - 1) // page_size)
    start = (page - 1) * page_size
    page_df = flagged.iloc[start:start + page_size]

    rows = []
    for _, row in page_df.iterrows():
        rows.append({
            "component_id": row["component_id"],
            "lot_id": row["lot_id"],
            "anomaly_score": round(float(row["anomaly_score"]), 1) if pd.notna(row.get("anomaly_score")) else None,
            "triggered_parameters": str(row.get("triggered_parameters_all", "")) if pd.notna(row.get("triggered_parameters_all")) else "",
            "flag_module_a": bool(row.get("flag_module_a", False)),
            "flag_module_b": bool(row.get("flag_module_b", False)),
        })

    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
        "components": rows,
    }


# ========================== Original Endpoints ==========================

@app.post("/predict", response_model=PredictionResponse)
def predict(reading: ComponentReading):
    """
    Predicts the 168h value from early (0h, 24h) readings for a new
    or in-progress component, for whichever parameter the reading is
    for, and flags it if the predicted drift slope exceeds that
    parameter's own safety threshold. Uses lot-level baseline stats
    from the reference dataset for the given lot_id if available.
    """
    param_by_label = _state["param_by_label"]
    if reading.parameter not in param_by_label:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown parameter '{reading.parameter}'. Known: {list(param_by_label.keys())}",
        )
    p = param_by_label[reading.parameter]  # actual parameter key (or None for legacy)
    suf = "" if p is None else f"_{p}"
    c0, c24 = value_col(0, p), value_col(24, p)

    wide_df = _state["wide_df"]
    lot_matches = wide_df[wide_df["lot_id"] == reading.lot_id]
    lot_mean_0h = lot_matches[c0].mean() if len(lot_matches) else reading.value_0h

    row = pd.DataFrame([{
        "component_id": reading.component_id,
        "lot_id": reading.lot_id,
        c0: reading.value_0h,
        c24: reading.value_24h,
        f"lot_mean_0h{suf}": lot_mean_0h,
    }])
    row = engineer_features(row, parameter=p)
    # engineer_features recomputes lot_mean_0h via groupby, which is fine
    # only if the lot exists in this frame -- for a single-row frame it
    # will just equal Value_0h. Override with the reference-dataset value.
    row[f"lot_mean_0h{suf}"] = lot_mean_0h

    X = row[feature_cols_for(p)]
    model = _state["xgb_models"][reading.parameter]
    predicted_168h = float(model.predict(X)[0])
    predicted_slope = (predicted_168h - reading.value_24h) / (168 - 24)
    safety_slope = _state["safety_slopes"][reading.parameter]
    flagged = predicted_slope > safety_slope

    explainer = _state["explainers"][reading.parameter]
    text, _, contributions = explain_module_b_for_parameter(explainer, row, parameter=p)

    return PredictionResponse(
        component_id=reading.component_id,
        parameter=reading.parameter,
        predicted_168h=round(predicted_168h, 3),
        predicted_slope=round(predicted_slope, 5),
        safety_slope_threshold=safety_slope,
        recommendation="REJECT - early rejection recommended" if flagged else "PASS",
        explanation=text,
        feature_contributions={k: float(v) for k, v in contributions.items()},
    )


@app.get("/component/{component_id}")
def get_component(component_id: str):
    """Looks up a previously-processed component's full QA report
    (anomaly score, flags, explanation, per-parameter readings) from
    the reference dataset."""
    report = _state["final_report"]
    match = report[report["component_id"] == component_id]
    if match.empty:
        raise HTTPException(status_code=404, detail="Component not found")
    row = match.iloc[0].to_dict()
    return {k: (None if pd.isna(v) else v) for k, v in row.items()}


@app.get("/lot/{lot_id}/summary")
def get_lot_summary(lot_id: str):
    """Returns lot-level statistics (per parameter) and how many
    components in the lot were flagged -- useful for the dashboard's
    lot view."""
    report = _state["final_report"]
    lot_df = report[report["lot_id"] == lot_id]
    if lot_df.empty:
        raise HTTPException(status_code=404, detail="Lot not found")

    parameter_stats = {}
    for p in _state["parameters"]:
        label = _param_label(p)
        col168 = value_col(168, p)
        parameter_stats[label] = {
            "mean_168h": round(float(lot_df[col168].mean()), 3),
            "std_168h": round(float(lot_df[col168].std()), 3),
        }

    return {
        "lot_id": lot_id,
        "n_components": len(lot_df),
        "n_flagged": int(lot_df["final_flag"].sum()),
        "parameter_stats": parameter_stats,
        "flagged_component_ids": lot_df[lot_df["final_flag"]]["component_id"].tolist(),
    }


@app.get("/lots")
def list_lots():
    report = _state["final_report"]
    return sorted(report["lot_id"].unique().tolist())
