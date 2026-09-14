"""
QA Dashboard (Streamlit) -- multi-parameter.

Gives a QA inspector a visual view of:
  - Overall screening summary (flagged vs passed, recall/precision
    against ground truth where available)
  - Lot-level distribution with outliers highlighted, for whichever
    screened parameter (leakage current, Iddq, propagation delay) the
    inspector selects
  - Per-component drill-down with plain-language explanation, and a
    per-parameter drift-curve view
  - A live "what-if" predictor for early (0h/24h) readings, per parameter
  - A flagged-parts table showing which parameter(s) triggered each flag

Run with:
    streamlit run dashboard/app.py
"""

import os
import sys

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate_data import detect_parameters, value_col, CHECKPOINTS
from module_b.drift_predictor import (
    engineer_features, feature_cols_for, compute_safety_slope, _param_label,
)
from explainability.shap_utils import (
    build_shap_explainers, explain_module_b_for_parameter, _display_name, UNITS,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

st.set_page_config(page_title="Burn-In Anomaly Detection - QA Dashboard", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv(f"{OUT_DIR}/final_report.csv")
    parameters = detect_parameters(df)
    return df, parameters


@st.cache_resource
def load_models(_df, parameters):
    models = {}
    for p in parameters:
        label = _param_label(p)
        models[label] = joblib.load(f"{OUT_DIR}/xgb_model_{label}.joblib")
    explainers = build_shap_explainers(models, _df)
    return models, explainers


df, parameters = load_data()
xgb_models, explainers = load_models(df, parameters)

param_labels = [_param_label(p) for p in parameters]
label_to_param = dict(zip(param_labels, parameters))
param_display = {label: _display_name(p) for label, p in label_to_param.items()}

st.title("Burn-In Anomaly Detection — QA Dashboard")
st.caption("SIH 2026 · Problem Statement 26170 · ISRO Dept. of Space · Team CodeWalkers")
st.caption(f"Screening {len(parameters)} parameter(s): {', '.join(param_display.values())}")

# ---------------- Top summary row ----------------
col1, col2, col3, col4, col5 = st.columns(5)
n_total = len(df)
n_flagged = int(df["final_flag"].sum())
tp = int(((df["final_flag"]) & (df["is_defective"])).sum())
fn = int(((~df["final_flag"]) & (df["is_defective"])).sum())
fp = int(((df["final_flag"]) & (~df["is_defective"])).sum())
recall = tp / (tp + fn) if (tp + fn) else 0

col1.metric("Total components", n_total)
col2.metric("Flagged for review", n_flagged)
col3.metric("Recall (defects caught)", f"{recall:.0%}")
col4.metric("False negatives", fn, delta_color="inverse")
col5.metric("False positives", fp)

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    ["Lot explorer", "Component drill-down", "Live predictor", "Flagged parts"]
)

# ---------------- Tab 1: Lot explorer ----------------
with tab1:
    st.subheader("Lot-relative outlier view")
    lc1, lc2 = st.columns([1, 1])
    lot_ids = sorted(df["lot_id"].unique())
    selected_lot = lc1.selectbox("Select a lot", lot_ids)
    selected_param_label = lc2.selectbox(
        "Parameter", param_labels, format_func=lambda x: param_display[x], key="lot_param",
    )
    selected_param = label_to_param[selected_param_label]
    unit = UNITS.get(selected_param, "")
    col168 = value_col(168, selected_param)
    display_name = param_display[selected_param_label]

    lot_df = df[df["lot_id"] == selected_lot].copy()
    lot_df["status"] = lot_df["final_flag"].map({True: "Flagged", False: "Normal"})

    fig = px.strip(
        lot_df, x=col168, y=[f"168h {display_name}"] * len(lot_df),
        color="status",
        color_discrete_map={"Flagged": "#D85A30", "Normal": "#5DCAA5"},
        hover_data=["component_id", "anomaly_score", "triggered_parameters_all"],
        title=f"{selected_lot} — 168h {display_name}, flagged parts highlighted",
    )
    fig.add_vline(
        x=lot_df[col168].mean(), line_dash="dash",
        annotation_text="lot mean", annotation_position="top",
    )
    fig.update_layout(height=280, yaxis_title="", xaxis_title=f"{display_name} ({unit})")
    st.plotly_chart(fig, use_container_width=True)

    checkpoint_cols = [value_col(h, selected_param) for h in CHECKPOINTS]
    st.dataframe(
        lot_df[["component_id"] + checkpoint_cols +
               ["anomaly_score", "final_flag", "triggered_parameters_all"]]
        .sort_values("anomaly_score", ascending=False),
        use_container_width=True, height=250,
    )

# ---------------- Tab 2: Component drill-down ----------------
with tab2:
    st.subheader("Component-level explanation")
    flagged_ids = df[df["final_flag"]]["component_id"].tolist()
    selected_component = st.selectbox(
        "Select a flagged component", flagged_ids, format_func=lambda x: x,
    )

    if selected_component:
        row = df[df["component_id"] == selected_component].iloc[0]
        st.caption(f"Triggered parameter(s): **{row['triggered_parameters_all']}**")

        drill_param_label = st.selectbox(
            "View drift curve for", param_labels, format_func=lambda x: param_display[x],
            key="drill_param",
        )
        drill_param = label_to_param[drill_param_label]
        suf = "" if drill_param is None else f"_{drill_param}"
        unit = UNITS.get(drill_param, "")
        display_name = param_display[drill_param_label]

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Anomaly score", f"{row['anomaly_score']:.1f} / 100")
            st.metric("Lot", row["lot_id"])
            st.metric(f"168h {display_name}", f"{row[value_col(168, drill_param)]:.2f} {unit}")
            st.metric("Predicted 168h (from 0h/24h)", f"{row[f'predicted_168h{suf}']:.2f} {unit}")

        with c2:
            values = [row[value_col(h, drill_param)] for h in CHECKPOINTS]
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=CHECKPOINTS, y=values, mode="lines+markers",
                                       name="Actual", line=dict(color="#D85A30", width=3)))
            fig2.add_trace(go.Scatter(
                x=[24, 168], y=[row[value_col(24, drill_param)], row[f"predicted_168h{suf}"]],
                mode="lines+markers", name="Predicted (from 0h/24h)",
                line=dict(color="#7F77DD", width=2, dash="dash"),
            ))
            fig2.update_layout(
                title=f"{display_name} drift curve vs. prediction", height=300,
                xaxis_title="Hours", yaxis_title=f"{display_name} ({unit})",
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.info(f"**Explanation:** {row['explanation']}")

# ---------------- Tab 3: Live predictor ----------------
with tab3:
    st.subheader("Live drift prediction (what-if)")
    st.caption("Enter early burn-in readings for a component still in test — no need to wait for 168h.")

    live_param_label = st.selectbox(
        "Parameter", param_labels, format_func=lambda x: param_display[x], key="live_param",
    )
    live_param = label_to_param[live_param_label]
    unit = UNITS.get(live_param, "")
    c0col, c24col = value_col(0, live_param), value_col(24, live_param)
    suf = "" if live_param is None else f"_{live_param}"

    lc1, lc2, lc3 = st.columns(3)
    lot_choice = lc1.selectbox("Lot", sorted(df["lot_id"].unique()), key="live_lot")
    v0 = lc2.number_input(f"Value at 0h ({unit})", min_value=0.0, value=10.0, step=0.1, key="live_v0")
    v24 = lc3.number_input(f"Value at 24h ({unit})", min_value=0.0, value=12.0, step=0.1, key="live_v24")

    if st.button("Predict 168h and assess risk", type="primary"):
        lot_mean_0h = df[df["lot_id"] == lot_choice][c0col].mean()
        row_df = pd.DataFrame([{
            "component_id": "LIVE_TEST", "lot_id": lot_choice,
            c0col: v0, c24col: v24, f"lot_mean_0h{suf}": lot_mean_0h,
        }])
        row_df = engineer_features(row_df, parameter=live_param)
        row_df[f"lot_mean_0h{suf}"] = lot_mean_0h

        model = xgb_models[live_param_label]
        predicted_168h = float(model.predict(row_df[feature_cols_for(live_param)])[0])
        predicted_slope = (predicted_168h - v24) / (168 - 24)
        safety_slope = compute_safety_slope(df, parameter=live_param, percentile=95.0)
        flagged = predicted_slope > safety_slope
        explainer = explainers[live_param_label]
        reason, _, _ = explain_module_b_for_parameter(explainer, row_df, parameter=live_param)

        rc1, rc2 = st.columns(2)
        rc1.metric("Predicted 168h value", f"{predicted_168h:.2f} {unit}")
        rc2.metric("Predicted drift slope", f"{predicted_slope:.4f} {unit}/hr",
                   delta=f"threshold {safety_slope}", delta_color="inverse")

        if flagged:
            st.error(f"**REJECT — early rejection recommended.** {reason}")
        else:
            st.success(f"**PASS.** {reason}")

# ---------------- Tab 4: Flagged parts ----------------
with tab4:
    st.subheader("All flagged components, with triggering parameter(s)")
    st.caption(
        "Every part flagged by Module A and/or Module B, across all screened "
        "parameters, with which parameter(s) drove each flag."
    )

    flagged_df = df[df["final_flag"]].copy()
    param_filter = st.multiselect(
        "Filter by triggering parameter",
        options=list(param_display.values()),
        default=[],
        help="Leave empty to show all flagged parts.",
    )
    if param_filter:
        pattern = "|".join(param_filter)
        flagged_df = flagged_df[
            flagged_df["triggered_parameters_all"].str.contains(pattern, case=False, na=False)
        ]

    display_cols = [
        "component_id", "lot_id", "anomaly_score", "triggered_parameters_all",
        "flag_module_a", "flag_module_b", "is_defective",
    ]
    st.dataframe(
        flagged_df[display_cols].sort_values("anomaly_score", ascending=False),
        use_container_width=True, height=450,
    )
    st.caption(f"{len(flagged_df)} of {n_flagged} flagged components shown.")
