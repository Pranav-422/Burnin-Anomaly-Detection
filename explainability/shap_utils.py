"""
Explainability layer (Module C) -- multi-parameter.

Wraps Module A and Module B outputs with human-readable justifications
that NAME which parameter(s) drove each flag (leakage current, Iddq,
propagation delay), ranked by severity when a component is flagged on
more than one parameter or by more than one method.

Uses SHAP for each parameter's tree-based drift model, and lot-relative
deviation summaries for the anomaly detector.
"""

import numpy as np
import pandas as pd
import shap

from data.generate_data import value_col, detect_parameters, PARAMETERS
from module_b.drift_predictor import feature_cols_for, engineer_features, _suffix


UNITS = {
    "leakage_ua": "uA", "iddq_ua": "uA", "prop_delay_ns": "ns", None: "uA",
}


def _display_name(p):
    """Human-friendly name for a parameter key (or legacy None)."""
    names = {
        "leakage_ua": "leakage current",
        "iddq_ua": "Iddq (standby current)",
        "prop_delay_ns": "propagation delay",
        None: "value",
    }
    return names.get(p, p)


def explain_module_a_for_parameter(row: pd.Series, parameter: str = None) -> tuple:
    """
    Plain-language justification for ONE parameter's Module A flag.
    Returns (text, severity) where severity is the max |Z-score| found,
    used for ranking when multiple parameters are flagged.
    """
    suf = _suffix(parameter)
    unit = UNITS.get(parameter, "")
    name = _display_name(parameter)
    col168 = value_col(168, parameter)

    reasons = []
    severity = 0.0
    if row.get(f"flag_zscore{suf}"):
        z = row[f"zscore_168h{suf}"]
        severity = max(severity, abs(z))
        reasons.append(
            f"168h {name} is {abs(z):.1f}sigma from its lot's mean "
            f"({row[col168]:.2f}{unit} vs lot mean {row[f'lot_mean_168h{suf}']:.2f}{unit})"
        )
    if row.get(f"flag_iqr{suf}"):
        reasons.append(
            f"168h {name} ({row[col168]:.2f}{unit}) falls outside the lot's "
            f"normal IQR range [{row[f'iqr_lower{suf}']:.2f}, {row[f'iqr_upper{suf}']:.2f}]{unit}"
        )
    if not reasons:
        return None, 0.0
    return "; and ".join(reasons), severity


def explain_module_a(row: pd.Series, parameters) -> list:
    """
    Returns a list of (severity, text) tuples -- one per parameter that
    triggered a Module A flag on this row, PLUS a multivariate entry if
    Isolation Forest flagged the component without any single parameter
    individually crossing its Z-score/IQR threshold.
    """
    entries = []
    any_single_param = False
    for p in parameters:
        text, severity = explain_module_a_for_parameter(row, parameter=p)
        if text:
            any_single_param = True
            entries.append((severity, f"[Module A - {_display_name(p)}] Flagged because {text}."))

    if row.get("flag_isoforest") and not any_single_param:
        entries.append((
            0.5,
            "[Module A - cross-parameter] The combination of readings across "
            "parameters and checkpoints is unusual relative to this lot, even "
            "though no single parameter individually crossed its threshold "
            "(multivariate outlier).",
        ))
    return entries


def build_shap_explainers(xgb_models: dict, background_df: pd.DataFrame) -> dict:
    """Builds one SHAP TreeExplainer per parameter's XGBoost model.
    xgb_models: {parameter_label: xgb_model}"""
    return {label: shap.TreeExplainer(model) for label, model in xgb_models.items()}


def explain_module_b_for_parameter(explainer, row: pd.DataFrame, parameter: str = None) -> tuple:
    """
    Returns (text, severity, contributions) for ONE parameter's Module B
    prediction on a single-row DataFrame, using SHAP values. severity is
    the magnitude of the strongest feature contribution.
    """
    cols = feature_cols_for(parameter)
    unit = UNITS.get(parameter, "")
    name = _display_name(parameter)

    X_row = row[cols]
    shap_values = explainer.shap_values(X_row)[0]
    contributions = dict(zip(cols, shap_values.round(3)))

    ranked = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    top_feature, top_value = ranked[0]
    direction = "pushed the predicted 168h value up" if top_value > 0 else "pulled the predicted 168h value down"

    text = (
        f"[Module B - {name}] The strongest driver was {top_feature} "
        f"({row[top_feature].values[0]:.3f}), which {direction} by "
        f"{abs(top_value):.2f}{unit} relative to baseline."
    )
    return text, abs(top_value), contributions


def build_full_report(
    module_a_df: pd.DataFrame,
    module_b_df: pd.DataFrame,
    xgb_models: dict,
    parameters=None,
) -> pd.DataFrame:
    """
    Merges Module A + B outputs into one QA-ready report per component,
    with a combined flag and a severity-ranked, parameter-named
    plain-language explanation for each.

    xgb_models: {parameter_label: xgb_model} -- label is the parameter
    key (e.g. "leakage_ua") or "value" for the legacy single-parameter case.
    """
    if parameters is None:
        parameters = detect_parameters(module_a_df)

    b_cols = ["component_id", "flag_module_b"]
    for p in parameters:
        suf = _suffix(p)
        b_cols += [f"predicted_168h{suf}", f"predicted_slope{suf}", f"flag_param_b{suf}"]

    merged = module_a_df.merge(module_b_df[b_cols], on="component_id", how="left")
    merged["final_flag"] = merged["flag_module_a"] | merged["flag_module_b"]

    explainers = build_shap_explainers(xgb_models, module_b_df)
    module_b_features = module_b_df.copy()
    for p in parameters:
        module_b_features = engineer_features(module_b_features, parameter=p)

    reasons = []
    triggered_all = []
    for _, row in merged.iterrows():
        entries = []  # (severity, text, display_name)

        if row["flag_module_a"]:
            for severity, text in explain_module_a(row, parameters):
                display = text.split("[")[1].split("]")[0].split(" - ", 1)[1]
                entries.append((severity, text, display))

        if row["flag_module_b"]:
            row_features = module_b_features[
                module_b_features["component_id"] == row["component_id"]
            ]
            for p in parameters:
                suf = _suffix(p)
                if row.get(f"flag_param_b{suf}"):
                    label = p if p is not None else "value"
                    explainer = explainers.get(label)
                    if explainer is not None:
                        text, severity, _ = explain_module_b_for_parameter(
                            explainer, row_features, parameter=p,
                        )
                        entries.append((severity, text, _display_name(p)))

        entries.sort(key=lambda x: x[0], reverse=True)  # most severe first
        reasons.append(" ".join(text for _, text, _ in entries) if entries else "No issues detected.")
        triggered_all.append(", ".join(sorted({name for _, _, name in entries})))

    merged["explanation"] = reasons
    merged["triggered_parameters_all"] = triggered_all
    return merged


if __name__ == "__main__":
    import joblib

    module_a_df = pd.read_csv("outputs/module_a_results.csv")
    module_b_df = pd.read_csv("outputs/module_b_results.csv")
    parameters = detect_parameters(module_a_df)

    xgb_models = {}
    for p in parameters:
        label = p if p is not None else "value"
        xgb_models[label] = joblib.load(f"outputs/xgb_model_{label}.joblib")

    report = build_full_report(module_a_df, module_b_df, xgb_models, parameters=parameters)

    tp = ((report["final_flag"]) & (report["is_defective"])).sum()
    fn = ((~report["final_flag"]) & (report["is_defective"])).sum()
    fp = ((report["final_flag"]) & (~report["is_defective"])).sum()
    recall = tp / (tp + fn) if (tp + fn) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0

    print(f"Combined (Module A OR B, all parameters) results:")
    print(f"  Recall: {recall:.1%}  |  Precision: {precision:.1%}  |  False Negatives: {fn}")
    print()
    flagged_sample = report[report["final_flag"]].head(3)
    for _, row in flagged_sample.iterrows():
        print(f"{row['component_id']} ({row['lot_id']}): {row['explanation']}\n")

    report.to_csv("outputs/final_report.csv", index=False)
