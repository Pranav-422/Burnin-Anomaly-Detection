"""
Module A: Dynamic outlier detection -- multi-parameter.

Compares each component against its OWN LOT's statistical behavior,
per PARAMETER, not just the fixed datasheet limit. Three layers:

1. Per-lot Z-score on each parameter's 168h value (fast, explainable,
   per-parameter)
2. Per-lot IQR on each parameter's 168h value (robust to skew)
3. Isolation Forest across ALL checkpoints x ALL parameters jointly,
   per lot -- this is where CROSS-parameter anomalies show up (e.g. a
   part with normal leakage but abnormal propagation delay)

A component's final flag is the OR across every parameter and every
method (recall-biased, unchanged from the single-parameter design --
just extended in scope). The output names exactly which parameter(s)
triggered each flag, required for explainability (Task 4).

Backward-compatible: if the dataframe has only the legacy unsuffixed
Value_0h/24h/96h/168h columns, this runs exactly as the original
single-parameter Module A did (parameters=[None]).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from data.generate_data import value_col, detect_parameters, CHECKPOINTS


def _param_label(p):
    """Human-readable label for a parameter key (or the legacy None)."""
    return p if p is not None else "value"


def _suffix(p):
    """Empty string for the legacy (None) parameter -- so column names
    are exactly zscore_168h, flag_zscore, etc. with no suffix, matching
    the original single-parameter module's names precisely."""
    return "" if p is None else f"_{p}"


def zscore_flags(df: pd.DataFrame, parameter: str = None, threshold: float = 3.0) -> pd.DataFrame:
    """Per-lot Z-score outlier detection on one parameter's 168h value."""
    out = df.copy()
    col168 = value_col(168, parameter)
    suf = _suffix(parameter)
    mean_col = f"lot_mean_168h{suf}"
    std_col = f"lot_std_168h{suf}"
    z_col = f"zscore_168h{suf}"
    flag_col = f"flag_zscore{suf}"

    lot_stats = out.groupby("lot_id")[col168].agg(["mean", "std"]).reset_index()
    lot_stats.columns = ["lot_id", mean_col, std_col]
    out = out.merge(lot_stats, on="lot_id", how="left")

    safe_std = out[std_col].replace(0, np.nan)
    out[z_col] = (out[col168] - out[mean_col]) / safe_std
    out[z_col] = out[z_col].fillna(0)
    out[flag_col] = out[z_col].abs() >= threshold
    return out


def iqr_flags(df: pd.DataFrame, parameter: str = None, k: float = 2.0, direction: str = "upper") -> pd.DataFrame:
    """Per-lot IQR outlier detection on one parameter's 168h value.
    In burn-in screening, degradation manifests as elevated drift (positive/upper departure).
    direction='upper' flags components exceeding upper IQR limit; direction='two_sided' flags both."""
    out = df.copy()
    col168 = value_col(168, parameter)
    suf = _suffix(parameter)
    lower_col = f"iqr_lower{suf}"
    upper_col = f"iqr_upper{suf}"
    flag_col = f"flag_iqr{suf}"

    def _iqr_bounds(group):
        q1, q3 = group.quantile(0.25), group.quantile(0.75)
        iqr = q3 - q1
        return q1 - k * iqr, q3 + k * iqr

    bounds = out.groupby("lot_id")[col168].apply(_iqr_bounds)
    lower = bounds.apply(lambda x: x[0]).rename(lower_col)
    upper = bounds.apply(lambda x: x[1]).rename(upper_col)
    out = out.merge(lower, on="lot_id", how="left").merge(upper, on="lot_id", how="left")
    if direction == "upper":
        out[flag_col] = out[col168] > out[upper_col]
    else:
        out[flag_col] = (out[col168] < out[lower_col]) | (out[col168] > out[upper_col])
    return out


def isolation_forest_flags(
    df: pd.DataFrame,
    parameters,
    contamination: float = 0.02,
    score_threshold: float = -0.03,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Fits ONE Isolation Forest per lot across ALL checkpoints x ALL
    parameters jointly -- catches cross-parameter multivariate
    anomalies a single-parameter check would miss (e.g. normal
    leakage, abnormal propagation delay).
    """
    out = df.copy()
    out["if_score"] = np.nan
    out["flag_isoforest"] = False

    feature_cols = [value_col(h, p) for p in parameters for h in CHECKPOINTS]

    for lot_id, group in out.groupby("lot_id"):
        if len(group) < 10:
            continue
        X = group[feature_cols].values
        model = IsolationForest(
            n_estimators=200, contamination=contamination, random_state=random_state,
        )
        model.fit(X)
        scores = model.decision_function(X)
        preds = model.predict(X)

        out.loc[group.index, "if_score"] = scores
        out.loc[group.index, "flag_isoforest"] = (preds == -1) & (scores < score_threshold)

    return out


def run_module_a(
    df: pd.DataFrame,
    parameters=None,
    contamination: float = 0.02,
    iqr_k: float = 2.0,
    iqr_direction: str = "upper",
    if_score_threshold: float = -0.03,
) -> pd.DataFrame:
    """
    Full Module A pipeline across every parameter present in `df`.
    parameters=None auto-detects from column names (multi- or
    single-parameter schema, see data.generate_data.detect_parameters).
    """
    if parameters is None:
        parameters = detect_parameters(df)

    out = df.copy()
    for p in parameters:
        out = zscore_flags(out, parameter=p)
        out = iqr_flags(out, parameter=p, k=iqr_k, direction=iqr_direction)

    out = isolation_forest_flags(
        out, parameters, contamination=contamination, score_threshold=if_score_threshold
    )

    # Per-parameter combined flag (z OR iqr for that parameter). Uses
    # _suffix() so legacy single-parameter mode produces exactly
    # "flag_param" (no suffix), matching pre-multi-param behavior where
    # this concept didn't exist as a separate column but flag_module_a
    # captured the same OR logic.
    per_param_flag_cols = []
    for p in parameters:
        label = _param_label(p)
        suf = _suffix(p)
        combined_col = f"flag_param{suf}"
        out[combined_col] = out[f"flag_zscore{suf}"] | out[f"flag_iqr{suf}"]
        per_param_flag_cols.append((label, combined_col))

    # Overall Module A flag = OR across every parameter AND isolation forest
    flag_module_a = out["flag_isoforest"].copy()
    for _, col in per_param_flag_cols:
        flag_module_a = flag_module_a | out[col]
    out["flag_module_a"] = flag_module_a

    # Which parameter(s) triggered -- needed for explainability (Task 4)
    def _triggered(row):
        names = [label for label, col in per_param_flag_cols if row[col]]
        if row["flag_isoforest"] and not names:
            names.append("multivariate (cross-parameter)")
        return ",".join(names) if names else ""

    out["triggered_parameters"] = out.apply(_triggered, axis=1)

    # Combined 0-100 anomaly score: max per-parameter Z magnitude, blended
    # with the isolation forest score (same blend style as before, now
    # taking the WORST parameter rather than a single fixed one).
    z_cols = [f"zscore_168h{_suffix(p)}" for p in parameters]
    max_abs_z = out[z_cols].abs().max(axis=1)
    z_component = max_abs_z.clip(0, 6) / 6
    if_component = (1 - (out["if_score"].fillna(out["if_score"].median()) + 0.5)).clip(0, 1)
    out["anomaly_score"] = (0.5 * z_component + 0.5 * if_component).clip(0, 1) * 100
    out["anomaly_score"] = out["anomaly_score"].round(1)

    return out


if __name__ == "__main__":
    wide_df = pd.read_csv("outputs/burnin_data_wide.csv")
    parameters = detect_parameters(wide_df)
    result = run_module_a(wide_df, parameters=parameters)

    print(f"Parameters detected: {parameters}")
    print(f"Flagged {result['flag_module_a'].sum()} / {len(result)} components")
    print()

    tp = ((result["flag_module_a"]) & (result["is_defective"])).sum()
    fn = ((~result["flag_module_a"]) & (result["is_defective"])).sum()
    fp = ((result["flag_module_a"]) & (~result["is_defective"])).sum()
    tn = ((~result["flag_module_a"]) & (~result["is_defective"])).sum()

    recall = tp / (tp + fn) if (tp + fn) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0

    print(f"True Positives:  {tp}")
    print(f"False Negatives: {fn}  <- the costly failure mode")
    print(f"False Positives: {fp}")
    print(f"True Negatives:  {tn}")
    print(f"Recall (catch rate): {recall:.1%}")
    print(f"Precision: {precision:.1%}")
    print()
    print("Sample triggered_parameters values:")
    print(result[result["flag_module_a"]]["triggered_parameters"].value_counts().head(10))

    result.to_csv("outputs/module_a_results.csv", index=False)
