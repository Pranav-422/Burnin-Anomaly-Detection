"""
Module B: Time-series drift predictor -- multi-parameter.

Trains ONE regression model PER PARAMETER (option (a) from the task
sheet -- simpler, and each parameter gets its own physically-meaningful
safety-slope threshold since leakage/Iddq/propagation-delay are on
different scales and units).

Each parameter's model takes that SAME parameter's 0h/24h readings and
forecasts its own 168h value. If the predicted drift slope for ANY
parameter exceeds that parameter's own safety-slope threshold, the
component is flagged (OR across parameters, matching Module A's
recall-biased policy).

BUG FIX (per task sheet): MAE is now reported strictly on the held-out
TEST split, never by re-predicting on the full dataset the model was
trained on (which silently included training rows and inflated the
apparent accuracy in the previous single-parameter version).

Backward-compatible: pass parameter=None for the legacy unsuffixed
Value_0h/24h/96h/168h schema.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from data.generate_data import value_col, detect_parameters


def _param_label(p):
    return p if p is not None else "value"


def _suffix(p):
    """Empty string for the legacy (None) parameter -- so column names
    are exactly delta_24h, pct_change_24h, etc. with no suffix, matching
    the original single-parameter module's names precisely."""
    return "" if p is None else f"_{p}"


def feature_cols_for(parameter: str = None):
    """Column names of the engineered feature set for one parameter."""
    suf = _suffix(parameter)
    return [
        value_col(0, parameter), value_col(24, parameter),
        f"delta_24h{suf}", f"pct_change_24h{suf}",
        f"slope_0_24{suf}", f"lot_mean_0h{suf}",
    ]


# Backward-compat alias: the single-parameter FEATURE_COLS list, exactly
# as in the original module, for any code that still imports it directly.
FEATURE_COLS = feature_cols_for(None)


def engineer_features(df: pd.DataFrame, parameter: str = None) -> pd.DataFrame:
    """Derive drift-behavior features for ONE parameter from its 0h/24h readings."""
    out = df.copy()
    suf = _suffix(parameter)
    c0, c24 = value_col(0, parameter), value_col(24, parameter)

    delta_col = f"delta_24h{suf}"
    pct_col = f"pct_change_24h{suf}"
    slope_col = f"slope_0_24{suf}"
    lot_mean_col = f"lot_mean_0h{suf}"

    out[delta_col] = out[c24] - out[c0]
    out[pct_col] = out[delta_col] / out[c0].replace(0, np.nan)
    out[pct_col] = out[pct_col].fillna(0)
    out[slope_col] = out[delta_col] / 24.0
    out[lot_mean_col] = out.groupby("lot_id")[c0].transform("mean")
    return out


def train_models(df: pd.DataFrame, parameter: str = None, random_state: int = 42):
    """
    Trains Linear Regression + XGBoost for ONE parameter.

    Returns (lin_model, xgb_model, report) where report's MAE values are
    computed ONLY on the held-out test split -- never on the full
    dataset (the bug this task sheet asked to fix).
    """
    label = _param_label(parameter)
    df = engineer_features(df, parameter=parameter)
    cols = feature_cols_for(parameter)
    X = df[cols]
    y = df[value_col(168, parameter)]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    lin_test_mae = mean_absolute_error(y_test, lin_model.predict(X_test))

    xgb_model = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=random_state,
    )
    xgb_model.fit(X_train, y_train)
    xgb_test_mae = mean_absolute_error(y_test, xgb_model.predict(X_test))

    report = {
        "parameter": label,
        "linear_regression_test_mae": round(lin_test_mae, 4),
        "xgboost_test_mae": round(xgb_test_mae, 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "linear_coefficients": dict(zip(cols, lin_model.coef_.round(4))),
    }
    return lin_model, xgb_model, report


def compute_safety_slope(df: pd.DataFrame, parameter: str = None, percentile: float = 99.8) -> float:
    """
    Derives the safety-slope threshold for ONE parameter, empirically,
    from the KNOWN-GOOD population's actual (24h->168h) drift slope.
    Each parameter gets its OWN threshold since units/scales differ.
    Defaults to 99.8th percentile (~3.1-sigma normal variation limit).
    """
    c24, c168 = value_col(24, parameter), value_col(168, parameter)
    good = df[~df["is_defective"]] if "is_defective" in df.columns else df
    actual_slope = (good[c168] - good[c24]) / (168 - 24)
    threshold = np.percentile(actual_slope, percentile)
    return round(float(threshold), 6)


def predict_and_flag(df: pd.DataFrame, model, safety_slope_threshold: float, parameter: str = None) -> pd.DataFrame:
    """Runs predictions for ONE parameter across every component and
    applies that parameter's safety-slope early-rejection rule.
    Column names use _suffix() (empty for legacy None) so single-
    parameter mode produces exactly predicted_168h / predicted_slope /
    flag_param_b -- matching the original module's column names."""
    suf = _suffix(parameter)
    df = engineer_features(df, parameter=parameter)
    cols = feature_cols_for(parameter)
    c24 = value_col(24, parameter)

    pred_col = f"predicted_168h{suf}"
    slope_col = f"predicted_slope{suf}"
    flag_col = f"flag_param_b{suf}"

    df[pred_col] = model.predict(df[cols])
    df[slope_col] = (df[pred_col] - df[c24]) / (168 - 24)
    df[flag_col] = df[slope_col] > safety_slope_threshold
    return df


def train_and_predict_all(df: pd.DataFrame, parameters=None, percentile: float = 99.8, random_state: int = 42):
    """
    Orchestrates Module B across every parameter present in `df`.
    Returns (result_df, reports_dict, models_dict) where result_df has
    per-parameter predicted_168h_*, predicted_slope_*, flag_param_b_*
    columns plus a combined flag_module_b and triggered_parameters_b.

    BACKWARD-COMPAT: with a single legacy (None) parameter, the output
    columns are exactly predicted_168h / predicted_slope / flag_module_b
    -- no suffix -- matching the original single-parameter module.
    """
    if parameters is None:
        parameters = detect_parameters(df)

    result = df.copy()
    reports = {}
    models = {}

    for p in parameters:
        label = _param_label(p)
        suf = _suffix(p)
        lin_model, xgb_model, report = train_models(df, parameter=p, random_state=random_state)
        threshold = compute_safety_slope(df, parameter=p, percentile=percentile)
        report["safety_slope_threshold"] = threshold
        reports[label] = report
        models[label] = {"linear": lin_model, "xgb": xgb_model, "safety_slope": threshold}

        predicted = predict_and_flag(df, xgb_model, threshold, parameter=p)
        keep_cols = [f"predicted_168h{suf}", f"predicted_slope{suf}", f"flag_param_b{suf}"]
        result = result.merge(
            predicted[["component_id"] + keep_cols], on="component_id", how="left",
        )

    flag_cols = [f"flag_param_b{_suffix(p)}" for p in parameters]
    flag_module_b = result[flag_cols[0]].copy()
    for c in flag_cols[1:]:
        flag_module_b = flag_module_b | result[c]
    result["flag_module_b"] = flag_module_b

    def _triggered(row):
        names = [_param_label(p) for p in parameters if row[f"flag_param_b{_suffix(p)}"]]
        return ",".join(names)

    result["triggered_parameters_b"] = result.apply(_triggered, axis=1)

    return result, reports, models


if __name__ == "__main__":
    wide_df = pd.read_csv("outputs/burnin_data_wide.csv")
    parameters = detect_parameters(wide_df)

    result, reports, models = train_and_predict_all(wide_df, parameters=parameters)

    print(f"Parameters detected: {parameters}\n")
    for label, report in reports.items():
        print(f"[{label}] Linear Regression test MAE: {report['linear_regression_test_mae']}")
        print(f"[{label}] XGBoost test MAE:           {report['xgboost_test_mae']}  "
              f"(n_test={report['n_test']})")
        print(f"[{label}] Safety slope threshold (95th pct known-good): "
              f"{report['safety_slope_threshold']}")
        print()

    tp = ((result["flag_module_b"]) & (result["is_defective"])).sum()
    fn = ((~result["flag_module_b"]) & (result["is_defective"])).sum()
    fp = ((result["flag_module_b"]) & (~result["is_defective"])).sum()
    tn = ((~result["flag_module_b"]) & (~result["is_defective"])).sum()
    recall = tp / (tp + fn) if (tp + fn) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0

    print(f"Combined Module B: flagged {result['flag_module_b'].sum()} / {len(result)}")
    print(f"True Positives:  {tp}")
    print(f"False Negatives: {fn}")
    print(f"False Positives: {fp}")
    print(f"True Negatives:  {tn}")
    print(f"Recall: {recall:.1%}  Precision: {precision:.1%}")

    result.to_csv("outputs/module_b_results.csv", index=False)
