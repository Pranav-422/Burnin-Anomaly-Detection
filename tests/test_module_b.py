"""
Module B tests. Uses hand-crafted early-reading pairs with known
DIRECTIONAL expectations (e.g. faster early drift => higher predicted
168h), independent of any specific trained model's exact numbers.
Runs against EVERY parameter's own trained model (leakage, Iddq, prop delay).
"""

import os
import sys

import joblib
import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate_data import ALL_PARAMETERS, value_col
from module_b.drift_predictor import (
    feature_cols_for, engineer_features, predict_and_flag, compute_safety_slope,
)


def _model_path(parameter):
    return f"outputs/xgb_model_{parameter}.joblib"


@pytest.fixture(scope="module", params=ALL_PARAMETERS)
def parameter_and_model(request):
    parameter = request.param
    path = _model_path(parameter)
    if not os.path.exists(path):
        pytest.skip(f"Trained model not found for {parameter} -- run run_pipeline.py first")
    return parameter, joblib.load(path)


def _make_component(parameter, v0, v24, lot_id="LOT_000"):
    return pd.DataFrame([{
        "component_id": "C_TEST", "lot_id": lot_id,
        value_col(0, parameter): v0, value_col(24, parameter): v24,
    }])


def test_fast_drift_predicts_higher_168h_than_slow_drift(parameter_and_model):
    """A component drifting fast in the first 24h should be predicted to
    end up higher at 168h than one drifting slowly, all else equal."""
    parameter, model = parameter_and_model
    baseline = 10.0
    slow = _make_component(parameter, baseline, baseline * 1.03)
    fast = _make_component(parameter, baseline, baseline * 1.8)

    cols = feature_cols_for(parameter)
    suf = f"_{parameter}"
    slow_feat = engineer_features(slow, parameter=parameter)
    fast_feat = engineer_features(fast, parameter=parameter)
    slow_feat[f"lot_mean_0h{suf}"] = baseline
    fast_feat[f"lot_mean_0h{suf}"] = baseline

    pred_slow = model.predict(slow_feat[cols])[0]
    pred_fast = model.predict(fast_feat[cols])[0]

    assert pred_fast > pred_slow, (
        f"[{parameter}] Fast-drifting component predicted LOWER 168h ({pred_fast:.2f}) than "
        f"slow-drifting one ({pred_slow:.2f}) -- model direction is wrong"
    )


def test_flat_reading_predicts_near_baseline(parameter_and_model):
    """A component with zero drift from 0h to 24h should be predicted to
    stay close to its starting value at 168h, not swing wildly."""
    parameter, model = parameter_and_model
    baseline = 10.0
    flat = _make_component(parameter, baseline, baseline)
    cols = feature_cols_for(parameter)
    suf = f"_{parameter}"
    feat = engineer_features(flat, parameter=parameter)
    feat[f"lot_mean_0h{suf}"] = baseline
    pred = model.predict(feat[cols])[0]
    assert 3.0 < pred < 25.0, f"[{parameter}] Flat-reading prediction ({pred:.2f}) is implausibly far from baseline"


@pytest.mark.parametrize("parameter", ALL_PARAMETERS)
def test_safety_slope_threshold_is_positive_and_small(parameter):
    df = pd.read_csv("outputs/burnin_data_wide.csv")
    threshold = compute_safety_slope(df, parameter=parameter, percentile=95.0)
    assert threshold > 0, f"[{parameter}] Safety slope threshold should be positive"
    assert threshold < 1.0, f"[{parameter}] Safety slope threshold ({threshold}) implausibly large"


def test_predict_and_flag_adds_expected_columns(parameter_and_model):
    parameter, model = parameter_and_model
    df = pd.read_csv("outputs/burnin_data_wide.csv").head(20)
    full_df = pd.read_csv("outputs/burnin_data_wide.csv")
    threshold = compute_safety_slope(full_df, parameter=parameter, percentile=95.0)
    result = predict_and_flag(df, model, threshold, parameter=parameter)
    suf = f"_{parameter}"
    for col in [f"predicted_168h{suf}", f"predicted_slope{suf}", f"flag_param_b{suf}"]:
        assert col in result.columns, f"[{parameter}] Missing expected column: {col}"
    assert result[f"predicted_168h{suf}"].notna().all(), f"[{parameter}] NaN predictions found"


def test_legacy_single_parameter_mode_uses_unsuffixed_columns():
    """Backward-compat: parameter=None must produce EXACTLY the original
    unsuffixed column names (predicted_168h, not predicted_168h_value),
    so any code written against the single-parameter schema still works."""
    from data.generate_data import generate_burnin_dataset, to_wide
    long_df = generate_burnin_dataset(parameters=["leakage_ua"], n_lots=3, parts_per_lot=15)
    wide_df = to_wide(long_df, parameters=["leakage_ua"])
    cols = feature_cols_for(None)
    assert cols == ["Value_0h", "Value_24h", "delta_24h", "pct_change_24h", "slope_0_24", "lot_mean_0h"]
