"""
Generalization test -- multi-parameter.

Runs the ALREADY-TRAINED models (one per parameter: leakage_ua, iddq_ua,
prop_delay_ns -- from outputs/xgb_model_<parameter>.joblib, trained on
seed=42 data) against a genuinely held-out dataset (seed=7, generated
separately in tests/generate_holdout_data.py).

This is the test that actually answers "does this work on data the models
have never seen, or did we just get lucky on one seed?" -- for every
screened parameter, not just leakage.
"""

import os
import sys

import joblib
import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate_data import ALL_PARAMETERS, PARAMETERS, detect_parameters
from module_a.anomaly_detection import run_module_a
from module_b.drift_predictor import compute_safety_slope, predict_and_flag, _param_label
from sklearn.metrics import mean_absolute_error

HOLDOUT_PATH = "outputs/holdout_data_wide.csv"
TRAIN_PATH = "outputs/burnin_data_wide.csv"


@pytest.fixture(scope="module")
def holdout_df():
    if not os.path.exists(HOLDOUT_PATH):
        pytest.skip("Holdout dataset not found -- run tests/generate_holdout_data.py first")
    return pd.read_csv(HOLDOUT_PATH)


@pytest.fixture(scope="module")
def train_df():
    if not os.path.exists(TRAIN_PATH):
        pytest.skip("Training dataset not found -- run run_pipeline.py first")
    return pd.read_csv(TRAIN_PATH)


@pytest.fixture(scope="module", params=ALL_PARAMETERS)
def parameter_and_model(request):
    """One entry per screened parameter, each with its own trained
    XGBoost model (outputs/xgb_model_<parameter>.joblib)."""
    parameter = request.param
    path = f"outputs/xgb_model_{parameter}.joblib"
    if not os.path.exists(path):
        pytest.skip(f"Trained model not found for {parameter} -- run run_pipeline.py first")
    return parameter, joblib.load(path)


def test_holdout_is_genuinely_different_from_training_data(holdout_df, train_df):
    """Sanity check the holdout set isn't accidentally identical to the
    training set (would defeat the whole point of holding it out).
    Checked across EVERY screened parameter's 168h value, since it would
    be possible for one parameter's values to coincide by chance even if
    another genuinely differs."""
    merged = holdout_df.merge(train_df, on="component_id", suffixes=("_holdout", "_train"))
    for p in ALL_PARAMETERS:
        col = f"Value_168h_{p}"
        identical_rows = (merged[f"{col}_holdout"] == merged[f"{col}_train"]).sum()
        assert identical_rows < len(merged) * 0.5, (
            f"[{p}] Holdout and training data look suspiciously identical -- "
            "holdout may not be a genuinely different seed"
        )


def test_module_a_recall_on_holdout(holdout_df):
    """Module A's recall should stay high on unseen data -- it's
    unsupervised (per-lot statistics), so it should generalize by
    construction, but this confirms it in practice, across every
    parameter and the cross-parameter Isolation Forest jointly."""
    result = run_module_a(holdout_df)
    tp = ((result["flag_module_a"]) & (result["is_defective"])).sum()
    fn = ((~result["flag_module_a"]) & (result["is_defective"])).sum()
    recall = tp / (tp + fn) if (tp + fn) else 0
    print(f"\nModule A on HOLDOUT set: recall={recall:.1%}, TP={tp}, FN={fn}")
    assert recall >= 0.90, f"Module A recall on holdout dropped to {recall:.1%} (expected >=90%)"


def test_module_b_mae_on_holdout(holdout_df, train_df, parameter_and_model):
    """EACH parameter's trained model (fit on seed=42 data) is applied to
    the seed=7 holdout set. MAE should stay reasonably low, relative to
    that parameter's own scale, if the model generalizes rather than
    having memorized the training set. Threshold is derived as 6% of the
    parameter's max_limit -- the same ratio the original single-parameter
    test used (3.0 uA / 50.0 uA max_limit for leakage), just generalized
    to Iddq and propagation delay's own units/scales instead of reusing
    a flat absolute number that wouldn't mean the same thing across
    different units."""
    parameter, model = parameter_and_model
    threshold_slope = compute_safety_slope(train_df, parameter=parameter, percentile=99.8)
    result = predict_and_flag(holdout_df, model, threshold_slope, parameter=parameter)
    suf = f"_{parameter}"
    mae = mean_absolute_error(result[f"Value_168h{suf}"], result[f"predicted_168h{suf}"])
    mae_limit = 0.06 * PARAMETERS[parameter]["max_limit"]
    print(f"\nModule B [{parameter}] (trained on seed=42) MAE on HOLDOUT (seed=7): {mae:.3f}")
    assert mae < mae_limit, (
        f"[{parameter}] MAE on holdout ({mae:.3f}) exceeds {mae_limit:.3f} "
        "-- suggests overfitting to training seed"
    )


def test_combined_recall_on_holdout(holdout_df, train_df):
    """The full combined system (A OR B, across every parameter) on
    genuinely unseen data."""
    parameters = detect_parameters(holdout_df)
    mod_a = run_module_a(holdout_df, parameters=parameters)

    combined_flag = mod_a["flag_module_a"].copy()
    for p in parameters:
        label = _param_label(p)
        model = joblib.load(f"outputs/xgb_model_{label}.joblib")
        threshold = compute_safety_slope(train_df, parameter=p, percentile=99.8)
        pred = predict_and_flag(holdout_df, model, threshold, parameter=p)
        suf = "" if p is None else f"_{p}"
        combined_flag = combined_flag | pred[f"flag_param_b{suf}"]

    tp = (combined_flag & holdout_df["is_defective"]).sum()
    fn = (~combined_flag & holdout_df["is_defective"]).sum()
    fp = (combined_flag & ~holdout_df["is_defective"]).sum()
    recall = tp / (tp + fn) if (tp + fn) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    print(f"\nCombined (A OR B, all parameters) on HOLDOUT: recall={recall:.1%}, "
          f"precision={precision:.1%}, TP={tp}, FN={fn}, FP={fp}")
    assert recall >= 0.90, f"Combined recall on holdout dropped to {recall:.1%} (expected >=90%)"
    assert precision >= 0.70, f"Combined precision on holdout is too low ({precision:.1%}, expected >=70%)"
