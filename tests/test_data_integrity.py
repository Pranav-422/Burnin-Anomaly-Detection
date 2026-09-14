"""
Data integrity tests -- including the exact leakage check that would catch
Module B accidentally using Value_96h / Value_168h as an input feature,
now generalized across every screened parameter (leakage, Iddq, prop delay).
"""

import os
import sys

import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate_data import ALL_PARAMETERS, value_col
from module_b.drift_predictor import feature_cols_for, engineer_features


@pytest.mark.parametrize("parameter", ALL_PARAMETERS + [None])
def test_feature_cols_excludes_future_readings(parameter):
    """The exact check for the leakage concern, run for EVERY parameter
    (and the legacy None case): Module B must never see the 96h or 168h
    reading as an input feature -- only 0h/24h and things derived from them."""
    cols = feature_cols_for(parameter)
    assert value_col(96, parameter) not in cols, f"LEAKAGE ({parameter}): 96h reading used as a model input"
    assert value_col(168, parameter) not in cols, f"LEAKAGE ({parameter}): 168h reading (the target!) used as a model input"


@pytest.mark.parametrize("parameter", ALL_PARAMETERS + [None])
def test_feature_cols_only_uses_early_readings(parameter):
    """Every feature must be derivable from 0h/24h alone (plus lot context)."""
    suf = "" if parameter is None else f"_{parameter}"
    allowed = {
        value_col(0, parameter), value_col(24, parameter),
        f"delta_24h{suf}", f"pct_change_24h{suf}", f"slope_0_24{suf}", f"lot_mean_0h{suf}",
    }
    assert set(feature_cols_for(parameter)) == allowed, f"Unexpected feature set for {parameter}: {feature_cols_for(parameter)}"


@pytest.mark.parametrize("parameter", ALL_PARAMETERS + [None])
def test_engineered_features_dont_touch_late_readings(parameter):
    """engineer_features() must not read the 96h/168h reading even if
    present in the input dataframe (e.g. from the training CSV) --
    poisoned with an extreme value to prove it's never touched."""
    c0, c24, c96, c168 = value_col(0, parameter), value_col(24, parameter), value_col(96, parameter), value_col(168, parameter)
    row = {"component_id": "C_TEST", "lot_id": "LOT_TEST", c0: 10.0, c24: 12.0, c96: 999.0, c168: 999.0}
    df = pd.DataFrame([row])
    out = engineer_features(df, parameter=parameter)
    suf = "" if parameter is None else f"_{parameter}"
    assert out[f"delta_24h{suf}"].iloc[0] == pytest.approx(2.0)
    assert out[f"slope_0_24{suf}"].iloc[0] == pytest.approx(2.0 / 24)


def test_no_missing_values_in_generated_dataset():
    wide = pd.read_csv("outputs/burnin_data_wide.csv")
    required = ["is_defective", "lot_id"] + [
        value_col(h, p) for p in ALL_PARAMETERS for h in (0, 24, 96, 168)
    ]
    for col in required:
        assert wide[col].isna().sum() == 0, f"NaNs found in {col}"


def test_dataset_defect_rate_is_plausible():
    """Sanity check the seeded defect rate hasn't drifted to something absurd."""
    wide = pd.read_csv("outputs/burnin_data_wide.csv")
    rate = wide["is_defective"].mean()
    assert 0.02 <= rate <= 0.15, f"Defect rate {rate:.1%} outside expected 2-15% range"


def test_same_defect_ground_truth_across_parameters():
    """A component's is_defective must be identical no matter which
    parameter's rows you look at -- the task sheet requires ONE defect
    draw per component, correlated across parameters, not independent
    per-parameter defects."""
    long_df = pd.read_csv("outputs/burnin_data_long.csv")
    per_component_unique = long_df.groupby("component_id")["is_defective"].nunique()
    assert (per_component_unique == 1).all(), "Found a component with inconsistent is_defective across parameters/rows"
