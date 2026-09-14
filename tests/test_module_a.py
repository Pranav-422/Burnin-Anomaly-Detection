"""
Module A tests. Uses hand-crafted synthetic lots with KNOWN expected
outcomes -- not just the real dataset -- so we're testing the detection
LOGIC itself, independent of any particular dataset's luck.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module_a.anomaly_detection import zscore_flags, iqr_flags, run_module_a


def _make_lot(values_168h, lot_id="LOT_X"):
    """Builds a minimal wide-format dataframe for one lot, given a list
    of 168h values. 0h/24h/96h are filled with plausible flat values so
    Isolation Forest has something reasonable to work with."""
    n = len(values_168h)
    return pd.DataFrame({
        "component_id": [f"C_{i}" for i in range(n)],
        "lot_id": [lot_id] * n,
        "Value_0h": [10.0] * n,
        "Value_24h": [10.2] * n,
        "Value_96h": [v * 0.7 for v in values_168h],
        "Value_168h": values_168h,
        "is_defective": [False] * n,  # ground truth not used by Module A itself
    })


def test_obvious_outlier_is_flagged():
    """A part spiking far above its lot's average must be flagged --
    this is the exact scenario from the problem statement (10uA lot
    average, one part at 45uA)."""
    values = [10.0] * 19 + [45.0]  # 19 normal parts, 1 clear spike
    df = _make_lot(values)
    result = run_module_a(df)
    flagged = result[result["Value_168h"] == 45.0]["flag_module_a"].iloc[0]
    assert flagged, "An obvious 4.5x-lot-average spike was NOT flagged"


def test_uniform_lot_produces_no_false_flags():
    """If every part in a lot behaves identically, nothing should be
    flagged -- there is no peer to be anomalous relative to."""
    values = [10.0 + np.random.default_rng(1).normal(0, 0.05) for _ in range(20)]
    df = _make_lot(values)
    result = run_module_a(df)
    assert result["flag_module_a"].sum() == 0, "False positive(s) on a near-uniform lot"


def test_zscore_flag_direction():
    """A part far below the lot mean should also be flagged (not just
    above) -- anomaly detection must be two-sided."""
    values = [10.0] * 19 + [2.0]  # one part far BELOW the average
    df = _make_lot(values)
    result = zscore_flags(df, threshold=3.0)
    flagged = result[result["Value_168h"] == 2.0]["flag_zscore"].iloc[0]
    assert flagged, "A part far below the lot mean was not flagged (should be two-sided)"


def test_iqr_robust_to_multiple_outliers():
    """IQR-based detection should still catch a spike even when Z-score
    might be distorted by having more than one outlier in the lot."""
    values = [10.0] * 17 + [40.0, 42.0, 45.0]  # three high outliers
    df = _make_lot(values)
    result = iqr_flags(df)
    n_flagged = result[result["Value_168h"] >= 40.0]["flag_iqr"].sum()
    assert n_flagged == 3, f"Expected all 3 outliers caught by IQR, got {n_flagged}"


def test_module_a_returns_required_columns():
    df = _make_lot([10.0] * 15 + [30.0])
    result = run_module_a(df)
    for col in ["flag_module_a", "anomaly_score", "zscore_168h"]:
        assert col in result.columns, f"Missing expected column: {col}"


def test_anomaly_score_is_bounded():
    df = _make_lot([10.0] * 15 + [80.0])
    result = run_module_a(df)
    assert result["anomaly_score"].between(0, 100).all(), "anomaly_score out of [0,100] bounds"
