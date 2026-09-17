"""
End-to-end pipeline: generate multi-parameter data -> Module A ->
Module B -> explainability -> final QA report.

Screens THREE parameters per component: leakage current, Iddq
(standby current), and propagation delay -- per the problem
statement's "e.g. standby current Iddq, leakage currents, or
propagation delays."

Run this from the project root:
    python3 run_pipeline.py
"""

import os
import joblib
import pandas as pd

from data.generate_data import generate_burnin_dataset, to_wide, ALL_PARAMETERS, detect_parameters
from module_a.anomaly_detection import run_module_a
from module_b.drift_predictor import train_and_predict_all
from explainability.shap_utils import build_full_report

OUT_DIR = "outputs"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Generating physics-informed synthetic burn-in data")
    print(f"        Parameters: {', '.join(ALL_PARAMETERS)}")
    print("=" * 60)
    long_df = generate_burnin_dataset()
    wide_df = to_wide(long_df)
    wide_df.to_csv(f"{OUT_DIR}/burnin_data_wide.csv", index=False)
    long_df.to_csv(f"{OUT_DIR}/burnin_data_long.csv", index=False)
    print(f"{len(wide_df)} components across {wide_df['lot_id'].nunique()} lots "
          f"({wide_df['is_defective'].sum()} latent defects, "
          f"{wide_df['is_defective'].mean():.1%})\n")

    parameters = detect_parameters(wide_df)

    print("=" * 60)
    print("STEP 2: Module A - dynamic lot-relative anomaly detection")
    print("        (per-parameter Z-score/IQR + cross-parameter Isolation Forest)")
    print("=" * 60)
    module_a_df = run_module_a(wide_df, parameters=parameters)
    module_a_df.to_csv(f"{OUT_DIR}/module_a_results.csv", index=False)
    _print_confusion(module_a_df, "flag_module_a", "Module A")

    print("=" * 60)
    print("STEP 3: Module B - time-series drift predictor (per parameter)")
    print("=" * 60)
    module_b_df, reports, models = train_and_predict_all(wide_df, parameters=parameters)
    module_b_df.to_csv(f"{OUT_DIR}/module_b_results.csv", index=False)

    xgb_models = {}
    for label, m in models.items():
        joblib.dump(m["xgb"], f"{OUT_DIR}/xgb_model_{label}.joblib")
        joblib.dump(m["linear"], f"{OUT_DIR}/linear_model_{label}.joblib")
        xgb_models[label] = m["xgb"]

        report = reports[label]
        print(f"[{label}] Linear Regression TEST-SET MAE: {report['linear_regression_test_mae']}  "
              f"(n_test={report['n_test']})")
        print(f"[{label}] XGBoost TEST-SET MAE:           {report['xgboost_test_mae']}  "
              f"(n_test={report['n_test']})")
        print(f"[{label}] Safety slope threshold (99.8th pct known-good): "
              f"{report['safety_slope_threshold']}")
        print()

    print("NOTE: MAE above is reported on the held-out TEST SPLIT ONLY -- "
          "not by re-predicting on the full dataset (that was a bug in the "
          "single-parameter version; fixed here per the task sheet).\n")

    _print_confusion(module_b_df, "flag_module_b", "Module B")

    print("=" * 60)
    print("STEP 4: Explainability layer + combined QA report")
    print("=" * 60)
    final_report = build_full_report(module_a_df, module_b_df, xgb_models, parameters=parameters)
    final_report.to_csv(f"{OUT_DIR}/final_report.csv", index=False)
    _print_confusion(final_report, "final_flag", "Combined (A OR B)")

    print("\nSample flagged components with explanations:")
    print("-" * 60)
    for _, row in final_report[final_report["final_flag"]].head(4).iterrows():
        print(f"\n{row['component_id']}  |  lot: {row['lot_id']}  |  "
              f"anomaly_score: {row['anomaly_score']}  |  "
              f"triggered: {row['triggered_parameters_all']}")
        print(f"  {row['explanation']}")

    print(f"\nAll outputs saved to ./{OUT_DIR}/")
    print("Pipeline complete.")


def _print_confusion(df: pd.DataFrame, flag_col: str, label: str):
    tp = ((df[flag_col]) & (df["is_defective"])).sum()
    fn = ((~df[flag_col]) & (df["is_defective"])).sum()
    fp = ((df[flag_col]) & (~df["is_defective"])).sum()
    tn = ((~df[flag_col]) & (~df["is_defective"])).sum()
    recall = tp / (tp + fn) if (tp + fn) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    print(f"[{label}] Flagged: {df[flag_col].sum()}/{len(df)}  |  "
          f"Recall: {recall:.1%}  |  Precision: {precision:.1%}  |  "
          f"False Negatives: {fn}\n")


if __name__ == "__main__":
    main()
