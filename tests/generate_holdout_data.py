"""
Generates a HELD-OUT test dataset using a different random seed than the
training dataset (seed=42). This is genuinely unseen data — the trained
models in outputs/*.joblib have never encountered these components.

Used to check generalization: does Module A / Module B perform similarly
well on data it wasn't trained on, or did we just overfit to one lucky seed?

Run from project root:
    python3 tests/generate_holdout_data.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate_data import generate_burnin_dataset, to_wide

HOLDOUT_SEED = 7  # deliberately different from the training seed (42)


def main():
    os.makedirs("outputs", exist_ok=True)
    long_df = generate_burnin_dataset(
        n_lots=25, parts_per_lot=40, defect_rate=0.06, seed=HOLDOUT_SEED
    )
    wide_df = to_wide(long_df)
    wide_df.to_csv("outputs/holdout_data_wide.csv", index=False)

    print(f"Held-out test set generated (seed={HOLDOUT_SEED}):")
    print(f"  {len(wide_df)} components across {wide_df['lot_id'].nunique()} lots")
    print(f"  {wide_df['is_defective'].sum()} latent defects "
          f"({wide_df['is_defective'].mean():.1%})")
    print("  Saved to outputs/holdout_data_wide.csv")


if __name__ == "__main__":
    main()
