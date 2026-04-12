"""
Step 3 — Class Balancing (Undersampling, No Duplicates)

Changes:
- ❌ Removed feature-space dedup
- ❌ No synthetic duplication
- ✅ Uses UNDERSAMPLING (reduces majority class)
"""

import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import FEATURES_CSV, RANDOM_SEED, LABEL_GOOD, LABEL_BAD


# ── Class balancing via undersampling ───────────────────────────

def balance_undersample(df):
    # Remove any leftover synthetic rows (safety)
    df = df[df["filename"] != "synthetic"].copy()

    good_df = df[df["label"] == LABEL_GOOD]
    bad_df  = df[df["label"] == LABEL_BAD]

    print(f"  Before: good={len(good_df)}, bad={len(bad_df)}")

    # Identify majority/minority
    if len(good_df) < len(bad_df):
        minority_df = good_df
        majority_df = bad_df
        majority_label = LABEL_BAD
    else:
        minority_df = bad_df
        majority_df = good_df
        majority_label = LABEL_GOOD

    # Undersample majority
    majority_downsampled = majority_df.sample(
        n=len(minority_df),
        random_state=RANDOM_SEED
    )

    # Combine
    balanced = pd.concat([minority_df, majority_downsampled])

    # Shuffle
    balanced = balanced.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(
        f"  After:  good={len(balanced[balanced.label==LABEL_GOOD])}, "
        f"bad={len(balanced[balanced.label==LABEL_BAD])}"
    )

    return balanced


# ── Main ────────────────────────────────────────────────────────

def run():
    if not FEATURES_CSV.exists():
        raise FileNotFoundError("Run Step 2 first — features.csv missing")

    df = pd.read_csv(FEATURES_CSV)

    print(f"  Loaded CSV: {len(df)} rows")

    # Balance dataset
    balanced_df = balance_undersample(df)

    # Save (overwrite safely)
    balanced_df.to_csv(FEATURES_CSV, index=False)

    print("  ✅ Step 3 complete (balanced CSV saved, no duplicates)")


if __name__ == "__main__":
    run()