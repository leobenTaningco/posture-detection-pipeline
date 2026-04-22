"""
Step 4 — Model Training (MLP + Stacking with Proper Validation)

Split:
  Train = 80%
  Val   = 10%
  Test  = 10%

- Threshold tuned on validation only
- Test used once for final evaluation
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, precision_recall_curve
)
from sklearn.calibration import CalibratedClassifierCV

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    FEATURES_CSV, MODELS_DIR, OUTPUTS_DIR,
    TEST_SIZE, VAL_SIZE, RANDOM_SEED, MLP_PARAMS
)

RESULTS_JSON = OUTPUTS_DIR / "results_validation.json"


# ---------------------------
# Data
# ---------------------------

def load_data():
    df = pd.read_csv(FEATURES_CSV)

    real_df = df[df["filename"] != "synthetic"].copy()
    syn_df = df[df["filename"] == "synthetic"].copy()

    meta_cols = {"label", "filename", "side"}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    return real_df, syn_df, feature_cols


# ---------------------------
# Pipeline
# ---------------------------

def make_mlp():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(**MLP_PARAMS)),
    ])


# ---------------------------
# Threshold optimization
# ---------------------------

def find_best_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)
    best_idx = np.argmax(f1_scores)
    return thresholds[max(best_idx - 1, 0)]


# ---------------------------
# Evaluation
# ---------------------------

def evaluate(name, clf, X_test, y_test, threshold):
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "name": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }


# ---------------------------
# Main
# ---------------------------

def run():
    real_df, syn_df, feat_cols = load_data()

    # ---------------------------
    # Split: Train / Val / Test
    # ---------------------------
    idx_train, idx_temp = train_test_split(
        np.arange(len(real_df)),
        test_size=TEST_SIZE,
        stratify=real_df["label"].values,
        random_state=RANDOM_SEED
    )

    val_ratio = VAL_SIZE / TEST_SIZE

    idx_val, idx_test = train_test_split(
        idx_temp,
        test_size=(1 - val_ratio),
        stratify=real_df.iloc[idx_temp]["label"].values,
        random_state=RANDOM_SEED
    )

    train_real = real_df.iloc[idx_train]
    val_real   = real_df.iloc[idx_val]
    test_real  = real_df.iloc[idx_test]

    # Downsample synthetic
    syn_df = syn_df.sample(frac=0.5, random_state=RANDOM_SEED)
    train_df = pd.concat([train_real, syn_df], ignore_index=True)

    X_train = train_df[feat_cols].values
    y_train = train_df["label"].values

    X_val = val_real[feat_cols].values
    y_val = val_real["label"].values

    X_test = test_real[feat_cols].values
    y_test = test_real["label"].values

    # ---------------------------
    # Train MLP
    # ---------------------------
    mlp = make_mlp()
    mlp.fit(X_train, y_train)

    mlp_cal = CalibratedClassifierCV(mlp, method="isotonic", cv=3)
    mlp_cal.fit(X_train, y_train)

    # Threshold from validation
    val_prob = mlp_cal.predict_proba(X_val)[:, 1]
    mlp_thr = find_best_threshold(y_val, val_prob)

    # ---------------------------
    # Stacking
    # ---------------------------
    stacking = StackingClassifier(
        estimators=[
            ("mlp1", make_mlp()),
            ("mlp2", make_mlp()),
        ],
        final_estimator=LogisticRegression(),
        cv=5,
        n_jobs=-1,
    )

    stacking.fit(X_train, y_train)

    # ---------------------------
    # Evaluation
    # ---------------------------
    results = [
        evaluate("MLP", mlp_cal, X_test, y_test, mlp_thr),
        evaluate("Stacking", stacking, X_test, y_test, 0.5),
    ]

    # ---------------------------
    # Save
    # ---------------------------
    joblib.dump(mlp_cal, MODELS_DIR / "mlp.joblib")
    joblib.dump(stacking, MODELS_DIR / "stacking.joblib")
    joblib.dump(feat_cols, MODELS_DIR / "feature_names.joblib")

    with open(RESULTS_JSON, "w") as f:
        json.dump({"models": results}, f, indent=2)

    print("Training complete.")


if __name__ == "__main__":
    run()