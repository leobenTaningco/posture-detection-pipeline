"""
Step 4 — Model Training (Improved + Calibrated + Threshold Optimized)

Models trained:
  1. Random Forest (RF)
  2. MLP
  3. Gradient Boosting (GBM) [comparison only]
  4. RF + MLP (Weighted Soft Voting)
  5. RF + MLP (Stacking - MAIN MODEL)
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, precision_recall_curve
)
from sklearn.inspection import permutation_importance
from sklearn.calibration import CalibratedClassifierCV

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    FEATURES_CSV, MODELS_DIR, OUTPUTS_DIR,
    TEST_SIZE, RANDOM_SEED
)

RESULTS_JSON = OUTPUTS_DIR / "results.json"
FEATURE_IMPORTANCE_JSON = OUTPUTS_DIR / "feature_importance.json"

CV_FOLDS = 5


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
# Pipelines
# ---------------------------

def make_rf_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_SEED
        )),
    ])


def make_mlp_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=300,
            random_state=RANDOM_SEED
        )),
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

    idx_train, idx_test = train_test_split(
        np.arange(len(real_df)),
        test_size=TEST_SIZE,
        stratify=real_df["label"].values,
        random_state=RANDOM_SEED
    )

    train_real = real_df.iloc[idx_train]
    test_real = real_df.iloc[idx_test]

    # Downsample synthetic (reduces bias)
    syn_df = syn_df.sample(frac=0.5, random_state=RANDOM_SEED)

    train_df = pd.concat([train_real, syn_df], ignore_index=True)

    X_train = train_df[feat_cols].values
    y_train = train_df["label"].values

    X_test = test_real[feat_cols].values
    y_test = test_real["label"].values

    X_full = real_df[feat_cols].values
    y_full = real_df["label"].values


    # ---------------------------
    # Models
    # ---------------------------

    rf = make_rf_pipeline()
    mlp = make_mlp_pipeline()
    gbm = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=RANDOM_SEED
        )),
    ])

    rf.fit(X_train, y_train)
    mlp.fit(X_train, y_train)
    gbm.fit(X_train, y_train)


    # ---------------------------
    # Probability calibration (IMPORTANT)
    # ---------------------------

    rf_cal = CalibratedClassifierCV(rf, method="isotonic", cv=3)
    mlp_cal = CalibratedClassifierCV(mlp, method="isotonic", cv=3)

    rf_cal.fit(X_train, y_train)
    mlp_cal.fit(X_train, y_train)


    # ---------------------------
    # Threshold tuning (IMPORTANT)
    # ---------------------------

    rf_thr = find_best_threshold(y_full, rf_cal.predict_proba(X_full)[:, 1])
    mlp_thr = find_best_threshold(y_full, mlp_cal.predict_proba(X_full)[:, 1])


    # ---------------------------
    # Ensemble models
    # ---------------------------

    voting = VotingClassifier(
        estimators=[
            ("rf", rf_cal),
            ("mlp", mlp_cal),
        ],
        voting="soft",
        weights=[1, 1.5],
    )

    stacking = StackingClassifier(
        estimators=[
            ("rf", rf_cal),
            ("mlp", mlp_cal),
        ],
        final_estimator=LogisticRegression(),
        cv=5,
        n_jobs=-1,
    )

    voting.fit(X_train, y_train)
    stacking.fit(X_train, y_train)


    # ---------------------------
    # Feature importance
    # ---------------------------

    rf_imp = dict(zip(
        feat_cols,
        rf.named_steps["clf"].feature_importances_
    ))

    mlp_perm = permutation_importance(
        mlp,
        X_full,
        y_full,
        n_repeats=10,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    mlp_imp = dict(zip(feat_cols, mlp_perm.importances_mean))

    with open(FEATURE_IMPORTANCE_JSON, "w") as f:
        json.dump({
            "rf_feature_importance": rf_imp,
            "mlp_feature_importance": mlp_imp,
            "feature_names": feat_cols
        }, f, indent=2)


    # ---------------------------
    # Evaluation
    # ---------------------------

    results = [
        evaluate("Random Forest", rf_cal, X_test, y_test, rf_thr),
        evaluate("MLP", mlp_cal, X_test, y_test, mlp_thr),
        evaluate("Gradient Boosting", gbm, X_test, y_test, 0.5),
        evaluate("RF + MLP (Voting)", voting, X_test, y_test, 0.5),
        evaluate("RF + MLP (Stacking)", stacking, X_test, y_test, 0.5),
    ]


    # ---------------------------
    # Save models
    # ---------------------------

    joblib.dump(rf_cal, MODELS_DIR / "rf.joblib")
    joblib.dump(mlp_cal, MODELS_DIR / "mlp.joblib")
    joblib.dump(gbm, MODELS_DIR / "gbm.joblib")
    joblib.dump(voting, MODELS_DIR / "voting.joblib")
    joblib.dump(stacking, MODELS_DIR / "stacking.joblib")

    joblib.dump(feat_cols, MODELS_DIR / "feature_names.joblib")


    # ---------------------------
    # Save results
    # ---------------------------

    with open(RESULTS_JSON, "w") as f:
        json.dump({"models": results}, f, indent=2)

    print("Training complete.")


if __name__ == "__main__":
    run()