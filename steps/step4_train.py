"""
Step 4 — Model Training (Improved)

Models trained:
  1. Random Forest (RF)
  2. MLP
  3. Gradient Boosting (GBM)
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
    roc_auc_score, confusion_matrix, classification_report,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    FEATURES_CSV, MODELS_DIR, OUTPUTS_DIR,
    RF_PARAMS, MLP_PARAMS,
    TEST_SIZE, RANDOM_SEED,
    LABEL_NAMES,
)

RESULTS_JSON = OUTPUTS_DIR / "results.json"
CV_FOLDS     = 5
THRESHOLD    = 0.5   

GBM_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "random_state": RANDOM_SEED,
}


# ── Pipeline builders ─────────────────────────────────────────────

def make_rf_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(**RF_PARAMS)),
    ])

def make_mlp_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(**MLP_PARAMS)),
    ])

def make_gbm_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", GradientBoostingClassifier(**GBM_PARAMS)),
    ])


# ── Data loading ──────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(FEATURES_CSV)

    real_df = df[df["filename"] != "synthetic"].copy()
    syn_df  = df[df["filename"] == "synthetic"].copy()

    meta_cols    = {"label", "filename", "side"}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X_real = real_df[feature_cols].values
    y_real = real_df["label"].values

    return real_df, syn_df, feature_cols


# ── Evaluation ────────────────────────────────────────────────────

def evaluate(name, clf, X_test, y_test, X_cv, y_cv, threshold=THRESHOLD):
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)
    cm   = confusion_matrix(y_test, y_pred).tolist()

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_res = cross_validate(
        clf, X_cv, y_cv,
        cv=skf,
        scoring=["f1", "accuracy", "roc_auc"],
        n_jobs=-1,
    )

    print(f"\n── {name} ──")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    return {
        "name": name,
        "cv_f1": cv_res["test_f1"].tolist(),
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "roc_auc": float(auc),
        "confusion_matrix": cm,
        "cv_f1_mean": float(np.mean(cv_res["test_f1"])),
        "cv_f1_std": float(np.std(cv_res["test_f1"])),
    }


# ── Main runner ───────────────────────────────────────────────────

def run():
    real_df, syn_df, feat_cols = load_data()

    idx_train, idx_test = train_test_split(
        np.arange(len(real_df)),
        test_size=TEST_SIZE,
        stratify=real_df["label"].values,
        random_state=RANDOM_SEED,
    )

    train_real = real_df.iloc[idx_train]
    test_real  = real_df.iloc[idx_test]

    train_df = pd.concat([train_real, syn_df], ignore_index=True)

    X_train = train_df[feat_cols].values
    y_train = train_df["label"].values

    X_test = test_real[feat_cols].values
    y_test = test_real["label"].values

    X_real = real_df[feat_cols].values
    y_real = real_df["label"].values


    # ── Models ────────────────────────────────────────────────────

    rf = make_rf_pipeline()
    mlp = make_mlp_pipeline()
    gbm = make_gbm_pipeline()

    voting = VotingClassifier(
        estimators=[
            ("rf", make_rf_pipeline()),
            ("mlp", make_mlp_pipeline()),
        ],
        voting="soft",
        weights=[1, 1.5],  #  MLP is now stronger
    )

    stacking = StackingClassifier(
        estimators=[
            ("rf", make_rf_pipeline()),
            ("mlp", make_mlp_pipeline()),
        ],
        final_estimator=LogisticRegression(),
        cv=5,
        n_jobs=-1,
    )


    # ── Training ──────────────────────────────────────────────────

    rf.fit(X_train, y_train)
    mlp.fit(X_train, y_train)
    gbm.fit(X_train, y_train)
    voting.fit(X_train, y_train)
    stacking.fit(X_train, y_train)


    # ── Evaluation ────────────────────────────────────────────────

    results = [
        evaluate("Random Forest", rf, X_test, y_test, X_real, y_real),
        evaluate("MLP", mlp, X_test, y_test, X_real, y_real),
        evaluate("Gradient Boosting", gbm, X_test, y_test, X_real, y_real),
        evaluate("RF + MLP (Voting)", voting, X_test, y_test, X_real, y_real),
        evaluate("RF + MLP (Stacking)", stacking, X_test, y_test, X_real, y_real),
    ]


    # ── Save ──────────────────────────────────────────────────────

    joblib.dump(rf, MODELS_DIR / "rf.joblib")
    joblib.dump(mlp, MODELS_DIR / "mlp.joblib")
    joblib.dump(gbm, MODELS_DIR / "gbm.joblib")
    joblib.dump(voting, MODELS_DIR / "voting.joblib")
    joblib.dump(stacking, MODELS_DIR / "stacking.joblib")

    with open(RESULTS_JSON, "w") as f:
        json.dump({"models": results}, f, indent=2)

    print("\n✅ Training complete.")


if __name__ == "__main__":
    run()