"""
Step 4 — Model Training

Models trained:
  1. Random Forest (RF)
  2. MLP
  3. Gradient Boosting (GBM)
  4. RF + MLP soft voting ensemble  (kept for comparison)
  5. RF + MLP + GBM soft voting ensemble
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
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

GBM_PARAMS = {
    "n_estimators":  200,
    "max_depth":     4,
    "learning_rate": 0.05,
    "subsample":     0.8,
    "random_state":  RANDOM_SEED,
}


# ── Data loading ──────────────────────────────────────────────────

def load_data():
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"Feature CSV not found: {FEATURES_CSV}\nRun Step 2 first."
        )

    df = pd.read_csv(FEATURES_CSV)
    print(f"  CSV loaded: {len(df)} total rows")

    real_df = df[df["filename"] != "synthetic"].copy()
    syn_df  = df[df["filename"] == "synthetic"].copy()
    print(f"  Real rows: {len(real_df)}  |  Synthetic rows: {len(syn_df)}")

    if real_df.empty:
        raise ValueError(
            "CSV contains 0 real rows (all rows are synthetic).\n"
            "Re-run Step 2 and check skip reasons."
        )

    meta_cols    = {"label", "filename", "side"}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    print(f"  Features ({len(feature_cols)}): {feature_cols}")

    X_real = real_df[feature_cols].values
    y_real = real_df["label"].values

    print(f"\n  Label distribution (real rows):")
    for lbl, name in LABEL_NAMES.items():
        print(f"    {name}: {(y_real == lbl).sum()}")

    if len(np.unique(y_real)) < 2:
        raise ValueError(
            "Only one class found in real rows. "
            "Check both 'goodPosture' and 'badPosture' images were processed."
        )

    return real_df, syn_df, feature_cols


# ── Evaluation ────────────────────────────────────────────────────

def evaluate(name, clf, X_test, y_test, X_cv, y_cv, cv=CV_FOLDS):
    y_pred = clf.predict(X_test)

    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test)[:, 1]
    else:
        raw    = clf.decision_function(X_test)
        y_prob = (raw - raw.min()) / (raw.ptp() + 1e-6)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="binary", zero_division=0)
    prec = precision_score(y_test, y_pred, average="binary", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="binary", zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)
    cm   = confusion_matrix(y_test, y_pred).tolist()
    cr   = classification_report(
        y_test, y_pred,
        target_names=list(LABEL_NAMES.values()),
        output_dict=True, zero_division=0,
    )

    skf    = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    cv_res = cross_validate(
        clf, X_cv, y_cv,
        cv=skf,
        scoring=["f1", "accuracy", "roc_auc"],
        n_jobs=-1,
    )

    print(f"\n  ── {name} ──")
    print(f"    Accuracy : {acc:.4f}")
    print(f"    F1       : {f1:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall   : {rec:.4f}")
    print(f"    ROC-AUC  : {auc:.4f}")
    print(f"    CV F1    : {np.mean(cv_res['test_f1']):.4f} "
        f"± {np.std(cv_res['test_f1']):.4f}")

    return {
        "name":                  name,
        "accuracy":              float(acc),
        "f1":                    float(f1),
        "precision":             float(prec),
        "recall":                float(rec),
        "roc_auc":               float(auc),
        "confusion_matrix":      cm,
        "classification_report": cr,
        "cv_f1":                 cv_res["test_f1"].tolist(),
        "cv_accuracy":           cv_res["test_accuracy"].tolist(),
        "cv_auc":                cv_res["test_roc_auc"].tolist(),
        "cv_f1_mean":            float(np.mean(cv_res["test_f1"])),
        "cv_f1_std":             float(np.std(cv_res["test_f1"])),
    }


# ── Feature importances ───────────────────────────────────────────

def get_tree_importances(pipeline, feature_names):
    clf = pipeline.named_steps["clf"]
    return {n: float(v) for n, v in zip(feature_names, clf.feature_importances_)}

def get_mlp_importance_proxy(pipeline, feature_names):
    mlp    = pipeline.named_steps["clf"]
    scaler = pipeline.named_steps["scaler"]
    W1     = mlp.coefs_[0]
    std    = scaler.scale_
    proxy  = (np.abs(W1) * std[:, np.newaxis]).sum(axis=1)
    proxy /= proxy.sum() + 1e-6
    return {n: float(v) for n, v in zip(feature_names, proxy)}


# ── Pipeline builder ──────────────────────────────────────────────

def make_pipeline(clf):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     clf),
    ])


# ── Main runner ───────────────────────────────────────────────────

def run():
    real_df, syn_df, feat_cols = load_data()

    real_reset = real_df.reset_index(drop=True)
    idx_train, idx_test = train_test_split(
        np.arange(len(real_reset)),
        test_size=TEST_SIZE,
        stratify=real_reset["label"].values,
        random_state=RANDOM_SEED,
    )
    train_real = real_reset.iloc[idx_train]
    test_real  = real_reset.iloc[idx_test]

    train_df = pd.concat([train_real, syn_df], ignore_index=True)
    X_train  = train_df[feat_cols].values
    y_train  = train_df["label"].values
    X_test   = test_real[feat_cols].values
    y_test   = test_real["label"].values
    X_real   = real_reset[feat_cols].values
    y_real   = real_reset["label"].values

    print(f"\n  Train: {len(X_train)} (real {len(train_real)} + "
        f"synthetic {len(syn_df)})  |  Test: {len(X_test)} (real only)")

    # ── Individual pipelines ──────────────────────────────────────
    rf_pipe  = make_pipeline(RandomForestClassifier(**RF_PARAMS))
    mlp_pipe = make_pipeline(MLPClassifier(**MLP_PARAMS))
    gbm_pipe = make_pipeline(GradientBoostingClassifier(**GBM_PARAMS))

    # ── RF + MLP ensemble (comparison) ───────────────────────────
    rf_mlp_ensemble = VotingClassifier(
        estimators=[
            ("rf",  make_pipeline(RandomForestClassifier(**RF_PARAMS))),
            ("mlp", make_pipeline(MLPClassifier(**MLP_PARAMS))),
        ],
        voting="soft", weights=[2, 1],
    )

    # ── RF + MLP + GBM ensemble ───────────────────────────────────
    rf_mlp_gbm_ensemble = VotingClassifier(
        estimators=[
            ("rf",  make_pipeline(RandomForestClassifier(**RF_PARAMS))),
            ("mlp", make_pipeline(MLPClassifier(**MLP_PARAMS))),
            ("gbm", make_pipeline(GradientBoostingClassifier(**GBM_PARAMS))),
        ],
        voting="soft", weights=[2, 1, 2],
    )

    # ── Train ─────────────────────────────────────────────────────
    print("\n  Training Random Forest…")
    rf_pipe.fit(X_train, y_train)

    print("  Training MLP…")
    mlp_pipe.fit(X_train, y_train)

    print("  Training Gradient Boosting…")
    gbm_pipe.fit(X_train, y_train)

    print("  Training RF + MLP Ensemble…")
    rf_mlp_ensemble.fit(X_train, y_train)

    print("  Training RF + MLP + GBM Ensemble…")
    rf_mlp_gbm_ensemble.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────
    rf_metrics         = evaluate("Random Forest",     rf_pipe,             X_test, y_test, X_real, y_real)
    mlp_metrics        = evaluate("MLP",               mlp_pipe,            X_test, y_test, X_real, y_real)
    gbm_metrics        = evaluate("Gradient Boosting", gbm_pipe,            X_test, y_test, X_real, y_real)
    rf_mlp_metrics     = evaluate("RF + MLP",          rf_mlp_ensemble,     X_test, y_test, X_real, y_real)
    rf_mlp_gbm_metrics = evaluate("RF + MLP + GBM",    rf_mlp_gbm_ensemble, X_test, y_test, X_real, y_real)

    rf_metrics["feature_importance"]  = get_tree_importances(rf_pipe,  feat_cols)
    mlp_metrics["feature_importance"] = get_mlp_importance_proxy(mlp_pipe, feat_cols)
    gbm_metrics["feature_importance"] = get_tree_importances(gbm_pipe, feat_cols)

    # ── Save models ───────────────────────────────────────────────
    joblib.dump(rf_pipe,             MODELS_DIR / "rf_pipeline.joblib")
    joblib.dump(mlp_pipe,            MODELS_DIR / "mlp_pipeline.joblib")
    joblib.dump(gbm_pipe,            MODELS_DIR / "gbm_pipeline.joblib")
    joblib.dump(rf_mlp_ensemble,     MODELS_DIR / "rf_mlp_pipeline.joblib")
    joblib.dump(rf_mlp_gbm_ensemble, MODELS_DIR / "rf_mlp_gbm_pipeline.joblib")
    print(f"\n  Models saved → {MODELS_DIR}")

    results = {
        "models": [
            rf_metrics, mlp_metrics, gbm_metrics,
            rf_mlp_metrics, rf_mlp_gbm_metrics,
        ],
        "feature_names": feat_cols,
        "test_size":     TEST_SIZE,
        "train_n":       int(len(X_train)),
        "test_n":        int(len(X_test)),
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Results saved → {RESULTS_JSON}")
    print("\n  ✅ Training complete.")
    return results


if __name__ == "__main__":
    run()
