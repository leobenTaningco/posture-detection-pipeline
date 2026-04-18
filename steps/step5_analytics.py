"""
Step 5 — Analytics & Visualisation (Paper-Friendly Version)

Generates:
  1.  Model comparison bar chart
  2.  Cross-validation F1 box plots
  3.  Confusion matrices (2x2 layout)
  4.  Feature importance charts (RF, MLP proxy)
  5.  ROC curves
  6a-6e. Prediction sample grids x5
  7.  Summary score card
"""

import json
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    FEATURES_CSV, MODELS_DIR, OUTPUTS_DIR,
    GOOD_DIR, BAD_DIR, AUG_DIR,
    LABEL_GOOD, LABEL_BAD, LABEL_NAMES,
    TEST_SIZE, RANDOM_SEED,
    VISIBILITY_THRESHOLD,
)

RESULTS_JSON = OUTPUTS_DIR / "results.json"
MODEL_PATH   = "models/pose_landmarker_heavy.task"

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#E74C3C"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 130,
})


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def load_results():
    with open(RESULTS_JSON) as f:
        return json.load(f)

def load_features():
    df = pd.read_csv(FEATURES_CSV)
    meta_cols = {"label", "filename", "side"}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    return (df[feature_cols].values,
            df["label"].values,
            df["filename"].values,
            feature_cols)

def load_models():
    return {
        "Random Forest": joblib.load(MODELS_DIR / "rf.joblib"),
        "MLP": joblib.load(MODELS_DIR / "mlp.joblib"),
        "RF + MLP (Voting)": joblib.load(MODELS_DIR / "voting.joblib"),
        "RF + MLP (Stacking)": joblib.load(MODELS_DIR / "stacking.joblib"),
    }

def save(fig, name):
    path = OUTPUTS_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.name}")
    return path


# ──────────────────────────────────────────────────────────────
# CHART 1 — Model comparison
# ──────────────────────────────────────────────────────────────

def chart_comparison(results):
    metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
    labels  = ["Accuracy", "F1", "Precision", "Recall", "ROC-AUC"]
    models  = [m for m in results["models"] if m["name"] != "Gradient Boosting"]

    x = np.arange(len(metrics))
    w = 0.18

    fig, ax = plt.subplots(figsize=(8, 4))

    for i, (m, col) in enumerate(zip(models, PALETTE)):
        vals = [m[k] for k in metrics]
        ax.bar(x + i*w - w, vals, w, label=m["name"], color=col)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", linewidth=0.5)

    return save(fig, "01_model_comparison.png")


# ──────────────────────────────────────────────────────────────
# CHART 2 — CV boxplot
# ──────────────────────────────────────────────────────────────

def chart_cv_boxplot(results):
    models = [m for m in results["models"] if m["name"] != "Gradient Boosting"]

    data = [m.get("cv_f1", []) for m in models]    
    names = [m["name"] for m in models]

    fig, ax = plt.subplots(figsize=(7, 4))

    bp = ax.boxplot(data, patch_artist=True)

    for patch, col in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(col)

    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("F1 Score")
    ax.set_title("Cross-Validation F1")

    return save(fig, "02_cv_f1_boxplot.png")


# ──────────────────────────────────────────────────────────────
# CHART 3 — Confusion matrices
# ──────────────────────────────────────────────────────────────

def chart_confusion_matrices(results):
    models = [m for m in results["models"] if m["name"] != "Gradient Boosting"]

    class_names = [LABEL_NAMES[LABEL_BAD], LABEL_NAMES[LABEL_GOOD]]

    fig, axes = plt.subplots(2, 2, figsize=(7, 6))
    axes = axes.flatten()

    for ax, m in zip(axes, models):
        cm = np.array(m["confusion_matrix"])
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-6)

        ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(m["name"], fontsize=10)

        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}",
                        ha="center", va="center", fontsize=9)

    plt.tight_layout()
    return save(fig, "03_confusion_matrices.png")


# ──────────────────────────────────────────────────────────────
# CHART 4 — Feature importance
# ──────────────────────────────────────────────────────────────
def chart_feature_importance(results):
    import matplotlib.pyplot as plt

    plt.rcdefaults()
    plt.style.use("default")

    # LOAD FEATURE IMPORTANCE FROM FILE (THIS IS THE FIX)
    imp_path = OUTPUTS_DIR / "feature_importance.json"

    if not imp_path.exists():
        print("[SKIP] feature_importance.json not found")
        return

    with open(imp_path, "r") as f:
        imp = json.load(f)

    rf_imp = imp.get("rf_feature_importance", {})
    mlp_imp = imp.get("mlp_feature_importance", {})

    if not rf_imp:
        print("[SKIP] RF importance missing")
        return

    names = list(rf_imp.keys())
    rf_vals = np.array([rf_imp[f] for f in names])
    mlp_vals = np.array([mlp_imp.get(f, 0) for f in names])

    order = np.argsort(rf_vals)[::-1]

    names = np.array(names)[order]
    rf_vals = rf_vals[order]
    mlp_vals = mlp_vals[order]

    y = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.barh(y + w/2, rf_vals, height=w, label="RF", color="#4C72B0")
    ax.barh(y - w/2, mlp_vals, height=w, label="MLP", color="#DD8452", alpha=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    ax.legend(fontsize=8)

    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", linewidth=0.5)

    plt.tight_layout()

    return save(fig, "04_feature_importance.png")

# ──────────────────────────────────────────────────────────────
# CHART 5 — ROC curves
# ──────────────────────────────────────────────────────────────

def chart_roc_curves(models_dict, X_test, y_test):
    fig, ax = plt.subplots(figsize=(6, 5))

    for (name, clf), col in zip(models_dict.items(), PALETTE):
        y_prob = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, label=name, color=col)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC Curves")
    ax.legend(fontsize=8)

    return save(fig, "05_roc_curves.png")


# ──────────────────────────────────────────────────────────────
# CHART 7 — Scorecard
# ──────────────────────────────────────────────────────────────

def chart_scorecard(results):
    models = [m for m in results["models"] if m["name"] != "Gradient Boosting"]

    metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
    headers = ["Model", "Acc", "F1", "Prec", "Rec", "AUC"]

    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.axis("off")

    rows = [[m["name"]] + [f"{m[k]:.3f}" for k in metrics] for m in models]

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)

    # FIX: proper scaling to avoid overlap
    table.scale(1.1, 1.6)

    # FIX: cleaner borders + padding
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        if r == 0:
            cell.set_text_props(weight="bold")

    plt.tight_layout()
    return save(fig, "07_scorecard.png")

# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def run():
    if not RESULTS_JSON.exists():
        print("Missing results.json")
        return

    results = load_results()
    X, y, fns, feature_names = load_features()
    models_dict = load_models()

    df = pd.read_csv(FEATURES_CSV)
    real_df = df[df["filename"] != "synthetic"]

    X_real = real_df[feature_names].values
    y_real = real_df["label"].values

    _, X_test, _, y_test = train_test_split(
        X_real, y_real,
        test_size=TEST_SIZE,
        stratify=y_real,
        random_state=RANDOM_SEED
    )

    chart_comparison(results)
    chart_cv_boxplot(results)
    chart_confusion_matrices(results)
    chart_feature_importance(results)
    chart_roc_curves(models_dict, X_test, y_test)
    chart_scorecard(results)

    print("All charts saved.")


if __name__ == "__main__":
    run()