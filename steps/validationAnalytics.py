"""
Step 5 — Analytics (MLP + Stacking Only)
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    FEATURES_CSV, MODELS_DIR, OUTPUTS_DIR,
    TEST_SIZE, RANDOM_SEED
)

RESULTS_JSON = OUTPUTS_DIR / "results.json"


# ---------------------------
# Load
# ---------------------------

def load_data(feat_cols):
    df = pd.read_csv(FEATURES_CSV)
    real_df = df[df["filename"] != "synthetic"]

    X = real_df[feat_cols].values
    y = real_df["label"].values

    return X, y


def save(fig, name):
    path = OUTPUTS_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", name)


# ---------------------------
# Charts
# ---------------------------

def chart_comparison(results):
    names = [m["name"] for m in results["models"]]
    f1 = [m["f1"] for m in results["models"]]

    fig, ax = plt.subplots()
    ax.bar(names, f1)
    ax.set_ylabel("F1 Score")
    ax.set_title("Model Comparison")

    save(fig, "comparison.png")


def chart_confusion(models, X_test, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        ax.imshow(cm, cmap="Blues")
        ax.set_title(name)

        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center")

    save(fig, "confusion.png")


def chart_roc(models, X_test, y_test):
    fig, ax = plt.subplots()

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.legend()
    ax.set_title("ROC Curve")

    save(fig, "roc.png")


def chart_score(results):
    fig, ax = plt.subplots()
    ax.axis("off")

    rows = []
    for m in results["models"]:
        rows.append([
            m["name"],
            f"{m['accuracy']:.3f}",
            f"{m['f1']:.3f}",
            f"{m['precision']:.3f}",
            f"{m['recall']:.3f}"
        ])

    table = ax.table(
        cellText=rows,
        colLabels=["Model", "Acc", "F1", "Prec", "Rec"],
        loc="center"
    )

    table.scale(1, 2)
    save(fig, "score_table.png")


# ---------------------------
# Main
# ---------------------------

def run():
    with open(RESULTS_JSON) as f:
        results = json.load(f)

    feat_cols = joblib.load(MODELS_DIR / "feature_names.joblib")

    X, y = load_data(feat_cols)

    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED
    )

    models = {
        "MLP": joblib.load(MODELS_DIR / "mlp.joblib"),
        "Stacking": joblib.load(MODELS_DIR / "stacking.joblib"),
    }

    chart_comparison(results)
    chart_confusion(models, X_test, y_test)
    chart_roc(models, X_test, y_test)
    chart_score(results)

    print("Analytics complete.")


if __name__ == "__main__":
    run()