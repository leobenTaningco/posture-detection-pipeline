"""
Step 5 — Analytics & Visualisation

Generates:
  1.  Model comparison bar chart (all 5 models)
  2.  Cross-validation F1 box plots
  3.  Confusion matrices (5 side-by-side)
  4.  Feature importance charts (RF, MLP proxy, GBM)
  5.  ROC curves (all 5 models)
  6a-6e. Prediction sample grids x5 (different random samples each)
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

# 5 models — extend palette accordingly
PALETTE_LIST = ["#4C72B0", "#DD8452", "#8E44AD", "#55A868", "#E74C3C"]

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        130,
})


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def load_results():
    with open(RESULTS_JSON) as f:
        return json.load(f)

def load_features():
    df = pd.read_csv(FEATURES_CSV)#.dropna()
    meta_cols    = {"label", "filename", "side"}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    return (df[feature_cols].values,
            df["label"].values,
            df["filename"].values,
            feature_cols)

def load_models():
    return {
        "Random Forest":     joblib.load(MODELS_DIR / "rf_pipeline.joblib"),
        "MLP":               joblib.load(MODELS_DIR / "mlp_pipeline.joblib"),
        "Gradient Boosting": joblib.load(MODELS_DIR / "gbm_pipeline.joblib"),
        "RF + MLP":          joblib.load(MODELS_DIR / "rf_mlp_pipeline.joblib"),
        "RF + MLP + GBM":    joblib.load(MODELS_DIR / "rf_mlp_gbm_pipeline.joblib"),
    }

def save(fig, name):
    path = OUTPUTS_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  💾 {path.name}")
    return path

def dark_ax(ax):
    ax.set_facecolor("#0f1117")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

def dark_fig(fig):
    fig.patch.set_facecolor("#0f1117")


# ═══════════════════════════════════════════════════════════════
#  CHART 1 — Model comparison bar chart
# ═══════════════════════════════════════════════════════════════

def chart_comparison(results):
    metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
    labels  = ["Accuracy", "F1", "Precision", "Recall", "ROC-AUC"]
    models  = results["models"]
    n       = len(models)

    x = np.arange(len(metrics))
    w = 0.15
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * w

    fig, ax = plt.subplots(figsize=(13, 5))
    dark_fig(fig); dark_ax(ax)

    for i, (m, col, offset) in enumerate(zip(models, PALETTE_LIST, offsets)):
        vals = [m[k] for k in metrics]
        bars = ax.bar(x + offset, vals, w, label=m["name"], color=col, alpha=0.9, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.006,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", color="white")
    ax.set_title("Model Comparison — All Metrics", color="white", fontsize=13, pad=10)
    ax.legend(facecolor="#1e2130", labelcolor="white", edgecolor="#444", fontsize=8,
              loc="upper right")
    ax.grid(axis="y", color="#333", linewidth=0.6, zorder=0)
    return save(fig, "01_model_comparison.png")


# ═══════════════════════════════════════════════════════════════
#  CHART 2 — CV F1 box plots
# ═══════════════════════════════════════════════════════════════

def chart_cv_boxplot(results):
    models = results["models"]
    data   = [m["cv_f1"] for m in models]
    names  = [m["name"] for m in models]

    fig, ax = plt.subplots(figsize=(9, 5))
    dark_fig(fig); dark_ax(ax)

    bp = ax.boxplot(data, patch_artist=True, notch=True,
                    medianprops=dict(color="white", linewidth=2))
    for patch, col in zip(bp["boxes"], PALETTE_LIST):
        patch.set_facecolor(col); patch.set_alpha(0.85)
    for el in bp["whiskers"] + bp["caps"] + bp["fliers"]:
        el.set_color("#aaa")

    ax.set_xticklabels(names, color="white", fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("F1 Score (CV fold)", color="white")
    ax.set_title("Cross-Validation F1 Distribution", color="white", fontsize=13)
    ax.grid(axis="y", color="#333", linewidth=0.6)
    plt.tight_layout()
    return save(fig, "02_cv_f1_boxplot.png")


# ═══════════════════════════════════════════════════════════════
#  CHART 3 — Confusion matrices
# ═══════════════════════════════════════════════════════════════

def chart_confusion_matrices(results):
    models      = results["models"]
    class_names = [LABEL_NAMES[LABEL_BAD], LABEL_NAMES[LABEL_GOOD]]
    n           = len(models)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    dark_fig(fig)
    fig.suptitle("Confusion Matrices", color="white", fontsize=13, y=1.02)

    for ax, m, col in zip(axes, models, PALETTE_LIST):
        cm      = np.array(m["confusion_matrix"])
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-6)

        ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_facecolor("#1a1d2e")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(class_names, color="white", fontsize=8)
        ax.set_yticklabels(class_names, color="white", fontsize=8)
        ax.set_xlabel("Predicted", color="white", fontsize=8)
        ax.set_ylabel("True",      color="white", fontsize=8)
        ax.set_title(m["name"], color=col, fontsize=9, pad=6)

        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]*100:.1f}%)",
                        ha="center", va="center", fontsize=9,
                        color="white" if cm_norm[i, j] < 0.5 else "black")

    plt.tight_layout()
    return save(fig, "03_confusion_matrices.png")


# ═══════════════════════════════════════════════════════════════
#  CHART 4 — Feature importances (RF, GBM, MLP proxy)
# ═══════════════════════════════════════════════════════════════

def chart_feature_importance(results):
    rf_imp  = results["models"][0].get("feature_importance", {})
    mlp_imp = results["models"][1].get("feature_importance", {})
    gbm_imp = results["models"][2].get("feature_importance", {})

    if not rf_imp:
        print("  [SKIP] Feature importance data not found.")
        return

    feat_names = list(rf_imp.keys())
    rf_vals    = [rf_imp[f]          for f in feat_names]
    mlp_vals   = [mlp_imp.get(f, 0)  for f in feat_names]
    gbm_vals   = [gbm_imp.get(f, 0)  for f in feat_names]

    order      = np.argsort(rf_vals)[::-1]
    feat_names = [feat_names[i] for i in order]
    rf_vals    = [rf_vals[i]    for i in order]
    mlp_vals   = [mlp_vals[i]   for i in order]
    gbm_vals   = [gbm_vals[i]   for i in order]

    y = np.arange(len(feat_names))
    w = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    dark_fig(fig); dark_ax(ax)

    ax.barh(y + w,   rf_vals,  w, label="RF",  color=PALETTE_LIST[0], alpha=0.9)
    ax.barh(y,       mlp_vals, w, label="MLP", color=PALETTE_LIST[1], alpha=0.9)
    ax.barh(y - w,   gbm_vals, w, label="GBM", color=PALETTE_LIST[2], alpha=0.9)

    ax.set_yticks(y)
    ax.set_yticklabels(feat_names, color="white", fontsize=9)
    ax.set_xlabel("Importance", color="white")
    ax.set_title("Feature Importance — RF, GBM & MLP proxy", color="white", fontsize=12)
    ax.legend(facecolor="#1e2130", labelcolor="white", edgecolor="#444")
    ax.grid(axis="x", color="#333", linewidth=0.6)
    plt.tight_layout()
    return save(fig, "04_feature_importance.png")


# ═══════════════════════════════════════════════════════════════
#  CHART 5 — ROC curves
# ═══════════════════════════════════════════════════════════════

def chart_roc_curves(models_dict, X_test, y_test):
    fig, ax = plt.subplots(figsize=(7, 6))
    dark_fig(fig); dark_ax(ax)

    for (name, clf), col in zip(models_dict.items(), PALETTE_LIST):
        y_prob = (clf.predict_proba(X_test)[:, 1]
                if hasattr(clf, "predict_proba")
                else clf.decision_function(X_test))
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_val     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{name}  (AUC={roc_val:.3f})")

    ax.plot([0, 1], [0, 1], color="#555", lw=1, linestyle="--")
    ax.set_xlabel("False Positive Rate", color="white")
    ax.set_ylabel("True Positive Rate",  color="white")
    ax.set_title("ROC Curves", color="white", fontsize=13)
    ax.legend(facecolor="#1e2130", labelcolor="white", edgecolor="#444", fontsize=8)
    ax.grid(color="#333", linewidth=0.5)
    return save(fig, "05_roc_curves.png")


# ═══════════════════════════════════════════════════════════════
#  CHART 6 — Sample prediction grids (x5 randomised copies)
# ═══════════════════════════════════════════════════════════════

SKELETON_CONNECTIONS = [("ear", "shoulder"), ("shoulder", "hip")]
KP_COLORS = {
    "ear":      (0,   255, 180),
    "shoulder": (0,   180, 255),
    "hip":      (255,  60, 120),
}

def _detect_side(lms):
    nose_x = lms[0].x
    if nose_x < 0.42: return "RIGHT"
    if nose_x > 0.58: return "LEFT"
    lv = sum(lms[i].visibility for i in [11, 23, 25])
    rv = sum(lms[i].visibility for i in [12, 24, 26])
    return "LEFT" if lv >= rv else "RIGHT"

def draw_pose_on_image(img_bgr, landmarker):
    h, w = img_bgr.shape[:2]
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    )
    result = landmarker.detect(mp_image)
    if not result.pose_landmarks:
        return img_bgr.copy()

    lms  = result.pose_landmarks[0]
    out  = img_bgr.copy()
    side = _detect_side(lms)
    idx_map = ({"ear": 7, "shoulder": 11, "hip": 23}
            if side == "LEFT"
            else {"ear": 8, "shoulder": 12, "hip": 24})

    def pt(name):
        lm = lms[idx_map[name]]
        return (int(lm.x * w), int(lm.y * h)) if lm.visibility >= VISIBILITY_THRESHOLD else None

    kpts = {name: pt(name) for name in idx_map}
    kpts = {k: v for k, v in kpts.items() if v is not None}

    for (n1, n2) in SKELETON_CONNECTIONS:
        if n1 in kpts and n2 in kpts:
            cv2.line(out, kpts[n1], kpts[n2], (200, 200, 200), 2, cv2.LINE_AA)
    for name, pos in kpts.items():
        col = KP_COLORS.get(name, (255, 255, 255))
        cv2.circle(out, pos, 7, col, -1, cv2.LINE_AA)
        cv2.circle(out, pos, 7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, name[:3], (pos[0] + 8, pos[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return out

def find_image_path(filename: str):
    for d in [AUG_DIR, GOOD_DIR, BAD_DIR]:
        p = d / filename
        if p.exists(): return p
        hits = list(d.rglob(filename))
        if hits: return hits[0]
    return None

def make_landmarker():
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)

def _build_sample_grid(clf, y, filenames, pred, rng, n_per_group, landmarker):
    """Build one prediction sample figure. rng controls which samples are picked."""
    real_mask = np.array([f != "synthetic" for f in filenames], dtype=bool)

    good_correct  = [i for i in np.where((y == LABEL_GOOD) & (pred == LABEL_GOOD))[0] if real_mask[i]]
    bad_correct   = [i for i in np.where((y == LABEL_BAD)  & (pred == LABEL_BAD))[0]  if real_mask[i]]
    misclassified = [i for i in np.where(y != pred)[0]                                 if real_mask[i]]

    rng.shuffle(good_correct)
    rng.shuffle(bad_correct)
    rng.shuffle(misclassified)

    groups = [
        ("Correctly → Good Posture", good_correct[:n_per_group],  "#55A868"),
        ("Correctly → Bad Posture",  bad_correct[:n_per_group],   "#DD8452"),
        ("Misclassified",            misclassified[:n_per_group], "#E74C3C"),
    ]

    fig = plt.figure(figsize=(n_per_group * 3.2, len(groups) * 3.5))
    fig.patch.set_facecolor("#0f1117")
    cell = 1

    for row_i, (group_name, indices, group_col) in enumerate(groups):
        for col_i in range(n_per_group):
            ax = fig.add_subplot(len(groups), n_per_group, cell)
            ax.set_facecolor("#0f1117")
            ax.axis("off")
            cell += 1

            if col_i < len(indices):
                idx  = indices[col_i]
                path = find_image_path(filenames[idx])
                if path:
                    img = cv2.imread(str(path))
                    if img is not None:
                        img = draw_pose_on_image(img, landmarker)
                        img = cv2.resize(img, (280, 280))
                        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                true_name = LABEL_NAMES[y[idx]]
                pred_name = LABEL_NAMES[pred[idx]]
                correct   = (y[idx] == pred[idx])
                ax.set_title(f"T:{true_name[:4]}  P:{pred_name[:4]}",
                            color="#55A868" if correct else "#E74C3C",
                            fontsize=8, pad=3)

        fig.text(0.01, 1 - (row_i + 0.5) / len(groups),
                group_name, va="center", ha="left",
                color=group_col, fontsize=10, rotation=90)

    return fig

def chart_sample_predictions(models_dict, X, y, filenames, n_per_group=4, n_copies=5):
    # Use the best ensemble for the sample display
    clf  = models_dict["RF + MLP + GBM"]
    pred = clf.predict(X)

    print(f"  Building {n_copies} randomised prediction sample grids…")
    landmarker = make_landmarker()

    for i in range(1, n_copies + 1):
        rng = random.Random()   # new unseeded RNG each iteration → different samples
        fig = _build_sample_grid(clf, y, filenames, pred, rng, n_per_group, landmarker)
        fig.suptitle(
            f"Prediction Samples — RF+MLP+GBM Ensemble  (set {i}/{n_copies})",
            color="white", fontsize=12, y=1.01,
        )
        plt.tight_layout()
        save(fig, f"06_prediction_samples_{i:02d}.png")

    landmarker.close()


# ═══════════════════════════════════════════════════════════════
#  CHART 7 — Score card
# ═══════════════════════════════════════════════════════════════

def chart_scorecard(results):
    models  = results["models"]
    metrics = ["accuracy", "f1", "precision", "recall", "roc_auc", "cv_f1_mean", "cv_f1_std"]
    headers = ["Accuracy", "F1", "Precision", "Recall", "ROC-AUC", "CV-F1 μ", "CV-F1 σ"]

    fig, ax = plt.subplots(figsize=(13, 2.8))
    dark_fig(fig)
    ax.set_facecolor("#0f1117")
    ax.axis("off")

    rows = [[m["name"]] + [f"{m[k]:.4f}" for k in metrics] for m in models]

    table = ax.table(cellText=rows, colLabels=["Model"] + headers,
                    loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.9)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#333")
        if r == 0:
            cell.set_facecolor("#1e2130")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#161922" if r % 2 == 0 else "#1a1d2e")
            cell.set_text_props(color="white")
            if c > 0:
                col_vals = [float(rows[ri][c]) for ri in range(len(rows))]
                best_idx = np.argmin(col_vals) if c == len(headers) else np.argmax(col_vals)
                if r - 1 == best_idx:
                    cell.set_facecolor("#1d3a26")

    ax.set_title("Model Score Card", color="white", fontsize=12, pad=8)
    plt.tight_layout()
    return save(fig, "07_scorecard.png")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def run():
    if not RESULTS_JSON.exists():
        print("  [ERROR] results.json not found. Run Step 4 first.")
        return

    results     = load_results()
    X, y, fns, feature_names = load_features()
    models_dict = load_models()

    # Reproduce real-only test split from Step 4
    df = pd.read_csv(FEATURES_CSV)

    real_df = df[df["filename"] != "synthetic"].copy()

    # safety check
    if len(real_df) == 0:
        raise ValueError(
            "No real samples found. Your CSV likely contains only synthetic data "
            "or dropna() removed everything."
        )

    X_real   = real_df[feature_names].values
    y_real   = real_df["label"].values
    fns_real = real_df["filename"].values

    _, X_test, _, y_test, _, _ = train_test_split(
        X_real, y_real, fns_real,
        test_size=TEST_SIZE,
        stratify=y_real,
        random_state=RANDOM_SEED,
    )

    print("  Generating charts…")
    chart_comparison(results)
    chart_cv_boxplot(results)
    chart_confusion_matrices(results)
    chart_feature_importance(results)
    chart_roc_curves(models_dict, X_test, y_test)
    chart_sample_predictions(models_dict, X, y, fns, n_per_group=4, n_copies=5)
    chart_scorecard(results)

    print(f"\n  ✅ All charts saved to: {OUTPUTS_DIR}")
    for f in sorted(OUTPUTS_DIR.glob("*.png")):
        print(f"    {f.name}")


if __name__ == "__main__":
    run()
