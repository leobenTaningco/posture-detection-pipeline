# Posture Detection Pipeline
**RF + MLP Soft-Voting Ensemble with MediaPipe Pose Landmarks**

Detects whether a seated person has **good** or **bad** posture from static images,
using biomechanical angles extracted by MediaPipe (hip→head segment only).

---

## Project Structure

```
posture_pipeline/
├── run_pipeline.py          ← entry point
├── config.py                ← all paths & hyper-parameters
├── requirements.txt
├── dataset/
│   ├── raw/                 ← ⬅ DROP YOUR IMAGES HERE
│   ├── augmented/           ← Step 1 output
│   ├── sorted/
│   │   ├── goodPosture/     ← Step 3 output
│   │   └── badPosture/      ← Step 3 output
│   └── features.csv         ← Step 2/3 output
├── models/
│   ├── rf_pipeline.joblib
│   ├── mlp_pipeline.joblib
│   └── ensemble_pipeline.joblib
├── outputs/                 ← all charts and results
│   ├── 01_model_comparison.png
│   ├── 02_cv_f1_boxplot.png
│   ├── 03_confusion_matrices.png
│   ├── 04_feature_importance.png
│   ├── 05_roc_curves.png
│   ├── 06_prediction_samples.png   ← randomised each run
│   ├── 07_scorecard.png
│   └── results.json
└── steps/
    ├── step1_augment.py
    ├── step2_extract.py
    ├── step3_dedupe_balance.py
    ├── step4_train.py
    └── step5_analytics.py
```

---

## Setup

```bash
# 1. Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Image Naming Convention

Images **must** contain either `goodPosture` or `badPosture` (case-insensitive)
anywhere in their filename.

```
✅  john_goodPosture_001.jpg
✅  badPosture_frame042.png
✅  session3_goodposture_side.jpg
❌  sitting_upright.jpg            ← will be ignored (no label in name)
```

---

## Running the Pipeline
```
python run_all.py
```
just edit run_all.py as needed

---

## Features Used (MediaPipe Hip→Head Segment)

| Feature | Description |
|---|---|
| `neck_inclination` | Angle between ear→shoulder and vertical |
| `torso_inclination` | Angle between hip→shoulder and vertical |
| `arm_inclination` | Angle between shoulder→elbow and vertical |
| `neck_ratio` | ear-shoulder distance / shoulder-hip distance |
| `ear_shoulder_y_diff` | Normalised vertical gap (forward-head proxy) |
| `torso_lean` | Horizontal displacement hip→shoulder (normalised) |
| `head_forward_angle` | 3-point angle: ear–shoulder–hip |
| `shoulder_hip_angle` | Shoulder→hip deviation from vertical |

---

## Augmentations Applied (Step 1)

- Brightness & contrast jitter
- Random rotation ±10° (small, suitable for sitting)
- Horizontal flip (left/right seated views are equivalent)
- Gaussian noise
- Occasional Gaussian blur (simulates camera focus variation)
- Sharpening on originals

**Default**: 6 augmented copies per original image (configurable in `config.py`)

---

## Deduplication (Step 3)

Perceptual hashing (pHash via `imagehash`, dHash fallback via OpenCV) removes
near-identical frames **within each class independently**, keeping the original
over augmented copies when there's a hash collision.
Hamming distance threshold: `SIMILARITY_THRESHOLD = 8` (configurable).

---

## Class Balancing (Step 3)

- If imbalance ratio > 1.5×: SMOTE (if `imbalanced-learn` installed) or
  random oversampling of minority class rows in the feature CSV.
- RF also uses `class_weight="balanced"` internally.

---

## Models Output (Step 4)

| Model | Notes |
|---|---|
| Random Forest | `n_estimators=300`, `class_weight=balanced` |
| MLP | `(128→64→32)`, adaptive LR, early stopping |
| Ensemble | Soft voting, equal weights |

All three are saved as `sklearn.pipeline.Pipeline` objects (scaler + clf) via
`joblib`. Load them directly:

```python
import joblib
clf = joblib.load("models/ensemble_pipeline.joblib")
prediction = clf.predict(feature_vector)   # shape (1, 8)
```

---

## Charts Output (Step 5)

| File | Content |
|---|---|
| `01_model_comparison.png` | Bar chart: Acc/F1/Precision/Recall/AUC across 3 models |
| `02_cv_f1_boxplot.png` | CV F1 distribution (notched box plots) |
| `03_confusion_matrices.png` | Normalised confusion matrices, side by side |
| `04_feature_importance.png` | RF importance + MLP weight-proxy, horizontal bar |
| `05_roc_curves.png` | ROC curves with AUC annotations |
| `06_prediction_samples.png` | Random sample images with MediaPipe overlay (re-randomised each run) |
| `07_scorecard.png` | Summary table (best values highlighted in green) |

---

## Configuration

All tunable parameters live in `config.py`:

```python
AUG_PER_IMAGE        = 6        # augmented copies per original
SIMILARITY_THRESHOLD = 8        # hash distance for dedup
RF_PARAMS            = { ... }  # RandomForestClassifier kwargs
MLP_PARAMS           = { ... }  # MLPClassifier kwargs
TEST_SIZE            = 0.20
RANDOM_SEED          = 42
```
