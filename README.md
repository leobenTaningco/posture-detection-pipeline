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
# Sitting Posture Classification Pipeline

A machine learning pipeline that classifies sitting posture as **Good** or **Bad** using pose estimation landmarks extracted from side-view images. The system uses an ensemble of a Random Forest and a Multilayer Perceptron, trained on geometric angles derived from three anatomical landmarks: the ear, shoulder, and hip.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Pipeline Summary](#2-pipeline-summary)
3. [Libraries Used](#3-libraries-used)
4. [Algorithms Explained](#4-algorithms-explained)
5. [Feature Extraction and Angles](#5-feature-extraction-and-angles)
6. [Function Reference](#6-function-reference)
7. [Parameter Rationale](#7-parameter-rationale)
8. [File Structure](#8-file-structure)

---

## 1. Project Overview

This project tackles a practical problem: detecting whether a person is sitting with good or bad posture from a side-view camera feed or image dataset. Rather than analyzing raw pixel data, the pipeline reduces each image to a small set of geometric measurements — angles and distances derived from three key body landmarks — and trains classification models on those measurements.

The approach is intentionally compact. Using only three landmarks (ear, shoulder, hip) keeps the system fast and interpretable, while still capturing the most meaningful signals of spinal alignment and head position.

The final product is a webcam-based real-time classifier that labels posture continuously, with smoothing applied to prevent jitter.

> **In the simplest terms:** The system looks at where your ear, shoulder, and hip are in a side-view image, measures the angles between them, and uses those angles to decide if you are sitting well or not.

---

## 2. Pipeline Summary

The pipeline is divided into five sequential steps, each implemented as an independent script:

**Step 1 — Augmentation (`step1_augment.py`)**
Raw images are expanded through augmentation — applying small rotations and horizontal flips to generate additional training samples. This compensates for datasets that are too small for reliable model training.

**Step 2 — Feature Extraction (`step2_extract.py`)**
MediaPipe processes each augmented image to locate the ear, shoulder, and hip landmarks. Geometric features (angles, ratios, distances) are computed from these three points and saved to a CSV file.

**Step 3 — Balancing (`step3_dedupe_balance.py`)**
The dataset is balanced by undersampling the majority class so that good and bad posture examples appear in equal numbers. This prevents the model from being biased toward whichever class has more images.

**Step 4 — Training (`step4_train.py`)**
Four models are trained on the feature CSV: a Random Forest, a Multilayer Perceptron, and two ensemble combinations (soft voting and stacking). All models are evaluated on a held-out test set.

**Step 5 — Analytics (`step5_analytics.py`)**
Evaluation charts are generated: model comparison bars, confusion matrices, ROC curves, feature importance rankings, and a summary scorecard.

> **In the simplest terms:** Gather images, measure body angles, make sure you have equal examples of both postures, train multiple models, and then chart the results.

---

## 3. Libraries Used

### MediaPipe (`mediapipe >= 0.10.0`)

MediaPipe is Google's framework for real-time perception tasks. This pipeline uses its Pose Landmarker model, which detects 33 body landmarks from an image and returns their coordinates along with visibility confidence scores. Only three of those landmarks are used here — the ear, shoulder, and hip — because the images are captured from a strict side view.

The `PoseLandmarker` is configured in `IMAGE` mode for batch processing during training and in `VIDEO` mode for the live webcam feed, which allows it to use temporal context between frames.

> **In the simplest terms:** MediaPipe is the part of the system that finds where your ear, shoulder, and hip are in a photo.

---

### OpenCV (`opencv-python >= 4.8.0`)

OpenCV handles all image input and output operations. In this pipeline it is used to read images from disk, apply preprocessing (sharpening, contrast enhancement, upscaling) before passing them to MediaPipe, perform small rotations during augmentation, and render the live camera overlay during inference.

> **In the simplest terms:** OpenCV is used to open, transform, and display images.

---

### NumPy (`numpy >= 1.24.0`)

NumPy provides the numerical foundation for all geometric calculations. Vector subtraction, dot products, norms, and trigonometric operations used in angle computation all rely on NumPy arrays. It is also used for constructing and reshaping the feature vectors fed into the models.

> **In the simplest terms:** NumPy is the math toolkit used to compute all the angles and distances.

---

### Pandas (`pandas >= 2.0.0`)

Pandas manages the feature dataset as a structured table (a DataFrame). After MediaPipe processes all images, the extracted features are collected into a CSV file via Pandas. Each row represents one image, and each column is one numeric feature. Pandas is also used during training to load and split this dataset.

> **In the simplest terms:** Pandas stores and manages the spreadsheet of angle measurements extracted from every image.

---

### scikit-learn (`scikit-learn >= 1.3.0`)

scikit-learn is the primary machine learning library. It provides:

- `RandomForestClassifier` and `MLPClassifier` for the two base models.
- `GradientBoostingClassifier` as a comparison model.
- `VotingClassifier` and `StackingClassifier` for the ensemble models.
- `StandardScaler` and `SimpleImputer` for data preprocessing within pipelines.
- `train_test_split` and `StratifiedKFold` for dataset splitting.
- Metrics such as accuracy, F1 score, ROC-AUC, and confusion matrices.
- `CalibratedClassifierCV` for probability calibration.
- `permutation_importance` for measuring MLP feature contributions.

> **In the simplest terms:** scikit-learn is the library that actually trains and evaluates all the models.

---

### imbalanced-learn (`imbalanced-learn >= 0.11.0`)

This library extends scikit-learn with tools for handling class imbalance. Although Step 3 uses a manual undersampling approach rather than imbalanced-learn's resampler classes directly, the library is listed as a dependency because it provides compatible interfaces when working with imbalanced datasets in the scikit-learn ecosystem.

> **In the simplest terms:** A utility library for dealing with datasets where one class has many more examples than the other.

---

### Matplotlib (`matplotlib >= 3.7.0`)

Matplotlib generates all evaluation charts in Step 5: bar charts, box plots, confusion matrix heatmaps, feature importance charts, and ROC curves. It is set to the `Agg` backend, meaning it renders charts to files rather than displaying an interactive window — suitable for headless or server environments.

> **In the simplest terms:** Matplotlib draws and saves all the result graphs.

---

### tqdm (`tqdm >= 4.65.0`)

tqdm provides progress bars for the long-running loops in Step 1 (augmentation) and Step 2 (feature extraction). Each bar shows how many images have been processed and estimates the time remaining.

> **In the simplest terms:** tqdm shows a progress bar so you can see how far along the processing is.

---

### joblib (bundled with scikit-learn)

joblib serializes trained model objects to `.joblib` files on disk so they can be loaded later without retraining. It is also used to parallelize certain computations such as permutation importance across CPU cores (`n_jobs=-1`).

> **In the simplest terms:** joblib saves and loads trained models to and from disk.

---

## 4. Algorithms Explained

### Random Forest (RF)

A Random Forest is an ensemble of decision trees. Each tree is trained on a randomly sampled subset of the training data, and each split within a tree considers only a random subset of features (`max_features="sqrt"`). When making a prediction, every tree votes, and the majority vote wins.

This randomness serves two purposes. First, it reduces overfitting: no single tree has seen all the data, so no single tree can memorize it. Second, it reduces correlation between trees, meaning their errors tend to cancel each other out rather than compounding.

The probability output of a Random Forest is the fraction of trees that voted for each class, which can be consumed directly by the ensemble.

> **In the simplest terms:** A Random Forest trains hundreds of slightly different decision trees on different slices of the data, then combines their votes. The diversity makes the overall result more reliable than any single tree.

---

### Multilayer Perceptron (MLP)

An MLP is a feedforward neural network. Data passes through an input layer, one or more hidden layers of neurons, and an output layer that produces class probabilities. Each neuron computes a weighted sum of its inputs and applies a non-linear activation function (`relu` in this case), which allows the network to learn non-linear boundaries between classes.

Because input features vary in scale (angles are in degrees, ratios are between 0 and 1), the MLP requires `StandardScaler` as a preprocessing step — without scaling, larger-valued features would dominate the gradient updates during training.

Early stopping is enabled: training monitors a validation set and halts when the score stops improving, preventing overfitting.

> **In the simplest terms:** An MLP is a small neural network that learns which combinations of angle measurements separate good posture from bad posture, improving itself through many training iterations.

---

### Soft Voting Ensemble

The voting ensemble combines the RF and MLP by averaging their probability outputs. Each model predicts a probability for the "Good Posture" class, these are averaged (with MLP weighted 1.5 to RF's 1.0), and the final label is determined by whether the average exceeds 0.5.

This is called "soft" voting because it uses the probability values rather than hard class labels. Soft voting generally outperforms hard voting when the component models produce well-calibrated probabilities.

> **In the simplest terms:** Both models give a confidence score, those scores are averaged, and the average decides the final answer.

---

### Stacking Ensemble

Stacking trains a meta-model (a Logistic Regression) on top of the base models. The RF and MLP each produce their class probability predictions for the training set using cross-validation, and the meta-model learns how to best combine those predictions into a final decision.

Unlike voting, which uses fixed weights, stacking learns an optimal combination function from data. This can outperform voting when the base models make systematically different types of errors.

> **In the simplest terms:** Instead of averaging both models equally, stacking trains a third model that learns the best way to combine the first two.

---

### Probability Calibration

After training, `CalibratedClassifierCV` is applied to both the RF and MLP. Raw classifier outputs are not always true probabilities — a model might output 0.9 when the actual class frequency at that confidence is only 0.7. Calibration (using the `isotonic` method) fits a monotone function that maps raw scores to reliable probabilities. This is important because the ensemble models and the threshold-tuning step both depend on probability values being meaningful.

> **In the simplest terms:** Calibration makes sure that when the model says "70% confident," it is actually correct about 70% of the time.

---

### Threshold Optimization

By default, a binary classifier predicts the positive class when probability exceeds 0.5. This default is not always optimal. The pipeline uses the precision-recall curve to find the threshold that maximizes the F1 score on the full dataset. This threshold is then applied during test evaluation. The result is a classifier tuned to balance precision and recall rather than just accuracy.

> **In the simplest terms:** Instead of always using 0.5 as the cutoff, the system finds the cutoff point where the model performs best overall.

---

## 5. Feature Extraction and Angles

All features are derived from three landmarks: the **ear**, the **shoulder**, and the **hip**, detected from side-view images.

### Why Side View?

In a side-view perspective, spinal alignment and head position are directly observable as geometric angles. A front-facing view would collapse depth information, making it impossible to detect forward head lean or spinal curvature.

Side detection is implemented to handle datasets where subjects face either left or right. The pipeline computes visibility scores for both the left set (ear index 7, shoulder 11, hip 23) and the right set (ear 8, shoulder 12, hip 24) and selects whichever side has higher aggregate visibility. This prevents features from being computed from partially occluded landmarks.

---

### Neck Inclination

```
find_inclination(shoulder_x, shoulder_y, ear_x, ear_y)
```

This is the angle between the vertical axis and the line connecting the shoulder to the ear. It is computed as `atan2(|x2 - x1|, |y2 - y1|)`, converted to degrees. A value near zero means the ear is directly above the shoulder (upright neck). As the head moves forward relative to the shoulder, this angle increases.

> **In the simplest terms:** Neck inclination measures how far the head has drifted forward from the shoulder. The larger the angle, the more the head is leaning forward.

---

### Torso Inclination

```
find_inclination(hip_x, hip_y, shoulder_x, shoulder_y)
```

The same formula applied to the line from hip to shoulder. Near zero indicates the torso is vertically upright. A larger value indicates slouching or leaning.

> **In the simplest terms:** Torso inclination measures how much the upper body is leaning away from vertical. A straighter torso gives a smaller angle.

---

### Head Forward Angle

```
three_point_angle(ear, shoulder, hip)
```

This computes the interior angle at the shoulder vertex of the triangle formed by the three landmarks. It uses the dot product formula: `cos(angle) = (BA · BC) / (|BA| |BC|)`. This angle captures the spatial relationship between all three points simultaneously. When the ear and hip are roughly aligned above the shoulder, this angle is large (near 180 degrees in ideal posture). As the head moves forward or the torso collapses, this angle decreases.

> **In the simplest terms:** This angle, measured at the shoulder, captures the overall shape of the body. Good posture keeps this angle close to 180 degrees; bad posture makes it smaller.

---

### Derived Features

From the three base angles and the raw landmark coordinates, additional composite features are computed in `features.py`:

| Feature | Formula | Interpretation |
|---|---|---|
| `neck_ratio` | `ear-shoulder distance / shoulder-hip distance` | How long the neck segment is relative to the torso segment |
| `torso_height_ratio` | `(hip_y - shoulder_y) / image height` | Normalized vertical torso length |
| `ear_shoulder_y_diff` | `(shoulder_y - ear_y) / image height` | Vertical offset between ear and shoulder; negative when head drops |
| `ear_hip_x_dist` | `abs(ear_x - hip_x) / image width` | Horizontal distance between ear and hip; increases with forward lean |
| `neck_forward_combo` | `neck_inclination * head_forward_angle` | Interaction term amplifying the joint effect of neck and head position |
| `posture_balance` | `torso_inclination - neck_inclination` | Whether the torso and neck are leaning in the same or opposite directions |
| `compactness` | `neck_ratio / torso_height_ratio` | Relative compactness of the upper body |
| `alignment_score` | `abs(ear_shoulder_y_diff) + abs(ear_hip_x_dist)` | Combined measure of vertical and horizontal misalignment |
| `neck_torso_interaction` | `neck_inclination * torso_inclination` | Joint severity of both inclinations |
| `forward_vs_tilt` | `head_forward_angle - torso_inclination` | Whether the head is leaning more than the torso, or vice versa |

> **In the simplest terms:** Beyond raw angles, these derived features capture combinations and ratios that help the model distinguish more subtle postural differences.

---

## 6. Function Reference

### `step1_augment.py`

**`rotate_image(img, angle)`**
Rotates an image by the given angle in degrees around its center using an affine transformation. Boundary pixels are filled by reflecting the image edge to avoid black borders. Rotation is kept small (within -10 to +10 degrees) because larger rotations would distort the anatomical angles the model is meant to learn.

**`augment_one(img)`**
Applies a randomly sampled rotation and a 50% chance of horizontal flip to produce one augmented copy of an image. The horizontal flip is valid for posture because the body geometry is symmetric — a person facing left and a person facing right in good posture are equivalent once the relevant side landmarks are selected.

**`run(dataset_path)`**
Iterates over all supported image files in the raw dataset directory, copies each original, generates `AUG_PER_IMAGE` augmented variants, and writes all outputs to the augmentation directory. The augmentation directory is wiped before each run to prevent stale data accumulation.

---

### `step2_extract.py`

**`find_inclination(x1, y1, x2, y2)`**
Computes the angle (in degrees) between the vertical axis and the line connecting two points using `atan2`. The vertical is used as the reference because posture deviations are naturally measured relative to the gravitational direction.

**`three_point_angle(a, b, c)`**
Computes the angle at vertex `b` in the triangle formed by points `a`, `b`, and `c` using the dot product formula. A small epsilon is added to the denominator to prevent division by zero when two points coincide.

**`sharpen(img)`, `enhance_contrast(img)`, `upscale(img)`**
Preprocessing functions applied before MediaPipe inference. Sharpening improves edge definition. Contrast enhancement using CLAHE on the luminance channel makes landmarks visible in low-contrast images. Upscaling by 1.5x improves landmark detection accuracy on small or low-resolution source images.

**`label_from_name(name)`**
Infers the ground-truth label from the image filename. Files containing "goodposture" in their name are assigned label 1; those containing "badposture" are assigned label 0. Files matching neither pattern are skipped.

**`get_keypoints(image_rgb, landmarker)`**
Runs MediaPipe on an image and extracts the ear, shoulder, and hip landmarks for both sides of the body. For each side, it computes a visibility score (sum of individual landmark visibility values). It selects the side with higher total visibility, provided all three landmarks on that side meet the minimum visibility threshold. If neither side passes this check, the function returns `None` and the image is skipped.

**`extract_features(kp, img_h, img_w)`**
Converts the raw pixel coordinates of the three landmarks into the ten base numeric features. Distances are normalized by image dimensions to make features scale-invariant across different image resolutions.

---

### `step3_dedupe_balance.py`

**`balance_undersample(df)`**
Identifies the majority and minority classes by count. The majority class is downsampled (without replacement) to match the minority class count exactly. The combined dataset is shuffled before saving. This ensures the model sees both classes with equal frequency during training, preventing it from defaulting to the majority class.

---

### `step4_train.py`

**`make_rf_pipeline()`**
Constructs a scikit-learn Pipeline combining a median imputer (to handle any missing feature values) with a Random Forest classifier. Wrapping these steps in a Pipeline ensures the imputer is always applied consistently whenever the model is used.

**`make_mlp_pipeline()`**
Constructs a Pipeline with a median imputer, a StandardScaler, and an MLP classifier. The scaler is essential for the MLP because the optimizer's gradient steps are sensitive to feature magnitude differences.

**`find_best_threshold(y_true, y_prob)`**
Iterates over the precision-recall curve to identify the probability threshold that produces the highest F1 score. This threshold replaces the default 0.5 during final evaluation.

**`evaluate(name, clf, X_test, y_test, threshold)`**
Runs a trained classifier on the test set at the specified threshold and returns a dictionary of accuracy, F1, precision, recall, ROC-AUC, and the confusion matrix.

---

### `features.py`

**`build_engineered_features(df)`**
Takes a DataFrame of base features and appends the six engineered interaction columns. Used during the training pipeline to produce the full feature set from the CSV.

**`build_webcam_features(raw)`**
Constructs the complete feature dictionary for a single live frame during webcam inference. The input is a dictionary of raw values (inclinations, landmark coordinates, distances). The function computes all base and engineered features to exactly match the schema used during training. This schema consistency is critical: if the features fed to the model at inference time differ from those used during training, predictions will be unreliable.

---

### `testUsingCamera.py`

**`smooth_keypoints(new_kp, old_kp, alpha)`**
Applies an exponential moving average to the landmark coordinates across frames. With `alpha=0.4`, 40% of the new position and 60% of the previous position contribute to the smoothed output. This reduces jitter caused by frame-to-frame noise in the landmark detector.

**`get_keypoints(image_rgb, timestamp_ms)`**
The real-time version of the training keypoint extractor. It uses `detect_for_video` rather than `detect`, passing a monotonic timestamp so that MediaPipe can apply temporal smoothing internally. Side selection logic mirrors the training pipeline.

**`extract_features(kp, img_h, img_w)`**
Converts the smoothed webcam keypoints into the feature vector fed to the model. The output shape is `(1, 10)` to match the model's expected input dimensionality.

**`draw_visuals(frame, kp)`**
Draws the three landmark points and the connecting lines (ear-to-shoulder, shoulder-to-hip) on the camera frame for visual feedback.

**`draw_overlay(frame, label, color, side, smooth_prob, show_warning, fps, ...)`**
Renders the prediction label, confidence score, detected body side, FPS counter, and a warning banner when bad posture has been sustained for more than ten consecutive prediction frames.

---

## 7. Parameter Rationale

### Augmentation Parameters

**`AUG_PER_IMAGE = 6`**
Each raw image produces six augmented copies plus the original, multiplying dataset size by seven. This number was chosen to meaningfully expand a small dataset without generating so many near-duplicate images that the model learns only the augmentation artifacts.

**`ROTATION_RANGE = (-10, 10)` degrees**
Sitting posture datasets typically capture subjects with minimal camera tilt. Rotations beyond +/-10 degrees would introduce unrealistic perspective distortions and could shift the anatomical angles enough to produce mislabeled training examples.

**`FLIP_HORIZONTAL = True`**
Horizontal flips create mirror images, effectively doubling the diversity of side-view directions. Since the side-detection logic in both training and inference selects the appropriate side automatically, flipped images are valid and correctly handled.

---

### MediaPipe Parameters

**`MP_MIN_DETECTION_CONF = 0.5`, `MP_MIN_TRACKING_CONF = 0.5`**
These thresholds determine the minimum confidence MediaPipe requires before reporting a detected landmark. A value of 0.5 is the standard default — low enough to detect landmarks in partially occluded or low-resolution images, but high enough to reject noise detections.

**`VISIBILITY_THRESHOLD = 0.5`**
Each individual landmark must meet this visibility score for its side to be considered valid. A landmark visible at 0.5 or above is generally reliably placed; below this, the coordinate estimate becomes too uncertain to use for angle computation.

---

### Training Parameters

**`TEST_SIZE = 0.20`**
Twenty percent of the real (non-augmented) dataset is held out for final evaluation. This follows the standard 80/20 split convention, providing enough test examples for stable metric estimates while maximizing training data.

**`RANDOM_SEED = 42`**
A fixed seed is set across all random operations (splits, sampling, model initialization) to ensure reproducibility. Any collaborator running the pipeline on the same data will obtain identical results.

---

### Random Forest Parameters

**`n_estimators = 500`**
Five hundred trees provide a stable ensemble. Accuracy typically plateaus well before 500, so this number ensures the ensemble is fully converged without being wasteful. With `n_jobs=-1`, all CPU cores are used in parallel, keeping runtime manageable.

**`max_depth = 12`**
Limiting tree depth prevents individual trees from memorizing the training data. Shallow trees underfit; unconstrained trees overfit. Depth 12 allows sufficient complexity for the feature space while providing regularization.

**`min_samples_split = 4`, `min_samples_leaf = 4`** (Step 4 values)
Requiring at least four samples to split a node and at least four samples in each leaf smooths the decision boundaries and further prevents overfitting on small clusters of training examples.

**`max_features = "sqrt"`**
At each split, the tree considers only the square root of the total number of features as candidates. This is the standard setting for classification tasks and is the primary source of decorrelation between trees in the forest.

**`class_weight = "balanced"`**
Adjusts per-class sample weights inversely proportional to class frequency. This provides an additional safeguard against class imbalance beyond the undersampling in Step 3.

---

### MLP Parameters

**`hidden_layer_sizes = (64, 32)`**
Two hidden layers with 64 and 32 neurons respectively. Given that the input has approximately 13 features, a deep or wide network would be dramatically over-parameterized. These compact layer sizes enforce a bottleneck that encourages the network to learn compressed representations rather than memorizing training examples.

**`alpha = 1e-2`**
L2 regularization coefficient. A higher value (compared to the default `1e-4`) adds stronger weight decay, penalizing large weights and reducing overfitting. This is set higher than typical because the feature space is small and the risk of overfitting is correspondingly high.

**`validation_fraction = 0.15`, `n_iter_no_change = 30`**
Early stopping monitors 15% of the training data as a validation set. Training halts if validation score does not improve for 30 consecutive iterations. The larger patience value (30 vs. the default 10) prevents premature stopping on noisy validation curves.

---

### Inference Parameters

**`ALPHA = 0.4` (keypoint smoothing)**
The exponential moving average coefficient for landmark position smoothing across webcam frames. A value of 0.4 gives moderate smoothing: the current frame contributes 40% and the history contributes 60%. Lower values produce smoother but more lagged output; higher values are more responsive but jittery.

**`GOOD_THRESHOLD = 0.55`, `BAD_THRESHOLD = 0.42`**
A hysteresis band around the decision boundary. If the smoothed probability exceeds 0.55, the system transitions to "Good Posture." If it falls below 0.42, it transitions to "Bad Posture." In between, the previous state is held. This dead zone prevents the label from flipping back and forth rapidly when the model is near its decision boundary.

**`DETECT_EVERY` (adaptive, 2 to 8 frames)**
MediaPipe runs every N frames rather than every frame, reducing CPU load. N adapts dynamically: when motion in the frame is high, detection runs more frequently (minimum every 2 frames); when the scene is static, it runs less often (maximum every 8 frames). This balances responsiveness against computational cost.

---

## 8. File Structure

```
project/
├── config.py                  # Shared paths and hyperparameters
├── features.py                # Feature engineering functions
├── step1_augment.py           # Data augmentation
├── step2_extract.py           # MediaPipe feature extraction
├── step3_dedupe_balance.py    # Class balancing
├── step4_train.py             # Model training and evaluation
├── step5_analytics.py         # Chart generation
├── testUsingCamera.py         # Real-time webcam inference
├── requirements.txt           # Python dependencies
├── dataset/
│   ├── raw/                   # Input images (place side-view images here)
│   ├── augmented/             # Step 1 output
│   └── features.csv           # Step 2/3 output
├── models/
│   ├── pose_landmarker_heavy.task   # MediaPipe model (training)
│   ├── pose_landmarker_lite.task    # MediaPipe model (inference)
│   ├── rf.joblib
│   ├── mlp.joblib
│   ├── voting.joblib
│   ├── stacking.joblib
│   └── feature_names.joblib
└── outputs/
    ├── results.json
    ├── feature_importance.json
    └── *.png                  # Generated charts
```

### Image Naming Convention

Images must include either `goodposture` or `badposture` (case-insensitive) in their filename for the label to be inferred automatically during feature extraction. For example: `subject01_goodposture.jpg`, `session3_badposture_002.png`.

---

## Running the Pipeline

```bash
pip install -r requirements.txt

python step1_augment.py
python step2_extract.py
python step3_dedupe_balance.py
python step4_train.py
python step5_analytics.py

# Real-time inference
python testUsingCamera.py
```

MediaPipe task files (`pose_landmarker_heavy.task` and `pose_landmarker_lite.task`) must be downloaded separately from the [MediaPipe Models](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) page and placed in the `models/` directory before running.