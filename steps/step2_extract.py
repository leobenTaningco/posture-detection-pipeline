"""
Step 2 — MediaPipe Feature Extraction (Side-Aware Version)

Improvements:
- Detects LEFT vs RIGHT facing direction
- Selects correct body side dynamically
- Fixes right-side bias issue in original pipeline
- Keeps backward compatibility with training CSV format
"""

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from math import atan2, degrees, acos
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    AUG_DIR, FEATURES_CSV,
    VISIBILITY_THRESHOLD,
    LABEL_GOOD, LABEL_BAD,
)

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MODEL_PATH = "models/pose_landmarker_heavy.task"

# ── Geometry helpers ────────────────────────────────────────────

def find_inclination(x1, y1, x2, y2):
    return degrees(atan2(abs(x2 - x1), abs(y2 - y1)))

def three_point_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos = np.clip(
        np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6),
        -1.0, 1.0,
    )
    return degrees(acos(cos))

# ── Image transforms ────────────────────────────────────────────

def sharpen(img):
    k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, k)

def enhance_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def upscale(img, factor=1.5):
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_CUBIC)

# ── Label helper ───────────────────────────────────────────────

def label_from_name(name: str):
    lower = name.lower()
    if "goodposture" in lower:
        return LABEL_GOOD
    if "badposture" in lower:
        return LABEL_BAD
    return None

# ── Keypoint extraction + SIDE DETECTION ───────────────────────

def get_keypoints(image_rgb, landmarker):
    h, w = image_rgb.shape[:2]

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(image_rgb)
    )

    results = landmarker.detect(mp_image)

    if not results.pose_landmarks:
        return None

    lms = results.pose_landmarks[0]

    def pt(idx):
        lm = lms[idx]
        if lm.visibility < VISIBILITY_THRESHOLD:
            return None
        return np.array([lm.x * w, lm.y * h]), lm.visibility

    # ── LEFT SIDE ─────────────────────────────
    left_shoulder = pt(11)
    left_ear = pt(7)
    left_hip = pt(23)

    # ── RIGHT SIDE ────────────────────────────
    right_shoulder = pt(12)
    right_ear = pt(8)
    right_hip = pt(24)

    # helper to check full validity
    def valid(points):
        return all(p is not None for p in points)

    # compute visibility scores
    def score(points):
        vals = [p[1] for p in points if p is not None]
        return sum(vals) if vals else 0

    left_points  = [left_ear, left_shoulder, left_hip]
    right_points = [right_ear, right_shoulder, right_hip]

    left_valid  = valid(left_points)
    right_valid = valid(right_points)

    left_score  = score(left_points)
    right_score = score(right_points)

    # ── selection logic (SAFE) ─────────────────────
    if left_valid and right_valid:
        # both sides valid → pick best
        if left_score >= right_score:
            side = "LEFT"
            chosen = left_points
        else:
            side = "RIGHT"
            chosen = right_points

    elif left_valid:
        side = "LEFT"
        chosen = left_points

    elif right_valid:
        side = "RIGHT"
        chosen = right_points

    else:
        return None  # no usable pose

    kp = {
        "ear": chosen[0][0],
        "shoulder": chosen[1][0],
        "hip": chosen[2][0],
    }

    return kp, side
# ── Feature engineering ────────────────────────────────────────

def extract_features(kp, img_h, img_w):
    ear = kp["ear"]
    shoulder = kp["shoulder"]
    hip = kp["hip"]

    neck_inc = find_inclination(*shoulder, *ear)
    torso_inc = find_inclination(*hip, *shoulder)

    sh_dist = np.linalg.norm(shoulder - hip) + 1e-6
    es_dist = np.linalg.norm(ear - shoulder)

    return {
        "neck_inclination": neck_inc,
        "torso_inclination": torso_inc,
        "neck_ratio": es_dist / sh_dist,
        "ear_shoulder_y_diff": (shoulder[1] - ear[1]) / img_h,
        "torso_lean": (shoulder[0] - hip[0]) / img_w,
        "head_forward_angle": three_point_angle(ear, shoulder, hip),
        "shoulder_hip_angle": find_inclination(hip[0], hip[1], shoulder[0], shoulder[1]),
        "neck_torso_ratio": neck_inc / (torso_inc + 1e-6),
        "ear_hip_x_dist": abs(ear[0] - hip[0]) / img_w,
        "torso_height_ratio": (hip[1] - shoulder[1]) / img_h,
    }

# ── Main runner ───────────────────────────────────────────────

def run(source_dir: Path = AUG_DIR):
    images = sorted([p for p in source_dir.rglob("*") if p.suffix.lower() in SUPPORTED])

    if not images:
        print(f"  [WARN] No images found in {source_dir}")
        return

    print(f"  Running MediaPipe on {len(images)} images…")

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE
    )

    landmarker = PoseLandmarker.create_from_options(options)

    records = []
    skipped = 0
    no_label = 0

    for img_path in tqdm(images, desc="Extracting", unit="img"):
        label = label_from_name(img_path.name)
        if label is None:
            no_label += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        img = sharpen(img)
        img = enhance_contrast(img)
        img = upscale(img)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        result = get_keypoints(rgb, landmarker)
        if result is None:
            skipped += 1
            continue

        kp, side = result

        feats = extract_features(kp, h, w)
        feats["label"] = label
        feats["filename"] = img_path.name
        feats["side"] = side

        records.append(feats)

    df = pd.DataFrame(records)
    df.to_csv(FEATURES_CSV, index=False)

    print("\n✅ Feature extraction done.")
    print(f"Records : {len(df)}")
    print(f"Skipped : {skipped}")
    print(f"Saved   : {FEATURES_CSV}")
    print("\nSide distribution:")
    print(df["side"].value_counts())

if __name__ == "__main__":
    run()