# features.py

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# Shared engineered feature builder (TRAINING)
# ─────────────────────────────────────────────
def build_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["neck_forward_combo"] = df["neck_inclination"] * df["head_forward_angle"]
    df["posture_balance"] = df["torso_inclination"] - df["neck_inclination"]
    df["compactness"] = df["neck_ratio"] / (df["torso_height_ratio"] + 1e-6)

    df["alignment_score"] = (
        abs(df["ear_shoulder_y_diff"]) + abs(df["ear_hip_x_dist"])
    )

    df["neck_torso_interaction"] = (
        df["neck_inclination"] * df["torso_inclination"]
    )

    df["forward_vs_tilt"] = (
        df["head_forward_angle"] - df["torso_inclination"]
    )

    return df


# ─────────────────────────────────────────────
# Webcam feature builder (IMPORTANT FIX)
# MUST MATCH training schema exactly
# ─────────────────────────────────────────────
def build_webcam_features(raw):
    """
    raw = dict with keys:
      neck_inc, torso_inc, es_dist, sh_dist, ear, shoulder, hip, angles...
    """

    neck_inc = raw["neck_inc"]
    torso_inc = raw["torso_inc"]

    es_dist = raw["es_dist"]
    sh_dist = raw["sh_dist"]

    ear = raw["ear"]
    shoulder = raw["shoulder"]
    hip = raw["hip"]

    features = {
        # base
        "neck_inclination": neck_inc,
        "torso_inclination": torso_inc,
        "neck_ratio": es_dist / (sh_dist + 1e-6),
        "torso_height_ratio": sh_dist,

        "ear_shoulder_y_diff": shoulder[1] - ear[1],
        "ear_hip_x_dist": abs(ear[0] - hip[0]),
        "head_forward_angle": raw["head_angle"],

        # engineered (MATCH TRAINING)
        "neck_forward_combo": neck_inc * raw["head_angle"],
        "posture_balance": torso_inc - neck_inc,
        "compactness": (es_dist / (sh_dist + 1e-6)) / (sh_dist + 1e-6),
        "alignment_score": abs(shoulder[1] - ear[1]) + abs(ear[0] - hip[0]),
        "neck_torso_interaction": neck_inc * torso_inc,
        "forward_vs_tilt": raw["head_angle"] - torso_inc,
    }

    return features