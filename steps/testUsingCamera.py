import cv2
import numpy as np
import joblib
import mediapipe as mp
import time
from math import atan2, degrees, acos

# ── Load model ─────────────────────────────
model = joblib.load("models/mlp.joblib")

# ── MediaPipe setup ───────────────────────
MODEL_PATH = "models/pose_landmarker_lite.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO
)

landmarker = PoseLandmarker.create_from_options(options)
VISIBILITY_THRESHOLD = 0.3

# ── Geometry helpers ──────────────────────

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

# ── Feature extraction ─────────────────────

def extract_features(kp, img_h, img_w):
    ear      = np.array(kp["ear"])
    shoulder = np.array(kp["shoulder"])
    hip      = np.array(kp["hip"])

    neck_inc  = find_inclination(*shoulder, *ear)
    torso_inc = find_inclination(*hip, *shoulder)

    sh_dist = np.linalg.norm(shoulder - hip) + 1e-6
    es_dist = np.linalg.norm(ear - shoulder)

    features = [
        neck_inc,
        torso_inc,
        es_dist / sh_dist,
        (shoulder[1] - ear[1]) / img_h,
        (shoulder[0] - hip[0]) / img_w,
        three_point_angle(ear, shoulder, hip),
        find_inclination(*hip, *shoulder),
        neck_inc / (torso_inc + 1e-6),
        abs(ear[0] - hip[0]) / img_w,
        (hip[1] - shoulder[1]) / img_h,
    ]

    return np.array(features).reshape(1, -1)

# ── Keypoint smoothing ────────────────────

ALPHA = 0.4  # lower = smoother, higher = more responsive

def smooth_keypoints(new_kp, old_kp, alpha=ALPHA):
    """Exponential moving average between new and old keypoints."""
    if old_kp is None:
        return new_kp
    return {
        k: [
            alpha * new_kp[k][0] + (1 - alpha) * old_kp[k][0],
            alpha * new_kp[k][1] + (1 - alpha) * old_kp[k][1],
        ]
        for k in new_kp
    }

# ── Keypoints (SIDE-AWARE) ────────────────

def get_keypoints(image_rgb, timestamp_ms):
    h, w = image_rgb.shape[:2]

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(image_rgb)
    )

    results = landmarker.detect_for_video(mp_image, timestamp_ms)

    if not results.pose_landmarks:
        return None

    lms = results.pose_landmarks[0]

    def pt(idx):
        lm = lms[idx]
        return {
            "x": lm.x * w,
            "y": lm.y * h,
            "v": lm.visibility
        }

    # LEFT side landmarks
    left = {
        "ear":      pt(7),
        "shoulder": pt(11),
        "hip":      pt(23),
    }

    # RIGHT side landmarks
    right = {
        "ear":      pt(8),
        "shoulder": pt(12),
        "hip":      pt(24),
    }

    def score(side):
        return sum(p["v"] for p in side.values())

    best  = left  if score(left) >= score(right) else right
    side  = "LEFT" if score(left) >= score(right) else "RIGHT"

    # Reject weak detections
    if min(p["v"] for p in best.values()) < VISIBILITY_THRESHOLD:
        return None

    kp = {k: [best[k]["x"], best[k]["y"]] for k in ("ear", "shoulder", "hip")}
    scores = {k: best[k]["v"] for k in ("ear", "shoulder", "hip")}

    return kp, side, scores

# ── Drawing ───────────────────────────────

def draw_visuals(frame, kp):
    for name, p in kp.items():
        cv2.circle(frame, tuple(map(int, p)), 6, (255, 255, 0), -1)
        cv2.putText(frame, name, (int(p[0]) + 5, int(p[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    cv2.line(frame, tuple(map(int, kp["ear"])),
            tuple(map(int, kp["shoulder"])), (255, 255, 0), 2)
    cv2.line(frame, tuple(map(int, kp["shoulder"])),
            tuple(map(int, kp["hip"])), (0, 255, 0), 3)

def draw_overlay(frame, label, color, side, smooth_prob,
                show_warning, fps, debug=False, stale_counter=0):
    cv2.putText(frame, f"{label} ({smooth_prob:.2f})",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    if side:
        cv2.putText(frame, f"Side: {side}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if show_warning:
        cv2.putText(frame, "!! FIX YOUR POSTURE !!",
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.putText(frame, f"FPS: {int(fps)}",
                (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if debug:
        cv2.putText(frame, f"Stale: {stale_counter}",
                    (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

# ── Webcam ────────────────────────────────

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Starting webcam... Press ESC to exit | Press D to toggle debug")

# ── State ─────────────────────────────────
prob_history  = []
pred_history  = []
bad_counter   = 0
last_kp       = None
last_side     = None
frame_count   = 0
stale_counter = 0
MAX_STALE_FRAMES = 5
current_state = 1           # 1 = good, 0 = bad (hysteresis state)
prev_gray     = None
DETECT_EVERY  = 5           # will adapt dynamically
DETECT_MIN    = 2
DETECT_MAX    = 8
debug_mode    = False

# Hysteresis thresholds
GOOD_THRESHOLD = 0.55
BAD_THRESHOLD  = 0.42

start_time = time.monotonic()
prev_time  = time.monotonic()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    frame_count += 1

    # ── Adaptive detection rate based on motion ──
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_gray is not None:
        motion = cv2.absdiff(gray, prev_gray).mean()
        DETECT_EVERY = DETECT_MIN if motion > 1.5 else DETECT_MAX
    prev_gray = gray

    # ── Run MediaPipe every N frames ──
    timestamp_ms = int((time.monotonic() - start_time) * 1000)

    if frame_count % DETECT_EVERY == 0:
        result = get_keypoints(rgb, timestamp_ms)
        if result is not None:
            raw_kp, side, scores = result
            kp        = smooth_keypoints(raw_kp, last_kp)
            last_kp   = kp
            last_side = side
            stale_counter = 0
        else:
            stale_counter += 1
            kp   = last_kp if stale_counter < MAX_STALE_FRAMES else None
            side = last_side
    else:
        kp   = last_kp
        side = last_side

    # ── Prediction ────────────────────────
    if kp is not None and stale_counter < MAX_STALE_FRAMES:
        draw_visuals(frame, kp)

        features = extract_features(kp, h, w)
        prob     = model.predict_proba(features)[0][1]

        # Weighted smoothing — recent frames matter more
        prob_history.append(prob)
        if len(prob_history) > 10:
            prob_history.pop(0)

        weights     = np.linspace(0.5, 1.0, len(prob_history))
        smooth_prob = float(np.average(prob_history, weights=weights))

        # Hysteresis — avoid rapid flipping at the boundary
        if smooth_prob >= GOOD_THRESHOLD:
            current_state = 1
        elif smooth_prob <= BAD_THRESHOLD:
            current_state = 0
        # else: keep current_state unchanged (dead zone)

        # Voting over recent predictions
        pred_history.append(current_state)
        if len(pred_history) > 5:
            pred_history.pop(0)
        final_pred = 1 if sum(pred_history) >= 3 else 0

        # Warning delay
        if final_pred == 0:
            bad_counter += 1
        else:
            bad_counter = 0

        show_warning = bad_counter > 10

        label = "Good Posture" if final_pred == 1 else "Bad Posture"
        color = (0, 255, 0)  if final_pred == 1 else (0, 0, 255)

    else:
        smooth_prob  = 0.0
        show_warning = False
        label        = "No Detection"
        color        = (0, 255, 255)
        side         = last_side

    # ── FPS ───────────────────────────────
    curr_time = time.monotonic()
    fps       = 1.0 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time

    draw_overlay(frame, label, color, side, smooth_prob,
                show_warning, fps, debug=debug_mode,
                stale_counter=stale_counter)

    cv2.imshow("Posture Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:       # ESC — quit
        break
    elif key == ord('d'):
        debug_mode = not debug_mode
        print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")

cap.release()
cv2.destroyAllWindows()