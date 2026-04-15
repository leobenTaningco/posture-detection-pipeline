import cv2
import numpy as np
import joblib
import mediapipe as mp
import time
from math import atan2, degrees, acos

# ── Load model ─────────────────────────────
model = joblib.load("models/voting.joblib")

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
VISIBILITY_THRESHOLD = 0.5

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
    ear = np.array(kp["ear"])
    shoulder = np.array(kp["shoulder"])
    hip = np.array(kp["hip"])

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

# ── Keypoints (SIDE-AWARE) ────────────────

def get_keypoints(image_rgb):
    h, w = image_rgb.shape[:2]

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(image_rgb)
    )

    timestamp = int(time.time() * 1000)
    results = landmarker.detect_for_video(mp_image, timestamp)

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

    # LEFT
    left = {
        "ear": pt(7),
        "shoulder": pt(11),
        "hip": pt(23),
    }

    # RIGHT
    right = {
        "ear": pt(8),
        "shoulder": pt(12),
        "hip": pt(24),
    }

    def score(side):
        return sum(p["v"] for p in side.values())

    left_score = score(left)
    right_score = score(right)

    if left_score >= right_score:
        best = left
        side = "LEFT"
    else:
        best = right
        side = "RIGHT"

    # reject weak detections
    if min(p["v"] for p in best.values()) < VISIBILITY_THRESHOLD:
        return None

    kp = {
        "ear": [best["ear"]["x"], best["ear"]["y"]],
        "shoulder": [best["shoulder"]["x"], best["shoulder"]["y"]],
        "hip": [best["hip"]["x"], best["hip"]["y"]],
    }

    return kp, side

# ── Drawing ───────────────────────────────

def draw_visuals(frame, kp):
    for name, p in kp.items():
        cv2.circle(frame, tuple(map(int, p)), 6, (255,255,0), -1)
        cv2.putText(frame, name, (int(p[0])+5, int(p[1])-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

    cv2.line(frame, tuple(map(int, kp["ear"])),
            tuple(map(int, kp["shoulder"])), (255,255,0), 2)

    cv2.line(frame, tuple(map(int, kp["shoulder"])),
            tuple(map(int, kp["hip"])), (0,255,0), 3)

# ── Webcam ────────────────────────────────

cap = cv2.VideoCapture(0)

# reduce resolution (performance boost)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("🎥 Starting webcam... Press ESC to exit")

prob_history = []
pred_history = []
bad_counter = 0

last_kp = None
last_side = None

frame_count = 0
DETECT_EVERY = 5

prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    frame_count += 1

    # ── Run MediaPipe only every N frames ──
    if frame_count % DETECT_EVERY == 0:
        result = get_keypoints(rgb)
        if result is not None:
            kp, side = result
            last_kp = kp
            last_side = side
        else:
            kp = last_kp
            side = last_side
    else:
        kp = last_kp
        side = last_side

    if kp is not None:
        draw_visuals(frame, kp)

        features = extract_features(kp, h, w)
        prob = model.predict_proba(features)[0][1]

        # ── smoothing ──
        prob_history.append(prob)
        if len(prob_history) > 10:
            prob_history.pop(0)

        smooth_prob = np.mean(prob_history)

        # neutral zone
        if 0.45 < smooth_prob < 0.55:
            pred = None
        else:
            pred = 1 if smooth_prob >= 0.5 else 0

        # voting
        if pred is not None:
            pred_history.append(pred)
            if len(pred_history) > 5:
                pred_history.pop(0)

            final_pred = 1 if sum(pred_history) >= 3 else 0
        else:
            final_pred = None

        # warning delay
        if final_pred == 0:
            bad_counter += 1
        else:
            bad_counter = 0

        show_warning = bad_counter > 10

        # ── Display ──
        if final_pred is None:
            label = "Adjusting..."
            color = (0,255,255)
        else:
            label = "Good Posture" if final_pred == 1 else "Bad Posture"
            color = (0,255,0) if final_pred == 1 else (0,0,255)

        cv2.putText(frame, f"{label} ({smooth_prob:.2f})",
                    (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # show side
        if side is not None:
            cv2.putText(frame, f"Side: {side}",
                        (20,80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255,255,255), 2)

        if show_warning:
            cv2.putText(frame, "⚠ FIX YOUR POSTURE!",
                        (20,120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    # ── FPS ──
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}",
                (20,160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Posture Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()