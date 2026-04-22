from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import joblib
import mediapipe as mp
import time
from math import atan2, degrees, acos
import threading

app = Flask(__name__)

# ── Models ─────────────────────────────────────────────
MODELS = {
    "mlp":      joblib.load("models/mlp.joblib"),
    "rf":       joblib.load("models/rf.joblib"),
    "voting":   joblib.load("models/voting.joblib"),
    "stacking": joblib.load("models/stacking.joblib"),
}

current_model = "mlp"

latest_status = "none"
latest_prob = 0.0
latest_side = "none"
draw_kp = True
camera_on = False   # IMPORTANT: start OFF

# ── MediaPipe ─────────────────────────────────────────
MODEL_PATH = "models/pose_landmarker_lite.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
)

landmarker = PoseLandmarker.create_from_options(options)

VISIBILITY_THRESHOLD = 0.3

# ── Camera state (IMPORTANT FIX) ───────────────────────
cap = None
cap_lock = threading.Lock()

# ── Helpers ───────────────────────────────────────────
def find_inclination(x1, y1, x2, y2):
    return degrees(atan2(abs(x2 - x1), abs(y2 - y1)))

def three_point_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos = np.clip(
        np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6),
        -1.0, 1.0,
    )
    return degrees(acos(cos))

def extract_features(kp, h, w):
    ear = np.array(kp["ear"])
    shoulder = np.array(kp["shoulder"])
    hip = np.array(kp["hip"])

    neck_inc = find_inclination(*shoulder, *ear)
    torso_inc = find_inclination(*hip, *shoulder)
    sh_dist = np.linalg.norm(shoulder - hip) + 1e-6
    es_dist = np.linalg.norm(ear - shoulder)

    return np.array([[
        neck_inc,
        torso_inc,
        es_dist / sh_dist,
        (shoulder[1] - ear[1]) / h,
        (shoulder[0] - hip[0]) / w,
        three_point_angle(ear, shoulder, hip),
        find_inclination(*hip, *shoulder),
        neck_inc / (torso_inc + 1e-6),
        abs(ear[0] - hip[0]) / w,
        (hip[1] - shoulder[1]) / h,
    ]])

# ── Camera control (REAL FIX) ─────────────────────────
def start_camera():
    global cap
    with cap_lock:
        if cap is None:
            cap = cv2.VideoCapture(0)

def stop_camera():
    global cap
    with cap_lock:
        if cap is not None:
            cap.release()
            cap = None

# ── Frame generator ───────────────────────────────────
def generate_frames():
    global latest_status, latest_prob, latest_side

    frame_count = 0
    start_time = time.monotonic()

    while True:

        if not camera_on:
            time.sleep(0.1)
            continue

        start_camera()

        with cap_lock:
            ret, frame = cap.read()

        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        frame_count += 1
        timestamp_ms = int((time.monotonic() - start_time) * 1000)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )

        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        label = "No Detection"
        color = (0, 255, 255)

        if results.pose_landmarks:
            lms = results.pose_landmarks[0]

            def pt(i):
                lm = lms[i]
                return [lm.x * w, lm.y * h, lm.visibility]

            kp = {
                "ear": pt(7),
                "shoulder": pt(11),
                "hip": pt(23),
            }

            if min(p[2] for p in kp.values()) > VISIBILITY_THRESHOLD:

                features = extract_features(
                    {k: v[:2] for k, v in kp.items()}, h, w
                )

                model = MODELS[current_model]
                prob = model.predict_proba(features)[0][1]

                latest_prob = float(prob)
                latest_status = "good" if prob > 0.5 else "bad"
                latest_side = "left"

                label = latest_status.upper()
                color = (0, 255, 0) if latest_status == "good" else (0, 0, 255)

        cv2.putText(frame, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        _, buffer = cv2.imencode(".jpg", frame)

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")

# ── Routes ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/toggle_camera", methods=["POST"])
def toggle_camera():
    global camera_on

    camera_on = not camera_on

    if not camera_on:
        stop_camera()

    return jsonify({"camera": camera_on})

@app.route("/toggle_kp", methods=["POST"])
def toggle_kp():
    global draw_kp
    draw_kp = not draw_kp
    return jsonify({"draw_kp": draw_kp})

@app.route("/set_model", methods=["POST"])
def set_model():
    global current_model
    current_model = request.json["model"]
    return jsonify({"ok": True})

@app.route("/stats")
def stats():
    return jsonify({
        "status": latest_status,
        "prob": latest_prob,
        "side": latest_side,
    })

# ── Run ───────────────────────────────────────────────
if __name__ == "__main__":
    app.run(threaded=True, debug=False)