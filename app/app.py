from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import joblib
import mediapipe as mp
import time
from math import atan2, degrees, acos
import threading

app = Flask(__name__)

MODELS = {
    "mlp": joblib.load("models/mlp.joblib"),
    "rf": joblib.load("models/rf.joblib"),
    "voting": joblib.load("models/voting.joblib"),
    "stacking": joblib.load("models/stacking.joblib"),
}

current_model = "mlp"
latest_status = "none"
latest_prob = 0.0
latest_side = "none"
latest_label = "No Detection"
draw_kp = True
camera_on = False

bad_since = None

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

cap = None
cap_lock = threading.Lock()

def find_inclination(x1, y1, x2, y2):
    return degrees(atan2(abs(x2 - x1), abs(y2 - y1)))

def three_point_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos = np.clip(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6), -1.0, 1.0)
    return degrees(acos(cos))

def reset_landmarker():
    global landmarker
    landmarker = PoseLandmarker.create_from_options(options)

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

def generate_frames():
    global latest_status, latest_prob, latest_side, latest_label, bad_since

    while True:
        if not camera_on:
            stop_camera()
            time.sleep(0.1)
            continue

        start_camera()

        with cap_lock:
            ret, frame = cap.read()

        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        timestamp_ms = int(time.time() * 1000)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )

        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        if results.pose_landmarks:
            lms = results.pose_landmarks[0]

            left_vis = lms[7].visibility + lms[11].visibility + lms[23].visibility
            right_vis = lms[8].visibility + lms[12].visibility + lms[24].visibility

            if right_vis >= left_vis:
                kp = {
                    "ear": lms[8],
                    "shoulder": lms[12],
                    "hip": lms[24],
                    "nose": lms[0],
                }
            else:
                kp = {
                    "ear": lms[7],
                    "shoulder": lms[11],
                    "hip": lms[23],
                    "nose": lms[0],
                }

            def get(lm):
                return np.array([lm.x * w, lm.y * h, lm.visibility])

            ear = get(kp["ear"])
            shoulder = get(kp["shoulder"])
            hip = get(kp["hip"])
            nose = get(kp["nose"])

            if min([ear[2], shoulder[2], hip[2], nose[2]]) > VISIBILITY_THRESHOLD:
                features = extract_features(
                    {
                        "ear": ear[:2],
                        "shoulder": shoulder[:2],
                        "hip": hip[:2],
                    },
                    h,
                    w,
                )

                model = MODELS[current_model]
                prob = model.predict_proba(features)[0][1]

                latest_prob = float(prob)
                latest_status = "good" if prob > 0.5 else "bad"
                latest_label = "GOOD POSTURE" if prob > 0.5 else "BAD POSTURE"

                if latest_status == "bad":
                    if bad_since is None:
                        bad_since = time.time()
                else:
                    bad_since = None

                torso_center_x = (shoulder[0] + hip[0]) / 2
                dx = nose[0] - torso_center_x

                if abs(dx) < 15:
                    latest_side = "center"
                elif dx < 0:
                    latest_side = "left"
                else:
                    latest_side = "right"
            else:
                latest_status = "none"
                latest_label = "No Detection"
                bad_since = None

            if draw_kp:
                def pt(lm):
                    return (int(lm.x * w), int(lm.y * h))

                ear_p = pt(kp["ear"])
                shoulder_p = pt(kp["shoulder"])
                hip_p = pt(kp["hip"])

                cv2.circle(frame, ear_p, 6, (0, 255, 255), -1)
                cv2.circle(frame, shoulder_p, 6, (0, 255, 255), -1)
                cv2.circle(frame, hip_p, 6, (0, 255, 255), -1)

                cv2.line(frame, ear_p, shoulder_p, (0, 255, 255), 2)
                cv2.line(frame, shoulder_p, hip_p, (0, 255, 255), 2)
        else:
            latest_status = "none"
            latest_label = "No Detection"
            bad_since = None

        label_color = (127, 255, 110) if latest_status == "good" else (77, 77, 255) if latest_status == "bad" else (0, 255, 255)
        cv2.putText(frame, latest_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)

        _, buffer = cv2.imencode(".jpg", frame)

        yield (b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() + b"\r\n")

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
    else:
        reset_landmarker()
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
    bad_duration = round(time.time() - bad_since, 1) if bad_since is not None else 0
    return jsonify({
        "status": latest_status,
        "prob": latest_prob,
        "side": latest_side,
        "bad_duration": bad_duration,
    })

if __name__ == "__main__":
    app.run(threaded=True, debug=False)