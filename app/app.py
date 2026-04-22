from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import joblib
import mediapipe as mp
import time
from math import atan2, degrees, acos

app = Flask(__name__)

MODELS = {
    "mlp":     joblib.load("models/mlp.joblib"),
    "rf":      joblib.load("models/rf.joblib"),
    "voting":  joblib.load("models/voting.joblib"),
    "stacking":joblib.load("models/stacking.joblib"),
}

current_model  = "mlp"
latest_status  = "none"
latest_prob    = 0.0
latest_side    = "none"
draw_kp        = True
camera_on      = True

MODEL_PATH = "models/pose_landmarker_lite.task"

BaseOptions        = mp.tasks.BaseOptions
PoseLandmarker     = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode  = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
)

landmarker = PoseLandmarker.create_from_options(options)
VISIBILITY_THRESHOLD = 0.3


def find_inclination(x1, y1, x2, y2):
    return degrees(atan2(abs(x2 - x1), abs(y2 - y1)))


def three_point_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc  = a - b, c - b
    cos = np.clip(
        np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6),
        -1.0, 1.0,
    )
    return degrees(acos(cos))


def extract_features(kp, h, w):
    ear      = np.array(kp["ear"])
    shoulder = np.array(kp["shoulder"])
    hip      = np.array(kp["hip"])

    neck_inc  = find_inclination(*shoulder, *ear)
    torso_inc = find_inclination(*hip, *shoulder)
    sh_dist   = np.linalg.norm(shoulder - hip) + 1e-6
    es_dist   = np.linalg.norm(ear - shoulder)

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


def get_keypoints(image_rgb, timestamp_ms):
    h, w = image_rgb.shape[:2]

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(image_rgb),
    )
    results = landmarker.detect_for_video(mp_image, timestamp_ms)

    if not results.pose_landmarks:
        return None

    lms = results.pose_landmarks[0]

    def pt(idx):
        lm = lms[idx]
        return {"x": lm.x * w, "y": lm.y * h, "v": lm.visibility}

    left  = {"ear": pt(7),  "shoulder": pt(11), "hip": pt(23)}
    right = {"ear": pt(8),  "shoulder": pt(12), "hip": pt(24)}

    def score(side):
        return sum(p["v"] for p in side.values())

    if score(left) >= score(right):
        best, side = left, "left"
    else:
        best, side = right, "right"

    if min(p["v"] for p in best.values()) < VISIBILITY_THRESHOLD:
        return None

    return {
        "points": {k: [best[k]["x"], best[k]["y"]] for k in ("ear", "shoulder", "hip")},
        "side": side,
    }


ALPHA    = 0.4
last_kp  = None


def smooth_keypoints(new_kp):
    global last_kp
    if last_kp is None:
        last_kp = new_kp
        return new_kp
    smoothed = {
        k: [
            ALPHA * new_kp[k][0] + (1 - ALPHA) * last_kp[k][0],
            ALPHA * new_kp[k][1] + (1 - ALPHA) * last_kp[k][1],
        ]
        for k in new_kp
    }
    last_kp = smoothed
    return smoothed


def draw_keypoints_on_frame(frame, kp):
    """Draw ear → shoulder → hip skeleton on the frame."""
    pts = {k: (int(v[0]), int(v[1])) for k, v in kp.items()}
    color_line = (0, 229, 255)
    color_dot  = (255, 255, 255)
    # Lines
    cv2.line(frame, pts["ear"],      pts["shoulder"], color_line, 2)
    cv2.line(frame, pts["shoulder"], pts["hip"],      color_line, 2)
    # Dots
    for pt in pts.values():
        cv2.circle(frame, pt, 5, color_dot, -1)
        cv2.circle(frame, pt, 5, color_line, 1)


def generate_frames():
    global current_model, latest_status, latest_prob, latest_side, last_kp

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    start_time    = time.monotonic()
    frame_count   = 0
    stale_counter = 0
    MAX_STALE     = 5
    DETECT_EVERY  = 3

    prob_history  = []
    pred_history  = []
    current_state = 1
    GOOD_T, BAD_T = 0.55, 0.42

    while True:
        if not camera_on:
            black = np.zeros((480, 640, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.jpg', black)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret:
            break

        rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w         = rgb.shape[:2]
        frame_count += 1
        timestamp_ms = int((time.monotonic() - start_time) * 1000)

        if frame_count % DETECT_EVERY == 0:
            result = get_keypoints(rgb, timestamp_ms)
            if result:
                kp            = smooth_keypoints(result["points"])
                latest_side   = result["side"]
                stale_counter = 0
            else:
                kp             = last_kp
                stale_counter += 1
        else:
            kp = last_kp

        label = "No Detection"
        color = (0, 255, 255)

        if kp and stale_counter < MAX_STALE:
            model    = MODELS[current_model]
            features = extract_features(kp, h, w)
            prob     = model.predict_proba(features)[0][1]

            prob_history.append(prob)
            if len(prob_history) > 10:
                prob_history.pop(0)

            weights     = np.linspace(0.5, 1.0, len(prob_history))
            smooth_prob = float(np.average(prob_history, weights=weights))

            if smooth_prob >= GOOD_T:
                current_state = 1
            elif smooth_prob <= BAD_T:
                current_state = 0

            pred_history.append(current_state)
            if len(pred_history) > 5:
                pred_history.pop(0)

            final_pred     = 1 if sum(pred_history) >= 3 else 0
            latest_prob    = smooth_prob
            latest_status  = "good" if final_pred == 1 else "bad"

            label = "Good" if final_pred == 1 else "Bad"
            color = (0, 255, 0) if final_pred == 1 else (0, 0, 255)

            # Draw keypoints if enabled
            if draw_kp:
                draw_keypoints_on_frame(frame, kp)
        else:
            latest_status = "none"
            latest_prob   = 0.0
            latest_side   = "none"

        cv2.putText(frame, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/set_model', methods=['POST'])
def set_model():
    global current_model
    current_model = request.json["model"]
    return jsonify({"ok": True})


@app.route('/toggle_kp', methods=['POST'])
def toggle_kp():
    global draw_kp
    draw_kp = not draw_kp
    # Return the new state so the frontend can reflect it
    return jsonify({"draw_kp": draw_kp})


@app.route('/toggle_camera', methods=['POST'])
def toggle_camera():
    global camera_on
    camera_on = not camera_on
    return jsonify({"camera": camera_on})


@app.route('/stats')
def stats():
    return jsonify({
        "status": latest_status,
        "prob":   latest_prob,
        "side":   latest_side,
    })


if __name__ == "__main__":
    app.run(threaded=True, debug=False)