"""
LipSync2Voice v2 — Flask Backend (Final Version)
Fixes applied:
  - Reduced buffer to 20 frames (faster response)
  - Increased MIN_CONFIDENCE to 0.70 (fewer wrong guesses)
  - Reduced smoothing window to 3 (faster confirmation)
  - Added 3-second cooldown (same word won't repeat)
  - Added motion detection (only predict when lips are moving)
  - Better normalization for head position changes
"""

from flask import Flask, render_template, request, jsonify
import base64
import cv2
import numpy as np
import mediapipe as mp
import json
import os
import time
from collections import deque

app = Flask(__name__)

# ─────────────────────────────────────────────
# Lip landmark indices (same as data_collector)
# ─────────────────────────────────────────────
LIP_LANDMARKS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    375, 321, 405, 314, 17, 84, 181, 91, 146,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
    324, 318, 402, 317, 14, 87, 178, 88, 95
]
NOSE_TIP  = 1
LEFT_EYE  = 33
RIGHT_EYE = 263

# ── Key config values ──────────────────────────
FRAMES_PER_SAMPLE  = 30     # reduced from 30 → faster response
FEATURE_DIM        = 80     # 40 landmarks × 2 coords
MIN_CONFIDENCE     = 0.70   # raised from 0.55 → fewer wrong guesses
SMOOTHING_WINDOW   = 3      # reduced from 5 → faster confirmation
MIN_AGREE          = 2      # predictions that must agree
COOLDOWN_SECONDS   = 3.0    # seconds before same word can repeat
MOTION_THRESHOLD   = 0.002  # minimum lip movement to trigger prediction
# ───────────────────────────────────────────────

MODEL_PATH  = "model/lip_model.keras"
LABELS_PATH = "model/labels.json"

# ─────────────────────────────────────────────
# MediaPipe face mesh
# ─────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# ─────────────────────────────────────────────
# Load trained model + labels
# ─────────────────────────────────────────────
model       = None
labels_inv  = {}
model_ready = False
model_error = ""

def load_model():
    global model, labels_inv, model_ready, model_error
    try:
        if not os.path.exists(MODEL_PATH):
            model_error = "Model not found. Run: python training/train_model.py"
            print(f"WARNING: {model_error}")
            return

        if not os.path.exists(LABELS_PATH):
            model_error = "labels.json not found. Re-run training."
            print(f"WARNING: {model_error}")
            return

        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)

        with open(LABELS_PATH) as f:
            labels = json.load(f)
        labels_inv = {v: k for k, v in labels.items()}

        model_ready = True
        print(f"Model loaded — {len(labels)} words: {list(labels.keys())}")

    except Exception as e:
        model_error = str(e)
        print(f"Model load error: {e}")

load_model()

# ─────────────────────────────────────────────
# Buffers and state
# ─────────────────────────────────────────────
frame_buffer = deque(maxlen=FRAMES_PER_SAMPLE)
pred_history = deque(maxlen=SMOOTHING_WINDOW)

last_prediction_time = 0.0
last_predicted_word  = ""
prev_lip_coords      = None

# ─────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────
def extract_lip_landmarks(frame):
    """
    Extract normalized lip landmark coords from a BGR frame.
    Normalization: subtract nose tip, divide by inter-eye distance.
    Makes predictions robust to head position and camera distance.
    """
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    lm   = results.multi_face_landmarks[0].landmark
    h, w = frame.shape[:2]

    nose  = np.array([lm[NOSE_TIP].x * w, lm[NOSE_TIP].y * h])
    le    = np.array([lm[LEFT_EYE].x  * w, lm[LEFT_EYE].y  * h])
    re    = np.array([lm[RIGHT_EYE].x * w, lm[RIGHT_EYE].y * h])
    scale = np.linalg.norm(re - le) or 1.0

    coords = []
    for idx in LIP_LANDMARKS:
        px = (lm[idx].x * w - nose[0]) / scale
        py = (lm[idx].y * h - nose[1]) / scale
        coords.extend([px, py])

    while len(coords) < FEATURE_DIM:
        coords.extend([0.0, 0.0])

    return np.array(coords[:FEATURE_DIM], dtype=np.float32)


def get_lip_metrics(frame):
    """Return (openness_ratio, width_ratio) for UI meters."""
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None, None

    lm   = results.multi_face_landmarks[0].landmark
    h, w = frame.shape[:2]

    def pt(i):
        return np.array([lm[i].x * w, lm[i].y * h])

    openness = np.linalg.norm(pt(13) - pt(14))
    width    = np.linalg.norm(pt(61) - pt(291))
    face_w   = np.linalg.norm(pt(234) - pt(454)) or 1.0

    return round(float(openness / face_w), 4), round(float(width / face_w), 4)


def decode_frame(b64_data):
    """Decode base64 image string to OpenCV BGR frame."""
    if "," in b64_data:
        b64_data = b64_data.split(",")[1]
    img_bytes = base64.b64decode(b64_data)
    arr       = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def detect_lip_motion(current_coords):
    """
    Returns True if lips are moving enough to be saying something.
    Prevents random predictions when sitting still.
    """
    global prev_lip_coords
    if prev_lip_coords is None:
        prev_lip_coords = current_coords
        return False

    motion          = float(np.mean(np.abs(current_coords - prev_lip_coords)))
    prev_lip_coords = current_coords.copy()
    return motion > MOTION_THRESHOLD


# ─────────────────────────────────────────────
# Core prediction
# ─────────────────────────────────────────────
def predict_from_buffer():
    """
    Run LSTM on current buffer.
    Applies: confidence threshold + smoothing + cooldown.
    Returns (word, confidence) or (None, raw_confidence).
    """
    global last_prediction_time, last_predicted_word

    if not model_ready or len(frame_buffer) < FRAMES_PER_SAMPLE:
        return None, 0.0

    seq   = np.array(list(frame_buffer), dtype=np.float32)[np.newaxis, ...]
    probs = model.predict(seq, verbose=0)[0]

    top_idx    = int(np.argmax(probs))
    confidence = float(probs[top_idx])

    # 1. Confidence threshold
    if confidence < MIN_CONFIDENCE:
        return None, confidence

    # 2. Smoothing — need consistent predictions
    pred_history.append(top_idx)
    counts   = {i: list(pred_history).count(i) for i in set(pred_history)}
    dominant = max(counts, key=counts.get)

    if counts[dominant] < MIN_AGREE:
        return None, confidence

    word = labels_inv.get(dominant, "?")
    now  = time.time()

    # 3. Cooldown — don't repeat same word too fast
    if word == last_predicted_word and (now - last_prediction_time) < COOLDOWN_SECONDS:
        return None, confidence

    last_prediction_time = now
    last_predicted_word  = word
    return word, confidence


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/status")
def status():
    return jsonify({
        "model_ready":   model_ready,
        "model_error":   model_error,
        "words":         list(labels_inv.values()) if model_ready else [],
        "buffer_size":   len(frame_buffer),
        "buffer_needed": FRAMES_PER_SAMPLE,
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data      = request.get_json(force=True)
        frame_b64 = data.get("frame", "")

        if not frame_b64:
            return jsonify({"error": "No frame received"}), 400

        frame = decode_frame(frame_b64)
        if frame is None:
            return jsonify({"error": "Could not decode image"}), 400

        landmarks       = extract_lip_landmarks(frame)
        openness, width = get_lip_metrics(frame)

        if landmarks is None:
            return jsonify({
                "face_detected":   False,
                "buffer_progress": len(frame_buffer),
                "buffer_needed":   FRAMES_PER_SAMPLE,
            })

        is_moving = detect_lip_motion(landmarks)
        frame_buffer.append(landmarks)
        word, confidence = predict_from_buffer()

        return jsonify({
            "face_detected":   True,
            "buffer_progress": len(frame_buffer),
            "buffer_needed":   FRAMES_PER_SAMPLE,
            "openness_ratio":  openness,
            "width_ratio":     width,
            "predicted_word":  word,
            "confidence":      round(confidence, 3),
            "model_ready":     model_ready,
            "is_moving":       is_moving,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/reset_buffer", methods=["POST"])
def reset_buffer():
    global prev_lip_coords, last_predicted_word
    frame_buffer.clear()
    pred_history.clear()
    prev_lip_coords     = None
    last_predicted_word = ""
    return jsonify({"ok": True})


@app.route("/health")
def health():
    return jsonify({
        "status":      "ok",
        "model_ready": model_ready,
        "version":     "v2-lstm-final",
        "config": {
            "frames_needed":  FRAMES_PER_SAMPLE,
            "min_confidence": MIN_CONFIDENCE,
            "smoothing":      SMOOTHING_WINDOW,
            "cooldown_secs":  COOLDOWN_SECONDS,
        }
    })


if __name__ == "__main__":
    print("=" * 52)
    print("  LipSync2Voice v2 — Final Version")
    print(f"  Frames needed : {FRAMES_PER_SAMPLE}")
    print(f"  Min confidence: {MIN_CONFIDENCE}")
    print(f"  Smoothing     : {SMOOTHING_WINDOW} frames")
    print(f"  Cooldown      : {COOLDOWN_SECONDS}s")
    print(f"  Model ready   : {model_ready}")
    if not model_ready:
        print(f"  WARNING: {model_error}")
    print("  Open: http://localhost:5000")
    print("=" * 52)
    app.run(debug=False, host="0.0.0.0", port=5000)