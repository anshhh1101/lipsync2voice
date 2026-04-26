"""
STEP 1: data_collector.py
─────────────────────────
Records lip landmark sequences from your webcam for each word.
Run this FIRST to build your personal training dataset.

Usage:
    python data_collector.py

Controls:
    SPACE  → start/stop recording a sample
    N      → next word
    Q      → quit and save
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time

# ── Configuration ──────────────────────────────────────────────
WORDS = [
    "hello", "help", "water", "food", "pain",
    "yes", "no", "please", "doctor", "family",
    "okay", "thankyou", "sorry", "more", "stop"
]
SAMPLES_PER_WORD = 20       # how many recordings per word
FRAMES_PER_SAMPLE = 30      # frames captured per recording (~1 second at 30fps)
OUTPUT_FILE = "training/dataset.json"
# ───────────────────────────────────────────────────────────────

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils
face_mesh    = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Lip landmark indices (MediaPipe 478-point with irises)
LIP_LANDMARKS = [
    # Outer lip
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    375, 321, 405, 314, 17, 84, 181, 91, 146,
    # Inner lip
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
    324, 318, 402, 317, 14, 87, 178, 88, 95
]

# Key anchor points for normalization
NOSE_TIP   = 1
LEFT_EYE   = 33
RIGHT_EYE  = 263


def extract_lip_landmarks(frame, face_mesh_model):
    """
    Extract normalized lip landmark coordinates from a frame.
    Returns a flat numpy array of shape (len(LIP_LANDMARKS)*2,) or None.
    Coordinates are normalized relative to face bounding box so
    position/distance from camera doesn't affect the features.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh_model.process(rgb)

    if not results.multi_face_landmarks:
        return None

    lm = results.multi_face_landmarks[0].landmark
    h, w = frame.shape[:2]

    # Anchor: nose tip for translation normalization
    nose = np.array([lm[NOSE_TIP].x * w, lm[NOSE_TIP].y * h])

    # Scale: inter-eye distance for scale normalization
    le   = np.array([lm[LEFT_EYE].x  * w, lm[LEFT_EYE].y  * h])
    re   = np.array([lm[RIGHT_EYE].x * w, lm[RIGHT_EYE].y * h])
    scale = np.linalg.norm(re - le) or 1.0

    # Extract and normalize lip points
    coords = []
    for idx in LIP_LANDMARKS:
        px = (lm[idx].x * w - nose[0]) / scale
        py = (lm[idx].y * h - nose[1]) / scale
        coords.extend([px, py])

    return np.array(coords, dtype=np.float32)


def draw_ui(frame, word, word_idx, sample_idx, recording, countdown, buffer_len):
    """Draw overlay UI on the camera frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent top bar
    cv2.rectangle(overlay, (0, 0), (w, 80), (15, 20, 30), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    # Word to say
    cv2.putText(frame, f'Say: "{word.upper()}"', (15, 35),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (100, 200, 255), 2)

    # Progress
    progress = f"Word {word_idx+1}/{len(WORDS)}  |  Sample {sample_idx}/{SAMPLES_PER_WORD}"
    cv2.putText(frame, progress, (15, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 160), 1)

    # Recording indicator
    if recording:
        cv2.circle(frame, (w - 30, 30), 12, (50, 50, 240), -1)
        cv2.putText(frame, f"REC {buffer_len}/{FRAMES_PER_SAMPLE}", (w - 140, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 255), 1)
    else:
        cv2.putText(frame, "SPACE=Record  N=Next  Q=Quit", (15, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 120), 1)

    # Countdown overlay
    if countdown > 0:
        cv2.putText(frame, str(countdown), (w//2 - 20, h//2),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (50, 200, 100), 4)

    return frame


def collect_data():
    os.makedirs("training", exist_ok=True)

    # Load existing dataset if present
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            dataset = json.load(f)
        print(f"Loaded existing dataset with {sum(len(v) for v in dataset.values())} samples")
    else:
        dataset = {word: [] for word in WORDS}

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    word_idx    = 0
    recording   = False
    frame_buffer = []
    countdown   = 0

    print("\n" + "="*50)
    print("  LipSync2Voice — Data Collector")
    print("="*50)
    print(f"  Words to record: {WORDS}")
    print(f"  {SAMPLES_PER_WORD} samples per word × {FRAMES_PER_SAMPLE} frames")
    print("="*50)
    print("  Controls: SPACE=Record  N=Next word  Q=Save & quit")
    print("="*50 + "\n")

    while word_idx < len(WORDS):
        word        = WORDS[word_idx]
        sample_idx  = len(dataset.get(word, []))

        if sample_idx >= SAMPLES_PER_WORD:
            print(f"  ✓ '{word}' complete — moving to next")
            word_idx += 1
            continue

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror

        # Run face mesh for live preview
        landmarks = extract_lip_landmarks(frame, face_mesh)

        # Draw lip dots on preview
        if landmarks is not None:
            h, w = frame.shape[:2]
            nose  = np.array([face_mesh.process(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )])  # skip redraw; just show green dot
            for i in range(0, len(landmarks), 2):
                # Landmarks are normalized — skip re-drawing here for speed

                pass
            cv2.putText(frame, "FACE OK", (frame.shape[1]-100, frame.shape[0]-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 220, 50), 1)
        else:
            cv2.putText(frame, "NO FACE", (frame.shape[1]-100, frame.shape[0]-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 220), 1)

        frame = draw_ui(frame, word, word_idx, sample_idx,
                        recording, countdown, len(frame_buffer))

        # Auto-capture frames when recording
        if recording:
            if landmarks is not None:
                frame_buffer.append(landmarks.tolist())

            if len(frame_buffer) >= FRAMES_PER_SAMPLE:
                # Save sample
                if word not in dataset:
                    dataset[word] = []
                dataset[word].append(frame_buffer)
                print(f"  ✓ Saved sample {len(dataset[word])}/{SAMPLES_PER_WORD} for '{word}'")
                frame_buffer = []
                recording    = False

                # Auto-save after each sample
                with open(OUTPUT_FILE, "w") as f:
                    json.dump(dataset, f)

        cv2.imshow("LipSync2Voice — Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # Start/stop recording
            if not recording:
                frame_buffer = []
                recording    = True
                print(f"  ● Recording '{word}'...")
            else:
                recording    = False
                frame_buffer = []
                print("  ■ Recording cancelled")

        elif key == ord('n') or key == ord('N'):
            word_idx += 1
            recording    = False
            frame_buffer = []
            if word_idx < len(WORDS):
                print(f"\n→ Switching to: '{WORDS[word_idx]}'")

        elif key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Final save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(dataset, f)

    total = sum(len(v) for v in dataset.values())
    print(f"\n✅ Dataset saved to '{OUTPUT_FILE}'")
    print(f"   Total samples: {total}")
    for w, samples in dataset.items():
        print(f"   {w:12s}: {len(samples):2d} samples")


if __name__ == "__main__":
    collect_data()
