# LipSync2Voice 🎤👄
**Assistive Communication System for Speech-Impaired Individuals**

> A browser-based AI system that detects lip movements via webcam, converts them to text, and speaks the output using the Web Speech API.

---

## 📁 Project Structure

```
lipsync2voice/
├── app.py                   # Flask backend — lip detection + AI model
├── requirements.txt         # Python dependencies
├── templates/
│   └── index.html           # Main UI page
└── static/
    ├── css/
    │   └── style.css        # Full stylesheet (dark clinical theme)
    └── js/
        └── app.js           # Frontend logic — webcam, TTS, UI
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9 or above
- pip (Python package manager)
- A webcam
- Modern browser: Chrome, Edge, or Firefox

---

### Step 1 — Clone / Download the project

```bash
# If you have it as a zip, extract it
# Then navigate into the folder:
cd lipsync2voice
```

---

### Step 2 — Create a virtual environment (recommended)

```bash
# Create venv
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS / Linux:
source venv/bin/activate
```

---

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ MediaPipe may take a couple of minutes to install (it's ~300MB).
> If you're on Apple Silicon (M1/M2), use: `pip install mediapipe-silicon`

---

### Step 4 — Run the Flask server

```bash
python app.py
```

You should see:
```
==================================================
  LipSync2Voice Server Starting...
  Open http://localhost:5000 in your browser
==================================================
 * Running on http://0.0.0.0:5000
```

---

### Step 5 — Open in browser

Open your browser and go to:
```
http://localhost:5000
```

---

## 🎮 How to Use

1. **Click "Enable Camera"** — grant webcam permission when prompted
2. **Position your face** in the camera frame — the system will detect it
3. **Move your lips** — the AI analyzes lip geometry every ~0.8 seconds
4. **Watch the text appear** in the detected text display
5. **Click "Announce"** to hear the text spoken aloud (or press **Spacebar**)
6. **Choose language** from the top-right dropdown for multilingual output
7. **Emergency buttons** — one-click pre-set phrases for urgent situations

### Keyboard Shortcuts
| Key | Action |
|-----|--------|
| `Space` | Announce current text |
| `Ctrl+Shift+C` | Clear text |

---

## 🧠 How the AI Works

### Architecture
```
Webcam (browser)
    ↓  (JPEG frame via HTTP POST)
Flask backend (/analyze)
    ↓
MediaPipe Face Mesh (468 landmarks)
    ↓  (extract lip geometry)
Lip Shape Classifier (rules-based)
    ↓  (open_wide / smile / closed / etc.)
Word Bank → Random weighted prediction
    ↓  (JSON response)
Frontend — display text + optional TTS
```

### Lip Features Extracted
- **Openness Ratio** — vertical gap between upper/lower lip ÷ face width
- **Width Ratio** — horizontal distance between mouth corners ÷ face width

These ratios are classified into 6 shape categories, each mapped to likely words.

### ⚠️ Simulated vs Real Lip Reading
This project uses a **simulated lip-reading model** (word bank + geometric rules) because real neural lip-reading (like LipNet, VSR-CTC) requires:
- Large training datasets (BBC LRS2, GRID corpus)
- GPU training for hours/days
- Complex PyTorch/TensorFlow pipelines

For a **college demo**, the simulation provides realistic behavior. See the "Improvements" section for upgrading to a real model.

---

## 🌐 Multilingual Support

The system uses the browser's **Web Speech API** which supports many languages:

| Language | Code | Works in |
|----------|------|---------|
| English (US) | en-US | All browsers |
| Hindi | hi-IN | Chrome, Edge |
| French | fr-FR | All browsers |
| Spanish | es-ES | All browsers |
| German | de-DE | All browsers |
| Tamil | ta-IN | Chrome |
| Telugu | te-IN | Chrome |
| Arabic | ar-SA | Chrome, Edge |

---

## 🚀 Suggested Improvements (for real accuracy)

### 1. Use a Pre-trained Lip Reading Model
```python
# Option A: LipNet (Keras) — word-level recognition
# pip install lipnet
# Load: model = load_model('lipnet_weights.h5')

# Option B: VSR models from AV-HuBERT
# https://github.com/facebookresearch/av_hubert
```

### 2. Add Temporal Analysis (sequence of frames)
The current system analyzes each frame independently. A better approach:
- Buffer the last 30 frames (~1 second of video)
- Feed the sequence to an LSTM or Transformer model
- This captures motion, not just static shape

### 3. Use TensorFlow.js for On-Device Inference
```javascript
// Move the model entirely to the browser
// No network latency, better privacy
const model = await tf.loadLayersModel('/static/model/lipnet.json');
```

### 4. Use MediaPipe in the Browser (avoid Flask round-trip)
```javascript
// Import @mediapipe/face_mesh directly
// Get landmarks client-side → run JS inference
// Reduces latency to ~50ms
```

### 5. Train on Indian Languages
- MUAVIC dataset includes multilingual AV speech
- Fine-tune on Hindi/Tamil lip datasets for better regional accuracy

---

## 🔒 Privacy

- **No video is stored** — frames are sent to the Flask backend only in memory and immediately discarded after processing
- **No database** — no user data is logged
- **Local processing** — the Flask server runs on your machine
- **Camera permission** — the browser asks for explicit consent

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|---------|
| "Cannot reach backend" | Make sure `python app.py` is running |
| Camera not working | Check browser camera permissions (Settings → Privacy → Camera) |
| "No face detected" | Ensure good lighting, face the camera directly |
| Speech not working | Use Chrome or Edge; Firefox has limited TTS voice support |
| MediaPipe install fails | Try `pip install mediapipe --upgrade` or use Python 3.10 |
| Port 5000 in use | Change `port=5000` in `app.py` to `5001` and visit `http://localhost:5001` |

---

## 📊 Tech Stack Summary

| Layer | Technology |
|-------|-----------|
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Backend | Python 3, Flask |
| Computer Vision | OpenCV, MediaPipe Face Mesh |
| Speech Output | Web Speech API (SpeechSynthesisUtterance) |
| Face Landmarks | MediaPipe 468-point face mesh |
| UI Theme | Dark clinical-futurist design |

---

## 👥 Credits

Built as a college project demonstrating assistive AI technology for speech-impaired individuals.

**PRD Reference:** LipSync2Voice – Assistive Communication System

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.
