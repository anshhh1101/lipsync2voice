# LipSync2Voice

**Assistive Communication System for Speech-Impaired Individuals**

A browser-based AI application that detects lip movements via webcam, converts them to text in real time, and speaks the output using the Web Speech API — enabling hands-free, voice-independent communication.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Multilingual Support](#multilingual-support)
- [How It Works](#how-it-works)
- [Privacy](#privacy)
- [Roadmap](#roadmap)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview

LipSync2Voice is an assistive technology project designed to give a voice to individuals with speech impairments. Using computer vision and AI, the system tracks facial landmarks in real time, classifies lip geometry, and maps movements to natural language — then reads the output aloud via the browser's built-in speech synthesis engine.

> Built as a college project demonstrating the potential of accessible AI technology.

---

## Features

- **Real-time lip detection** using MediaPipe Face Mesh (468 facial landmarks)
- **Automatic text generation** from lip shape classification
- **Text-to-speech output** via the Web Speech API
- **Emergency phrase shortcuts** — one-click pre-set phrases for urgent communication
- **Multilingual support** — 8+ languages including Hindi, Tamil, Arabic, and more
- **Keyboard shortcuts** for hands-free operation
- **Privacy-first design** — no data is stored or transmitted externally

---

## Project Structure

```
lipsync2voice/
├── app.py                   # Flask backend — lip detection & AI model
├── requirements.txt         # Python dependencies
├── templates/
│   └── index.html           # Main UI page
└── static/
    ├── css/
    │   └── style.css        # Stylesheet (dark clinical theme)
    └── js/
        └── app.js           # Frontend logic — webcam, TTS, UI
```

---

## Tech Stack

| Layer            | Technology                          |
|------------------|-------------------------------------|
| Frontend         | HTML5, CSS3, Vanilla JavaScript     |
| Backend          | Python 3, Flask                     |
| Computer Vision  | OpenCV, MediaPipe Face Mesh         |
| Speech Output    | Web Speech API (SpeechSynthesisUtterance) |
| Face Landmarks   | MediaPipe 468-point face mesh       |
| UI Theme         | Dark clinical-futurist design       |

---

## Getting Started

### Prerequisites

- Python 3.9 or above
- pip (Python package manager)
- A working webcam
- A modern browser: Chrome, Edge, or Firefox

### Step 1 — Clone or Download the Project

```bash
cd lipsync2voice
```

### Step 2 — Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Activate — Windows:
venv\Scripts\activate

# Activate — macOS / Linux:
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ MediaPipe (~300 MB) may take a few minutes to install.  
> On Apple Silicon (M1/M2), use: `pip install mediapipe-silicon`

### Step 4 — Start the Server

```bash
python app.py
```

Expected output:
```
==================================================
  LipSync2Voice Server Starting...
  Open http://localhost:5000 in your browser
==================================================
 * Running on http://0.0.0.0:5000
```

### Step 5 — Open in Browser

```
http://localhost:5000
```

---

## Usage

1. Click **"Enable Camera"** and grant webcam permission when prompted
2. **Position your face** in the camera frame — the system will detect it automatically
3. **Move your lips** — the AI analyzes lip geometry approximately every 0.8 seconds
4. The detected text appears in the **text display panel**
5. Click **"Announce"** or press `Spacebar` to hear the text spoken aloud
6. Select a **language** from the top-right dropdown for multilingual speech output
7. Use **Emergency Buttons** for instant one-click urgent phrases

### Keyboard Shortcuts

| Key              | Action              |
|------------------|---------------------|
| `Space`          | Announce current text |
| `Ctrl + Shift + C` | Clear displayed text |

---

## Multilingual Support

The system leverages the browser's **Web Speech API**, which supports a wide range of languages:

| Language     | Code  | Supported Browsers     |
|--------------|-------|------------------------|
| English (US) | en-US | All browsers           |
| Hindi        | hi-IN | Chrome, Edge           |
| French       | fr-FR | All browsers           |
| Spanish      | es-ES | All browsers           |
| German       | de-DE | All browsers           |
| Tamil        | ta-IN | Chrome                 |
| Telugu       | te-IN | Chrome                 |
| Arabic       | ar-SA | Chrome, Edge           |

---

## How It Works

```
Webcam (browser)
    ↓  JPEG frame via HTTP POST
Flask Backend (/analyze)
    ↓
MediaPipe Face Mesh (468 landmarks)
    ↓  Extract lip geometry
Lip Shape Classifier (rules-based)
    ↓  Classify: open_wide / smile / closed / etc.
Word Bank → Weighted random prediction
    ↓  JSON response
Frontend → Display text + optional TTS
```

### Lip Features Extracted

| Feature         | Description                                                |
|-----------------|------------------------------------------------------------|
| Openness Ratio  | Vertical gap between upper/lower lip ÷ face width         |
| Width Ratio     | Horizontal distance between mouth corners ÷ face width    |

These two ratios are mapped to 6 distinct lip shape categories, each associated with a weighted pool of likely words.

### Note on the AI Model

This project uses a **simulated lip-reading model** (geometric rules + word bank). A production-grade neural lip-reading system (e.g., LipNet, AV-HuBERT) requires large labelled datasets, GPU training, and complex deep learning pipelines — beyond the scope of a college demo. See [Roadmap](#roadmap) for upgrade paths.

---

## Privacy

- **No video storage** — frames are processed in-memory and immediately discarded
- **No database** — no user data is logged or persisted
- **Local processing** — the Flask server runs entirely on your machine
- **Explicit consent** — the browser requires camera permission before any access

---

## Roadmap

The following improvements would significantly increase accuracy and usability:

### 1. Integrate a Pre-trained Lip Reading Model

```python
# Option A: LipNet (Keras) — word-level recognition
# pip install lipnet
# model = load_model('lipnet_weights.h5')

# Option B: AV-HuBERT (Facebook Research)
# https://github.com/facebookresearch/av_hubert
```

### 2. Add Temporal Frame Analysis

Buffer 30 consecutive frames (~1 second of video) and pass the sequence to an LSTM or Transformer to capture motion, not just static shape.

### 3. On-Device Inference with TensorFlow.js

```javascript
const model = await tf.loadLayersModel('/static/model/lipnet.json');
// No server round-trip — latency drops to ~50ms
```

### 4. Client-Side MediaPipe

```javascript
// Use @mediapipe/face_mesh directly in the browser
// Eliminates Flask round-trip for landmark extraction
```

### 5. Indian Language Dataset Training

Fine-tune on MUAVIC or Hindi/Tamil lip datasets for improved regional accuracy.

---

## Troubleshooting

| Issue                   | Solution                                                                 |
|-------------------------|--------------------------------------------------------------------------|
| "Cannot reach backend"  | Ensure `python app.py` is running in the terminal                       |
| Camera not working      | Check browser camera permissions (Settings → Privacy → Camera)          |
| "No face detected"      | Ensure good lighting and face the camera directly                       |
| Speech not working      | Use Chrome or Edge; Firefox has limited TTS voice support               |
| MediaPipe install fails | Try `pip install mediapipe --upgrade` or use Python 3.10                |
| Port 5000 already in use | Change `port=5000` to `port=5001` in `app.py`, then visit `localhost:5001` |

---

## License

This project is licensed under the **MIT License** — free to use, modify, and distribute with attribution.

---

*Built as a college project demonstrating assistive AI technology for speech-impaired individuals.*
