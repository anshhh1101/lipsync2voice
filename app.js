/**
 * LipSync2Voice v2 — Frontend
 * Real LSTM model: accumulates 30-frame buffer, gets prediction per sequence.
 */

"use strict";

const state = {
  cameraActive:    false,
  isProcessing:    false,
  currentText:     "",
  selectedLang:    "en-US",
  autoSpeak:       false,
  modelReady:      false,
  bufferProgress:  0,
  bufferNeeded:    30,
  stats: { words: 0, phrases: 0, announced: 0, emergency: 0 },
  sessionStart:    null,
  intervalId:      null,
  lastPrediction:  null,   // avoid repeating same word
  captureMs:       100,    // 10fps → sends frames at 100ms intervals
};

// ── DOM refs ──
const $  = id => document.getElementById(id);

const webcam          = $("webcam");
const overlayCanvas   = $("overlay-canvas");
const overlayCtx      = overlayCanvas.getContext("2d");
const faceChip        = $("face-chip");
const faceChipText    = $("face-chip-text");
const permOverlay     = $("permission-overlay");
const statusBadge     = $("status-badge");
const statusLabel     = $("status-label");
const textDisplay     = $("text-display");
const wordLog         = $("word-log");
const historyList     = $("history-list");
const confMeter       = $("conf-meter");
const openMeter       = $("open-meter");
const confVal         = $("conf-val");
const openVal         = $("open-val");
const processingBar   = $("processing-bar");
const announceBtn     = $("announce-btn");
const clearBtn        = $("clear-btn");
const copyBtn         = $("copy-btn");
const grantBtn        = $("grant-camera-btn");
const toggleCameraBtn = $("toggle-camera-btn");
const snapshotBtn     = $("snapshot-btn");
const autoSpeakToggle = $("auto-speak-toggle");
const langSelect      = $("lang-select");
const fpsVal          = $("fps-value");
const sessionTimer    = $("session-timer");
const toastContainer  = $("toast-container");
const bufferBar       = $("buffer-bar");
const bufferLabel     = $("buffer-label");
const modelStatusBanner = $("model-status-banner");

const statWords     = $("stat-words");
const statPhrases   = $("stat-phrases");
const statAnnounced = $("stat-announced");
const statEmergency = $("stat-emergency");

// ── Background canvas ──
(function initBg() {
  const canvas = $("bg-canvas");
  const ctx    = canvas.getContext("2d");
  let dots     = [];

  function resize() {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
    dots = [];
    for (let x = 0; x < canvas.width; x += 40)
      for (let y = 0; y < canvas.height; y += 40)
        dots.push({ x, y, phase: Math.random() * Math.PI * 2, r: Math.random() * 1.2 + 0.4 });
  }

  function draw(t) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    dots.forEach(d => {
      const a = 0.12 + 0.08 * Math.sin(t * 0.001 + d.phase);
      ctx.beginPath();
      ctx.arc(d.x, d.y, d.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(59,130,246,${a})`;
      ctx.fill();
    });
    requestAnimationFrame(draw);
  }
  window.addEventListener("resize", resize);
  resize();
  requestAnimationFrame(draw);
})();

// ── Toasts ──
function showToast(msg, type = "info", ms = 3000) {
  const t = document.createElement("div");
  t.className = `toast ${type}`;
  t.innerHTML = `<span>${type === "success" ? "✓" : type === "error" ? "✗" : "ℹ"}</span> ${msg}`;
  toastContainer.appendChild(t);
  setTimeout(() => {
    t.style.animation = "toast-out 0.3s ease forwards";
    setTimeout(() => t.remove(), 320);
  }, ms);
}

function setStatus(label, type = "") {
  statusLabel.textContent = label;
  statusBadge.className   = `badge ${type}`;
}

// ── Session timer ──
setInterval(() => {
  if (state.sessionStart)
    sessionTimer.textContent = formatTime(Date.now() - state.sessionStart);
}, 1000);

function formatTime(ms) {
  const s = Math.floor(ms / 1000);
  return `${String(Math.floor(s / 60)).padStart(2,"0")}:${String(s % 60).padStart(2,"0")}`;
}

// ── Buffer progress bar ──
function updateBufferBar(progress, needed) {
  const pct = Math.min(Math.round((progress / needed) * 100), 100);
  if (bufferBar)    bufferBar.style.width = `${pct}%`;
  if (bufferLabel)  bufferLabel.textContent = `Buffer: ${progress}/${needed} frames`;
}

// ── Model status banner ──
async function checkModelStatus() {
  try {
    const r = await fetch("/status");
    const d = await r.json();
    state.modelReady    = d.model_ready;
    state.bufferNeeded  = d.buffer_needed || 30;

    if (modelStatusBanner) {
      if (!d.model_ready) {
        modelStatusBanner.classList.remove("hidden");
        modelStatusBanner.querySelector(".banner-msg").textContent =
          `⚠️ Model not trained yet. ${d.model_error || "Run training/train_model.py first."}`;
      } else {
        modelStatusBanner.classList.add("hidden");
        const wordPills = d.words.map(w =>
          `<span class="word-pill">${w}</span>`).join("");
        if ($("known-words")) $("known-words").innerHTML = wordPills;
      }
    }

    if (d.model_ready) {
      showToast(`Model ready — ${d.words.length} words loaded`, "success", 4000);
    } else {
      showToast("Model not trained. See README.", "error", 7000);
    }
  } catch (e) {
    setStatus("Backend offline", "error");
    showToast("Cannot reach backend. Is Flask running?", "error", 7000);
  }
}

// ── Webcam ──
async function startCamera() {
  try {
    setStatus("Requesting camera…", "warning");
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });

    webcam.srcObject = stream;
    await new Promise(res => webcam.onloadedmetadata = res);

    overlayCanvas.width  = webcam.videoWidth;
    overlayCanvas.height = webcam.videoHeight;

    permOverlay.classList.add("hidden");
    state.cameraActive = true;
    state.sessionStart = Date.now();

    setStatus("Live — buffering…", "active");
    showToast("Camera started! Speak a word clearly.", "success");
    toggleCameraBtn.textContent = "⏹ Stop";

    // Reset backend buffer
    await fetch("/reset_buffer", { method: "POST" });

    startLoop();
  } catch (err) {
    setStatus("Camera denied", "error");
    showToast("Camera denied. Check browser permissions.", "error", 6000);
  }
}

function stopCamera() {
  if (webcam.srcObject)
    webcam.srcObject.getTracks().forEach(t => t.stop());
  webcam.srcObject = null;
  state.cameraActive = false;
  clearInterval(state.intervalId);
  state.intervalId = null;
  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  faceChip.classList.remove("detected");
  faceChipText.textContent = "Camera off";
  setStatus("Stopped", "");
  toggleCameraBtn.textContent = "▶ Start";
  fpsVal.textContent = "—";
  updateBufferBar(0, state.bufferNeeded);
}

// ── Frame capture ──
const captureCanvas = document.createElement("canvas");
const captureCtx    = captureCanvas.getContext("2d");

function captureFrame() {
  if (!state.cameraActive || !webcam.videoWidth) return null;
  captureCanvas.width  = webcam.videoWidth;
  captureCanvas.height = webcam.videoHeight;
  captureCtx.drawImage(webcam, 0, 0);
  return captureCanvas.toDataURL("image/jpeg", 0.65);
}

// ── Processing loop ──
function startLoop() {
  state.intervalId = setInterval(async () => {
    if (!state.cameraActive || state.isProcessing) return;

    const frame = captureFrame();
    if (!frame) return;

    state.isProcessing = true;
    processingBar.classList.add("visible");

    try {
      const res = await fetch("/analyze", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ frame }),
      });
      const data = await res.json();
      handleResult(data);
    } catch (e) {
      // silent fail
    } finally {
      state.isProcessing = false;
      processingBar.classList.remove("visible");
    }
  }, state.captureMs);
}

// ── Result handler ──
function handleResult(d) {
  // Buffer progress
  updateBufferBar(d.buffer_progress || 0, d.buffer_needed || 30);

  if (!d.face_detected) {
    faceChip.classList.remove("detected");
    faceChipText.textContent = "No face detected";
    confMeter.style.width = "0%";
    openMeter.style.width = "0%";
    confVal.textContent   = "—";
    openVal.textContent   = "—";
    drawLipOverlay(false);
    return;
  }

  faceChip.classList.add("detected");
  faceChipText.textContent = "Face detected ✓";

  // Meters
  const confPct = d.confidence ? Math.round(d.confidence * 100) : 0;
  const openPct = d.openness_ratio ? Math.min(Math.round(d.openness_ratio * 500), 100) : 0;
  confMeter.style.width = `${confPct}%`;
  openMeter.style.width = `${openPct}%`;
  confVal.textContent   = `${confPct}%`;
  openVal.textContent   = d.openness_ratio ? d.openness_ratio.toFixed(3) : "—";

  // Model status in UI
  fpsVal.textContent = d.model_ready ? "10" : "—";

  // Predicted word
  if (d.predicted_word && d.predicted_word !== state.lastPrediction) {
    state.lastPrediction = d.predicted_word;
    appendWord(d.predicted_word, d.confidence);

    // Reset buffer after successful prediction for next word
    setTimeout(() => {
      fetch("/reset_buffer", { method: "POST" });
      state.lastPrediction = null;
    }, 2000);
  }

  drawLipOverlay(true, openPct);
}

function appendWord(text, confidence) {
  const sep = state.currentText ? " " : "";
  state.currentText += sep + text;
  textDisplay.textContent = state.currentText;
  textDisplay.classList.add("has-text");
  state.stats.words++;
  statWords.textContent = state.stats.words;

  // Word chip
  const chip = document.createElement("span");
  chip.className = "word-chip fresh";
  chip.textContent = text;
  wordLog.appendChild(chip);
  setTimeout(() => chip.classList.remove("fresh"), 1200);
  while (wordLog.children.length > 12) wordLog.removeChild(wordLog.firstChild);

  addHistory(text, confidence);
  if (state.autoSpeak) speakText(text);
}

// ── TTS ──
function speakText(text) {
  if (!text || !window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const u      = new SpeechSynthesisUtterance(text);
  u.lang       = state.selectedLang;
  u.rate       = 0.95;
  u.onstart    = () => { announceBtn.textContent = "🔊 Speaking…"; announceBtn.disabled = true; };
  u.onend      = () => {
    announceBtn.innerHTML = `<span class="btn-icon">🔊</span> Announce`;
    announceBtn.disabled  = false;
    state.stats.announced++;
    statAnnounced.textContent = state.stats.announced;
  };
  u.onerror    = () => { announceBtn.innerHTML = `<span class="btn-icon">🔊</span> Announce`; announceBtn.disabled = false; };
  window.speechSynthesis.speak(u);
}

// ── History ──
function addHistory(text, confidence) {
  const li = document.createElement("li");
  li.className = "history-item";
  li.innerHTML = `<span>${text}</span><span class="history-conf">${Math.round(confidence * 100)}%</span>`;
  historyList.insertBefore(li, historyList.firstChild);
  while (historyList.children.length > 30) historyList.removeChild(historyList.lastChild);
}

// ── Lip overlay drawing ──
function drawLipOverlay(detected, opennessPercent = 0) {
  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  if (!detected || !state.cameraActive) return;

  const cx = overlayCanvas.width  / 2;
  const cy = overlayCanvas.height * 0.68;
  const rx = overlayCanvas.width  * 0.18;
  const ry = Math.max(overlayCanvas.height * 0.04, overlayCanvas.height * 0.04 + opennessPercent * 0.002);

  const alpha = 0.4 + 0.4 * (opennessPercent / 100);
  overlayCtx.save();
  overlayCtx.strokeStyle = `rgba(59,130,246,${alpha})`;
  overlayCtx.lineWidth   = 1.5;
  overlayCtx.setLineDash([4, 4]);
  overlayCtx.beginPath();
  overlayCtx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
  overlayCtx.stroke();
  overlayCtx.setLineDash([]);
  overlayCtx.fillStyle = `rgba(59,130,246,${alpha})`;
  overlayCtx.font      = "11px DM Mono, monospace";
  overlayCtx.textAlign = "center";
  overlayCtx.fillText("LIP REGION", cx, cy - ry - 8);
  overlayCtx.restore();
}

// ── Event listeners ──
grantBtn.addEventListener("click", startCamera);
toggleCameraBtn.addEventListener("click", () => state.cameraActive ? stopCamera() : startCamera());

announceBtn.addEventListener("click", () => {
  const text = state.currentText.trim();
  if (!text) { showToast("No text to announce.", "info"); return; }
  speakText(text);
});

clearBtn.addEventListener("click", () => {
  state.currentText = "";
  state.lastPrediction = null;
  textDisplay.innerHTML = `<span class="placeholder-text">Start camera and say a word clearly…</span>`;
  textDisplay.classList.remove("has-text");
  wordLog.innerHTML = "";
  fetch("/reset_buffer", { method: "POST" });
  showToast("Cleared.", "info", 1500);
});

copyBtn.addEventListener("click", async () => {
  if (!state.currentText.trim()) { showToast("Nothing to copy.", "info"); return; }
  try {
    await navigator.clipboard.writeText(state.currentText.trim());
    showToast("Copied!", "success", 2000);
  } catch { showToast("Copy failed.", "error"); }
});

autoSpeakToggle.addEventListener("change", () => {
  state.autoSpeak = autoSpeakToggle.checked;
  showToast(`Auto-speak ${state.autoSpeak ? "ON" : "OFF"}`, "info", 1500);
});

langSelect.addEventListener("change", () => {
  state.selectedLang = langSelect.value;
  showToast(`Language: ${langSelect.options[langSelect.selectedIndex].text}`, "info", 1500);
});

snapshotBtn.addEventListener("click", () => {
  if (!state.cameraActive) { showToast("Start camera first.", "info"); return; }
  const f = captureFrame();
  if (!f) return;
  const a = document.createElement("a");
  a.href = f; a.download = `snap-${Date.now()}.jpg`; a.click();
  showToast("Snapshot saved!", "success", 2000);
});

document.querySelectorAll(".emergency-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const phrase = btn.dataset.phrase;
    btn.classList.add("triggered");
    setTimeout(() => btn.classList.remove("triggered"), 900);
    state.currentText = phrase;
    textDisplay.textContent = phrase;
    textDisplay.classList.add("has-text");
    speakText(phrase);
    state.stats.emergency++;
    statEmergency.textContent = state.stats.emergency;
    addHistory(`🚨 ${phrase}`, 1.0);
    showToast(`Emergency: "${phrase}"`, "error", 4000);
  });
});

document.addEventListener("keydown", e => {
  if (e.code === "Space" && e.target === document.body) {
    e.preventDefault();
    if (state.currentText.trim()) speakText(state.currentText);
  }
  if (e.code === "KeyR" && e.ctrlKey) {
    e.preventDefault();
    fetch("/reset_buffer", { method: "POST" });
    state.lastPrediction = null;
    showToast("Buffer reset — say next word", "info", 1500);
  }
});

// ── Init ──
(async function init() {
  setStatus("Checking backend…", "warning");
  await checkModelStatus();
  setStatus(state.modelReady ? "Ready" : "Model missing", state.modelReady ? "" : "error");
})();
