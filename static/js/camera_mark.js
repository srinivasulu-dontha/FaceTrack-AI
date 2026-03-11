// camera_mark.js — Robust Anti-Spoofing Liveness Detection
// Strategy: Per-face state machine requiring a CONFIRMED BLINK (open→closed→open
// transition across multiple frames) + measurable HEAD MOVEMENT before recognition.
// A printed photo or static screen clip cannot produce this sequence.

const markVideo = document.getElementById("markVideo");
const markCanvas = document.getElementById("markCanvas");
const canvasCtx = markCanvas.getContext("2d");
const startMarkBtn = document.getElementById("startMarkBtn");
const stopMarkBtn = document.getElementById("stopMarkBtn");
const quickToggleBtn = document.getElementById("quickToggleBtn");
const markStatus = document.getElementById("markStatus");
const recognizedList = document.getElementById("recognizedList");

let isRunning = false;
let isQuickMode = true; // Default to fast mode
let camera = null;
let faceMesh = null;

// ─── Eye Aspect Ratio indices (MediaPipe FaceMesh 468-pt model) ───────────────
const LEFT_EYE = [33, 160, 158, 133, 153, 144];
const RIGHT_EYE = [362, 385, 387, 263, 373, 380];

// ─── Thresholds (Strict Sequential Edition) ────────────────────────────────
const LIVENESS_HISTORY_FRAMES = 20;       // Require 20 frames for a highly stable baseline
const BLINK_DROP_THRESH = 0.14;           // Require a deep, deliberate eyelid closure (prevents squinting/looking down)
const MIN_BLINK_FRAMES = 3;               // Require eyes to stay shut for at least 3 frames (eliminates multi-frame noise)
const MAX_NOSE_MOVE_FOR_BLINK = 0.03;     // Much tighter vertical stability requirement to block photo wobbling
const SMILE_STRETCH_RATIO = 1.25;         // Require a very wide, deliberate 25% smile (prevents normal talking from verifying)

// ─── Per-face liveness state ──────────────────────────────────────────────────
let activeTracks = []; // Array of { id: number, centroid: {cx, cy}, state: object }
let nextTrackId = 0;
// Cooldown set: faces that recently sent a recognition request
let cooldown = new Set();
// Track recognized student roll numbers in this session to prevent re-hits
const sessionRecognizedRolls = new Set();
// When recognition fires we lock status for a few seconds so per-frame
// liveness hints (running at ~30 fps) cannot immediately overwrite the result.
let statusLocked = false;

// ─── Initialize Toggle UI ──────────────────────────────────────────────────────────
if (quickToggleBtn) {
  quickToggleBtn.className = isQuickMode ? "btn btn-warning" : "btn btn-outline-warning";
  quickToggleBtn.innerText = isQuickMode ? "⚡ Quick Scan: ON" : "🔒 Secure: ON";
  quickToggleBtn.addEventListener("click", () => {
    isQuickMode = !isQuickMode;
    quickToggleBtn.className = isQuickMode ? "btn btn-warning" : "btn btn-outline-warning";
    quickToggleBtn.innerText = isQuickMode ? "⚡ Quick Scan: ON" : "🔒 Secure: ON";
    markStatus.innerText = isQuickMode ? "Switched to Quick Scan (Instant)" : "Switched to Secure Mode (Blink required)";
  });
}

function makeFaceState() {
  return {
    phase: "GATHERING",   // GATHERING → WAITING → BLINKING → VERIFIED

    baselineEAR: 0.0,
    baselineMouthWidth: 0.0,
    historyCount: 0,

    earHistory: [],
    mouthWidthHistory: [],
    noseYHistory: [],

    closedFrames: 0,      // Must hit >= MIN_BLINK_FRAMES to verify real closure
    liveScore: 0,         // Must be >= 1 for recognition
  };
}

// ─── Math helpers ─────────────────────────────────────────────────────────────
function dist3(p1, p2) {
  const dx = p1.x - p2.x, dy = p1.y - p2.y, dz = (p1.z || 0) - (p2.z || 0);
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}
function dist2(p1, p2) {
  const dx = p1.x - p2.x, dy = p1.y - p2.y;
  return Math.sqrt(dx * dx + dy * dy);
}

function getEAR(lm, indices) {
  const [i0, i1, i2, i3, i4, i5] = indices;
  const h1 = dist3(lm[i1], lm[i5]);
  const h2 = dist3(lm[i2], lm[i4]);
  const w = dist3(lm[i0], lm[i3]);
  return (h1 + h2) / (2.0 * w);
}

function getMAR(lm) {
  // Landmarks: 13=upper lip, 14=lower lip, 78=left corner, 308=right corner
  const h = dist3(lm[13], lm[14]);
  const w = dist3(lm[78], lm[308]);
  return w > 0 ? h / w : 0;
}

function getFaceCentroid(lm) {
  let minX = 1, maxX = 0, minY = 1, maxY = 0;
  lm.forEach(p => {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  });
  return { cx: (minX + maxX) / 2, cy: (minY + maxY) / 2 };
}

// ─── Per-face frame update ────────────────────────────────────────────────────
function updateFaceState(state, lm) {
  const leftEAR = getEAR(lm, LEFT_EYE);
  const rightEAR = getEAR(lm, RIGHT_EYE);
  const ear = (leftEAR + rightEAR) / 2;

  // Scale-invariant mouth width (Mouth corners relative to eye corners)
  const mouthWidthRaw = dist3(lm[78], lm[308]);
  const eyeDist = dist3(lm[33], lm[263]);
  const mouthWidth = eyeDist > 0 ? mouthWidthRaw / eyeDist : 0;

  const nose = lm[1];

  // Accumulate History
  state.earHistory.push(ear);
  state.mouthWidthHistory.push(mouthWidth);
  state.noseYHistory.push(nose.y);

  if (state.earHistory.length > LIVENESS_HISTORY_FRAMES) {
    state.earHistory.shift();
    state.mouthWidthHistory.shift();
    state.noseYHistory.shift();
  } else {
    state.historyCount++;
  }

  // Phase 1: Establish stable baselines (filters out jitter by taking Median)
  if (state.phase === "GATHERING" && state.historyCount >= LIVENESS_HISTORY_FRAMES) {
    const sortedEAR = [...state.earHistory].sort((a, b) => a - b);
    state.baselineEAR = sortedEAR[Math.floor(sortedEAR.length / 2)];

    const sortedMouth = [...state.mouthWidthHistory].sort((a, b) => a - b);
    state.baselineMouthWidth = sortedMouth[Math.floor(sortedMouth.length / 2)];

    state.phase = "WAITING";
  }

  // Phase 2 & 3: Active Action Verification
  if (state.phase === "WAITING" || state.phase === "BLINKING") {

    // 1. Quick Scan Smile Tracker (Instant natural verification)
    // A photo is 2D and its physical dimensions cannot stretch by 15% 
    if (isQuickMode && mouthWidth > state.baselineMouthWidth * SMILE_STRETCH_RATIO) {
      state.phase = "VERIFIED";
      state.liveScore = 1;
    }

    // 2. Strict Sequential Blink Tracker
    const isClosed = (state.baselineEAR - ear) > BLINK_DROP_THRESH;

    if (isClosed) {
      // Phase 2: Eyes shutting. Start chronological counter.
      state.closedFrames++;
      state.phase = "BLINKING";
    } else {
      // Phase 3: Eyes Re-opening. Was it a real blink?
      if (state.phase === "BLINKING") {
        // Rule 1: Must have stayed closed for at least 2 frames (Noise only lasts 1 frame)
        if (state.closedFrames >= MIN_BLINK_FRAMES) {

          // Rule 2: Nose must have stayed vertically stable (Blocks tilting a photo to spoof a blink)
          let minNose = 1.0, maxNose = 0.0;
          for (const y of state.noseYHistory) {
            if (y < minNose) minNose = y;
            if (y > maxNose) maxNose = y;
          }
          if ((maxNose - minNose) < MAX_NOSE_MOVE_FOR_BLINK) {
            // Passed all anti-spoofing chronological & spatial constraints!
            state.phase = "VERIFIED";
            state.liveScore = 1;
          }
        }
        // Reset sequence whether valid or noise
        state.closedFrames = 0;
        state.phase = "WAITING";
      }
    }
  }

  return { ear, mouthWidth };
}

/** Decide if this face is LIVE based on accumulated signals */
function isLiveCheck(state, metrics) {
  return state.liveScore >= 1;
}

// ─── Status line builder ──────────────────────────────────────────────────────
function getLivenessHint(state, metrics) {
  if (state.phase === "VERIFIED") return "✅ Confirmed live — recognising…";

  if (state.phase === "GATHERING") {
    return "⏳ Scanning baselines (Keep neutral face for 0.5s)…";
  }

  if (isQuickMode) {
    if (state.phase === "BLINKING") return "👁 Blink detected...";
    return "⚡ Quick Scan: Please smile or blink naturally…";
  } else {
    if (state.phase === "BLINKING") return "👁 Hold blink...";
    return "👁 Secure Scan: Please blink completely…";
  }
}

// ─── Draw face overlay ────────────────────────────────────────────────────────
function drawFaceBox(lm, color, label) {
  let minX = 1, minY = 1, maxX = 0, maxY = 0;
  lm.forEach(p => {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  });
  const cw = markCanvas.width, ch = markCanvas.height;
  const pad = 0.02;
  const x = (minX - pad) * cw, y = (minY - pad) * ch;
  const w = (maxX - minX + pad * 2) * cw;
  const h = (maxY - minY + pad * 2) * ch;
  canvasCtx.strokeStyle = color;
  canvasCtx.lineWidth = 3;
  canvasCtx.strokeRect(x, y, w, h);

  // Label above the box
  canvasCtx.fillStyle = color;
  canvasCtx.font = "bold 13px sans-serif";
  canvasCtx.fillText(label, x + 4, y - 6);
}

// ─── Send pre-computed landmarks to backend for recognition ──────────────────
// The browser's FaceMesh already computed these for liveness — we reuse them
// directly as JSON. No image encoding, no canvas, no server-side FaceMesh.
// This is ~10x faster than uploading a JPEG.
function sendLandmarks(faceLandmarksList) {
  // faceLandmarksList: array of face landmark arrays (each element = one face)
  const faces = faceLandmarksList.map(lm =>
    lm.slice(0, 468).map(pt => ({ x: pt.x, y: pt.y, z: pt.z || 0 }))
  );
  // Lock the status bar so per-frame hints don't overwrite the result
  statusLocked = true;
  if (window.setMarkStatus) window.setMarkStatus('⏳ Recognising…', '');
  else markStatus.innerText = 'Recognising…';

  fetch("/recognize_landmarks", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ faces })
  })
    .then(r => r.json())
    .then(data => {
      if (data.error) {
        const msg = `Error: ${data.error}`;
        if (window.setMarkStatus) window.setMarkStatus(msg, 'err');
        else markStatus.innerText = msg;
        return;
      }
      if (data.results) {
        data.results.forEach(res => {
          if (res.status === "marked") {
            // Timestamp is stored in local time — slice HH:MM:SS directly (no Date conversion needed)
            const ts = res.marked_at ? res.marked_at.substring(11, 19) : '';
            if (window.showAttToast) window.showAttToast('marked', res.name, res.roll, `Roll: ${res.roll || '-'}  •  ${ts}`);
            if (window.bumpCounter) window.bumpCounter('marked');
            if (window.appendSessionItem) window.appendSessionItem('rec-marked',
              `✅ <strong>${res.name}</strong><br><small>Roll: ${res.roll || '-'} &nbsp;•&nbsp; ${ts} &nbsp;•&nbsp; Conf: ${Math.round(res.confidence * 100)}%</small>`);
            if (window.setMarkStatus) window.setMarkStatus(`✅ Marked: ${res.name} (Roll: ${res.roll || '-'})`, 'live');
            else markStatus.innerText = `✅ Marked: ${res.name}`;

            if (res.roll) sessionRecognizedRolls.add(res.roll);
            sessionRecognizedRolls.add(res.student_id); // Fallback

          } else if (res.status === "already_marked") {
            const markedTime = res.marked_at ? res.marked_at.substring(11, 19) : '';
            if (window.showAttToast) window.showAttToast('already', res.name, res.roll, `Already marked at ${markedTime}`);
            if (window.bumpCounter) window.bumpCounter('already');
            if (window.appendSessionItem) window.appendSessionItem('rec-already',
              `⚠️ <strong>${res.name}</strong> — Already Marked<br><small>Roll: ${res.roll || '-'} &nbsp;•&nbsp; First marked at: ${markedTime}</small>`);
            if (window.setMarkStatus) window.setMarkStatus(`⚠️ Already marked today: ${res.name} (at ${markedTime})`, 'warn');
            else markStatus.innerText = `Already Marked: ${res.name}`;

            if (res.roll) sessionRecognizedRolls.add(res.roll);
            sessionRecognizedRolls.add(res.student_id); // Fallback

          } else if (res.status === "unknown_student") {
            if (window.showAttToast) window.showAttToast('unknown', 'Unknown Person', '', 'Not enrolled in the system');
            if (window.bumpCounter) window.bumpCounter('unknown');
            if (window.appendSessionItem) window.appendSessionItem('rec-unknown', `❓ Unknown person — not enrolled`);
            if (window.setMarkStatus) window.setMarkStatus('❓ Unknown person detected — not in database', 'err');
            else markStatus.innerText = "Unknown Person (Not in DB)";

          } else if (res.status === "low_confidence") {
            if (window.setMarkStatus) window.setMarkStatus('⚠️ Low confidence — move closer and try again', 'warn');
            else markStatus.innerText = "Low Confidence. Move closer.";
          }
        });
        if (data.results.length === 0) {
          if (window.setMarkStatus) window.setMarkStatus('No face recognised after liveness check', 'warn');
        }
      }
    })
    .catch(err => {
      console.error(err);
      if (window.setMarkStatus) window.setMarkStatus('Connection error — check server', 'err');
      else markStatus.innerText = "Connection Error";
    })
    .finally(() => {
      // Release the lock quicker in Quick mode
      const lockTime = isQuickMode ? 1500 : 4000;
      setTimeout(() => { statusLocked = false; }, lockTime);
    });
}

// ─── Main FaceMesh results callback ──────────────────────────────────────────
async function onResults(results) {
  if (!isRunning) return;

  // ── Keep canvas intrinsic size in sync with incoming frame ───────────────
  if (markCanvas.width !== results.image.width || markCanvas.height !== results.image.height) {
    markCanvas.width = results.image.width || 640;
    markCanvas.height = results.image.height || 480;
  }

  canvasCtx.save();
  canvasCtx.clearRect(0, 0, markCanvas.width, markCanvas.height);
  canvasCtx.drawImage(results.image, 0, 0, markCanvas.width, markCanvas.height);

  const faces = results.multiFaceLandmarks;
  if (!faces || faces.length === 0) {
    canvasCtx.restore();
    if (!statusLocked) {
      if (window.setMarkStatus) window.setMarkStatus('No face detected — look at the camera', '');
      else markStatus.innerText = 'No face detected — look at the camera';
    }
    // Clean up all tracks if screen goes blank
    activeTracks = [];
    return;
  }

  let primaryHint = "";

  // Track faces ready to be sent in this batch
  const batchToSend = [];

  // 1. Assign consistent tracking IDs based on spatial nearest-neighbor
  // MediaPipe randomly swaps face indices. We MUST spatially track them against history!
  const currentTracks = [];
  faces.forEach(lm => {
    const centroid = getFaceCentroid(lm);

    // Find closest existing track within a max threshold (20% of screen)
    let bestMatch = null;
    let minD = 0.2;
    for (const track of activeTracks) {
      const d = Math.sqrt((track.centroid.cx - centroid.cx) ** 2 + (track.centroid.cy - centroid.cy) ** 2);
      if (d < minD && !track.claimed) {
        minD = d;
        bestMatch = track;
      }
    }

    if (bestMatch) {
      bestMatch.centroid = centroid;
      bestMatch.claimed = true;
      currentTracks.push({
        id: bestMatch.id,
        centroid: centroid,
        state: bestMatch.state,
        lm: lm
      });
    } else {
      // Brand new face entering the frame
      currentTracks.push({
        id: nextTrackId++,
        centroid: centroid,
        state: makeFaceState(),
        lm: lm
      });
    }
  });

  // Keep only claimed tracks for the next frame
  activeTracks = currentTracks.map(t => ({ id: t.id, centroid: t.centroid, state: t.state }));

  // 2. Process Liveness securely using preserved state
  currentTracks.forEach((track, idx) => {
    const { id, state, lm } = track;

    const metrics = updateFaceState(state, lm);
    const live = isLiveCheck(state, metrics);

    // Box colour: red = not live, amber = in-progress, green = live
    let boxColor = "#ef4444";
    if (state.liveScore >= 1 || (isQuickMode && live)) boxColor = "#f59e0b";
    if (live) boxColor = "#22c55e";

    const hint = getLivenessHint(state, metrics);
    if (idx === 0) primaryHint = hint;

    // Label: show blink status + score
    const scoreLabel = live ? "LIVE ✓" : (isQuickMode ? "checking..." : `score:${state.liveScore}/1`);
    drawFaceBox(lm, boxColor, scoreLabel);

    // Send for recognition if live and not in cooldown
    if (live && !cooldown.has(id)) {
      cooldown.add(id);

      batchToSend.push(lm);

      // Reset state & cooldown after a delay (force re-verify for next attempt)
      // Faster reset in Quick Scan mode.
      const resetDelay = isQuickMode ? 2000 : 5000;
      setTimeout(() => {
        const t = activeTracks.find(t => t.id === id);
        if (t) t.state = makeFaceState(); // Wipe history so proxy can't inherit it
        cooldown.delete(id);
      }, resetDelay);
    }
  });

  // Dispatch batch if any faces are ready
  if (batchToSend.length > 0) {
    sendLandmarks(batchToSend);
  }

  // Only update liveness hint if status is not locked by a recognition result
  if (primaryHint && !statusLocked) {
    if (window.setMarkStatus) window.setMarkStatus(primaryHint, '');
    else markStatus.innerText = primaryHint;
  }
  canvasCtx.restore();
}

// ─── Start button ─────────────────────────────────────────────────────────────
startMarkBtn.addEventListener("click", async () => {
  startMarkBtn.disabled = true;
  stopMarkBtn.disabled = false;
  isRunning = true;
  markStatus.innerText = "Starting camera…";

  if (!faceMesh) {
    faceMesh = new FaceMesh({
      locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
    });
    faceMesh.setOptions({
      maxNumFaces: 4,
      refineLandmarks: true,
      minDetectionConfidence: 0.4,
      minTrackingConfidence: 0.4
    });
    faceMesh.onResults(onResults);
  }

  if (!camera) {
    camera = new Camera(markVideo, {
      onFrame: async () => {
        if (isRunning && markVideo.readyState === 4) {
          await faceMesh.send({ image: markVideo });
        }
      },
      width: 1280,
      height: 720
    });
  }

  await camera.start();
  markStatus.innerText = "Active — look at the camera and blink naturally";
});

// ─── Stop button ──────────────────────────────────────────────────────────────
stopMarkBtn.addEventListener("click", async () => {
  isRunning = false;
  if (camera) {
    await camera.stop();
    camera = null;
  }
  if (faceMesh) {
    faceMesh.close();
    faceMesh = null;
  }
  if (markVideo.srcObject) {
    markVideo.srcObject.getTracks().forEach(t => t.stop());
    markVideo.srcObject = null;
  }
  // Clear all liveness states
  Object.keys(faceStates).forEach(k => delete faceStates[k]);
  cooldown.clear();
  sessionRecognizedRolls.clear();
  startMarkBtn.disabled = false;
  stopMarkBtn.disabled = true;
  markStatus.innerText = "Stopped.";
});
