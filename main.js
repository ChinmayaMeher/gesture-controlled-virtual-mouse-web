import {
  HandLandmarker,
  FilesetResolver,
  DrawingUtils
} from "@mediapipe/tasks-vision";

// DOM Elements
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const enableWebcamButton = document.getElementById("enable-webcam-btn");
const statusOverlay = document.getElementById("status-overlay");
const statusText = document.getElementById("status-text");
const logPanel = document.getElementById("log-panel");
const virtualCursor = document.getElementById("virtual-cursor");

// State
let handLandmarker = undefined;
let runningMode = "VIDEO";
let webcamRunning = false;
let isClicking = false;
let lastHoveredElement = null;

// Constants for gesture heuristics
const PINCH_THRESHOLD = 0.05; // Distance threshold for pinch

// Utility for logging
function logAction(msg) {
  const p = document.createElement("p");
  p.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  logPanel.prepend(p);
  if (logPanel.children.length > 10) {
    logPanel.removeChild(logPanel.lastChild);
  }
}

// Initialize MediaPipe HandLandmarker
async function initializeHandLandmarker() {
  statusText.textContent = "Loading AI Model...";
  try {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
        delegate: "GPU"
      },
      runningMode: runningMode,
      numHands: 1
    });
    statusText.textContent = "Model Loaded. Click Enable Webcam.";
    enableWebcamButton.disabled = false;
    logAction("Hand tracking model loaded successfully.");
  } catch (error) {
    console.error(error);
    statusText.textContent = "Error loading model!";
    logAction("Error loading model.");
  }
}

initializeHandLandmarker();

// Check if webcam access is supported
function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

if (!hasGetUserMedia()) {
  console.warn("getUserMedia() is not supported by your browser");
  statusText.textContent = "Webcam not supported";
}

// Enable webcam
enableWebcamButton.addEventListener("click", function () {
  if (!handLandmarker) {
    console.log("Wait! objectDetector not loaded yet.");
    return;
  }
  
  if (webcamRunning === true) {
    webcamRunning = false;
    enableWebcamButton.innerText = "Enable Webcam";
    statusOverlay.classList.remove("hidden");
    virtualCursor.classList.remove("active");
  } else {
    webcamRunning = true;
    enableWebcamButton.innerText = "Disable Webcam";
    statusOverlay.classList.add("hidden");
    virtualCursor.classList.add("active");
    
    // getUsermedia parameters
    const constraints = {
      video: { width: 640, height: 480 }
    };
    
    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
      video.srcObject = stream;
      video.addEventListener("loadeddata", predictWebcam);
    });
    logAction("Webcam activated.");
  }
});

let lastVideoTime = -1;
let results = undefined;
const drawingUtils = new DrawingUtils(canvasCtx);

async function predictWebcam() {
  canvasElement.style.width = video.videoWidth;
  canvasElement.style.height = video.videoHeight;
  canvasElement.width = video.videoWidth;
  canvasElement.height = video.videoHeight;
  
  if (runningMode === "IMAGE") {
    runningMode = "VIDEO";
    await handLandmarker.setOptions({ runningMode: "VIDEO" });
  }
  
  let startTimeMs = performance.now();
  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;
    results = handLandmarker.detectForVideo(video, startTimeMs);
  }
  
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  
  if (results.landmarks) {
    for (const landmarks of results.landmarks) {
      drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, {
        color: "#10b981",
        lineWidth: 3
      });
      drawingUtils.drawLandmarks(landmarks, { color: "#3b82f6", lineWidth: 2 });
      
      processGestures(landmarks);
    }
  }
  
  // If no hands detected, hide cursor or move it offscreen?
  if (!results.landmarks || results.landmarks.length === 0) {
     if(lastHoveredElement) {
       lastHoveredElement.classList.remove("simulated-hover");
       lastHoveredElement = null;
     }
  }
  
  canvasCtx.restore();
  
  if (webcamRunning) {
    window.requestAnimationFrame(predictWebcam);
  }
}

// Gesture Processing Logic
function processGestures(landmarks) {
  // Landmarks: 8 is index finger tip, 4 is thumb tip
  const indexTip = landmarks[8];
  const thumbTip = landmarks[4];
  
  // 1. Move Virtual Cursor based on index finger
  // Note: camera is mirrored, so we mirror X
  const screenX = (1 - indexTip.x) * window.innerWidth;
  const screenY = indexTip.y * window.innerHeight;
  
  virtualCursor.style.left = `${screenX}px`;
  virtualCursor.style.top = `${screenY}px`;
  
  // Find element under cursor
  const elementUnderCursor = document.elementFromPoint(screenX, screenY);
  
  // Handle hover state
  if (elementUnderCursor !== lastHoveredElement) {
    if (lastHoveredElement && lastHoveredElement.classList.contains("demo-target")) {
      lastHoveredElement.classList.remove("simulated-hover");
    } else if (lastHoveredElement && lastHoveredElement.classList.contains("demo-card")) {
      lastHoveredElement.classList.remove("simulated-hover");
    }
    
    if (elementUnderCursor && (elementUnderCursor.classList.contains("demo-target") || elementUnderCursor.classList.contains("demo-card"))) {
      elementUnderCursor.classList.add("simulated-hover");
    }
    lastHoveredElement = elementUnderCursor;
  }
  
  // 2. Detect Pinch to Click
  // Euclidean distance between thumb tip and index tip in normalized coordinates
  const distance = Math.sqrt(
    Math.pow(indexTip.x - thumbTip.x, 2) +
    Math.pow(indexTip.y - thumbTip.y, 2) +
    Math.pow(indexTip.z - thumbTip.z, 2)
  );
  
  if (distance < PINCH_THRESHOLD) {
    if (!isClicking) {
      isClicking = true;
      virtualCursor.classList.add("clicking");
      if (elementUnderCursor) {
         elementUnderCursor.classList.add("simulated-active");
         elementUnderCursor.click();
         logAction(`Clicked on ${elementUnderCursor.tagName} (ID: ${elementUnderCursor.id || 'none'})`);
      }
    }
  } else {
    if (isClicking) {
      isClicking = false;
      virtualCursor.classList.remove("clicking");
      if (elementUnderCursor) {
         elementUnderCursor.classList.remove("simulated-active");
      }
    }
  }
}

// Add event listeners to demo buttons to prove they work
document.querySelectorAll('.demo-target').forEach(btn => {
  btn.addEventListener('click', () => {
    btn.style.backgroundColor = 'var(--accent-color)';
    btn.textContent = 'Clicked!';
    setTimeout(() => {
      btn.style.backgroundColor = '';
      btn.textContent = btn.id === 'target-1' ? 'Target 1' : 'Target 2';
    }, 1000);
  });
});
