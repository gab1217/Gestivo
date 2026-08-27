"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { Tensor } from "@tensorflow/tfjs-core";
import type { HandLandmarker, NormalizedLandmark } from "@mediapipe/tasks-vision";

type TFLiteModel = {
  predict: (input: Tensor | Tensor[] | Record<string, Tensor>) => Tensor | Tensor[] | Record<string, Tensor>;
};

type TFLiteBrowserApi = {
  setWasmPath: (path: string) => void;
  loadTFLiteModel: (path: string) => Promise<TFLiteModel>;
};

const LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
const TARGET_ANALYSIS_FPS = 8;
const ANALYSIS_WIDTH = 320;
const IMAGE_MODEL_INTERVAL = 4;
const IMAGE_WEIGHT = 0.5;
const LANDMARK_WEIGHT = 0.5;
const HAND_SHAPE_REFRESH_THRESHOLD = 0.035;
const SMOOTHING_FRAMES = 3;
const STABLE_HOLD_MS = 400;
const MIN_CONFIRMATION_FRAMES = 3;
const AUTO_SPACE_DELAY_MS = 2000;
const UR_PARALLEL_COSINE_THRESHOLD = 0.936779;
const UR_TIP_ORDER_THRESHOLD = 0.003443;
const O_THUMB_INDEX_GAP_THRESHOLD = 0.29855;
const O_LITTLE_FINGER_ROUND_THRESHOLD = 0.32286;
const M_THUMB_TIP_DISTANCE_THRESHOLD = 0.4319;
const M_MIDDLE_FINGER_FOLD_THRESHOLD = 0.2141;
const PUBLIC_BASE = typeof window !== "undefined" && window.location.hostname.endsWith("github.io") ? "/Gestivo" : "";
const publicAsset = (path: string) => `${PUBLIC_BASE}${path}`;
const ANDROID_DOWNLOAD = "https://github.com/gab1217/Gestivo/releases/latest/download/Gestivo-Android.apk";
const WINDOWS_DOWNLOAD = "https://github.com/gab1217/Gestivo/releases/download/desktop-v1.0.3/Gestivo-Setup-v1.0.3.exe";
const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15],
  [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
] as const;

type Engine = {
  handLandmarker: HandLandmarker;
  imageModel: TFLiteModel;
  landmarkModel: TFLiteModel;
  tf: typeof import("@tensorflow/tfjs-core");
};

type TemporalState = {
  history: number[][];
  latestImage: number[] | null;
  previousLandmarks: Float32Array | null;
  requestNumber: number;
  handWasPresent: boolean;
  candidate: string | null;
  candidateStarted: number;
  candidateFrames: number;
  lastCommitted: string | null;
  repeatRearmed: boolean;
  noHandStarted: number;
  autoSpaceAdded: boolean;
};

type Message = { sender: "signer" | "speaker"; text: string };

const freshTemporalState = (): TemporalState => ({
  history: [], latestImage: null, previousLandmarks: null, requestNumber: 0, handWasPresent: false,
  candidate: null, candidateStarted: 0, candidateFrames: 0,
  lastCommitted: null, repeatRearmed: false, noHandStarted: 0,
  autoSpaceAdded: false,
});

function asTensor(output: Tensor | Tensor[] | Record<string, Tensor>): Tensor {
  if (Array.isArray(output)) return output[0];
  if ("dataSync" in output) return output as Tensor;
  return Object.values(output)[0];
}

function loadBrowserScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector<HTMLScriptElement>(`script[src="${src}"]`);
    if (existing?.dataset.loaded === "true") return resolve();
    const script = existing ?? document.createElement("script");
    script.src = src;
    script.async = true;
    script.addEventListener("load", () => {
      script.dataset.loaded = "true";
      resolve();
    }, { once: true });
    script.addEventListener("error", () => reject(new Error(`Could not load ${src}`)), { once: true });
    if (!existing) document.head.appendChild(script);
  });
}

function correctParallelUConfusedAsR(probabilities: number[], vector: Float32Array) {
  const rIndex = LABELS.indexOf("R"), uIndex = LABELS.indexOf("U");
  const bestIndex = probabilities.reduce((best, value, index) => value > probabilities[best] ? index : best, 0);
  if (bestIndex !== rIndex) return;
  const x = (point: number) => vector[point * 3], y = (point: number) => vector[point * 3 + 1];
  const indexX = x(8) - x(5), indexY = y(8) - y(5);
  const middleX = x(12) - x(9), middleY = y(12) - y(9);
  const indexLength = Math.hypot(indexX, indexY), middleLength = Math.hypot(middleX, middleY);
  if (!indexLength || !middleLength) return;
  const directionCosine = (indexX * middleX + indexY * middleY) / (indexLength * middleLength);
  const tipOrder = (x(8) - x(12)) * (x(5) - x(9));
  if (directionCosine > UR_PARALLEL_COSINE_THRESHOLD && tipOrder > UR_TIP_ORDER_THRESHOLD) {
    probabilities[uIndex] = Math.max(probabilities[uIndex], probabilities[rIndex] + 0.001);
  }
}

function correctClosedHandConfusedAsS(probabilities: number[], vector: Float32Array) {
  const sIndex = LABELS.indexOf("S"), mIndex = LABELS.indexOf("M"), oIndex = LABELS.indexOf("O");
  const bestIndex = probabilities.reduce((best, value, index) => value > probabilities[best] ? index : best, 0);
  if (bestIndex !== sIndex) return;

  const point = (index: number) => ({ x: vector[index * 3], y: vector[index * 3 + 1], z: vector[index * 3 + 2] });
  const distance = (first: number, second: number, dimensions = 3) => {
    const left = point(first), right = point(second);
    const x = left.x - right.x, y = left.y - right.y, z = dimensions === 3 ? left.z - right.z : 0;
    return Math.hypot(x, y, z);
  };
  const palm = Math.max(distance(0, 9), 1e-6);
  const thumbIndexGap2d = distance(4, 8, 2) / palm;
  const littleFingerRound = distance(20, 17) / palm;
  const isO = thumbIndexGap2d < O_THUMB_INDEX_GAP_THRESHOLD && littleFingerRound > O_LITTLE_FINGER_ROUND_THRESHOLD;
  if (isO) {
    probabilities[oIndex] = Math.max(probabilities[oIndex], probabilities[sIndex] + 0.001);
    return;
  }

  const thumbTipDistanceMax = Math.max(...[8, 12, 16, 20].map((tip) => distance(4, tip) / palm));
  const middleFingerFold = distance(12, 11) / palm;
  const isM = thumbTipDistanceMax > M_THUMB_TIP_DISTANCE_THRESHOLD && middleFingerFold > M_MIDDLE_FINGER_FOLD_THRESHOLD;
  if (isM) probabilities[mIndex] = Math.max(probabilities[mIndex], probabilities[sIndex] + 0.001);
}

export default function GestivoApp({ view = "home" }: { view?: "home" | "recognizer" }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const processCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const imageCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const engineRef = useRef<Engine | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationRef = useRef<number | null>(null);
  const temporalRef = useRef<TemporalState>(freshTemporalState());
  const lastVideoTimeRef = useRef(-1);
  const lastProcessedAtRef = useRef(0);

  const [modelStatus, setModelStatus] = useState("Preparing private AI models…");
  const [modelsReady, setModelsReady] = useState(false);
  const [modelFailed, setModelFailed] = useState(false);
  const [modelAttempt, setModelAttempt] = useState(0);
  const [cameraRunning, setCameraRunning] = useState(false);
  const [liveLetter, setLiveLetter] = useState("—");
  const [confirmedLetter, setConfirmedLetter] = useState("—");
  const [confidence, setConfidence] = useState(0);
  const [stability, setStability] = useState(0);
  const [draft, setDraft] = useState("");
  const [typedReply, setTypedReply] = useState("");
  const [mode, setMode] = useState<"sign" | "type">("sign");
  const [messages, setMessages] = useState<Message[]>([]);
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [voiceName, setVoiceName] = useState("");
  const [speechRate, setSpeechRate] = useState(1);
  const [textScale, setTextScale] = useState("normal");

  useEffect(() => {
    if (view !== "recognizer") return;
    let cancelled = false;
    async function prepareModels() {
      try {
        setModelFailed(false);
        setModelStatus("Preparing private AI models…");
        const [vision, tf] = await Promise.all([
          import("@mediapipe/tasks-vision"),
          import("@tensorflow/tfjs-core"),
          import("@tensorflow/tfjs-backend-webgl"),
          import("@tensorflow/tfjs-backend-cpu"),
        ]).then(([visionModule, tfModule]) => [visionModule, tfModule] as const);
        try { await tf.setBackend("webgl"); } catch { await tf.setBackend("cpu"); }
        await tf.ready();
        const browserGlobal = window as unknown as { tf: typeof tf; tflite?: TFLiteBrowserApi };
        browserGlobal.tf = tf;
        await loadBrowserScript(publicAsset("/vendor/tf-tflite.min.js"));
        const tflite = browserGlobal.tflite;
        if (!tflite) throw new Error("TensorFlow Lite browser engine did not initialize");
        tflite.setWasmPath(publicAsset("/tflite-wasm/"));
        const fileset = await vision.FilesetResolver.forVisionTasks(publicAsset("/mediapipe-wasm"));
        const [handLandmarker, imageModel, landmarkModel] = await Promise.all([
          vision.HandLandmarker.createFromOptions(fileset, {
            baseOptions: { modelAssetPath: publicAsset("/models/hand_landmarker.task") },
            runningMode: "VIDEO", numHands: 1,
            minHandDetectionConfidence: 0.55,
            minHandPresenceConfidence: 0.55,
            minTrackingConfidence: 0.55,
          }),
          tflite.loadTFLiteModel(publicAsset("/models/fsl_model.tflite")),
          tflite.loadTFLiteModel(publicAsset("/models/landmark_model.tflite")),
        ]);
        if (cancelled) { handLandmarker.close(); return; }
        engineRef.current = { handLandmarker, imageModel, landmarkModel, tf };
        setModelsReady(true);
        setModelStatus("AI ready · camera stays private");
      } catch (error) {
        console.error(error);
        setModelsReady(false);
        setModelFailed(true);
        setModelStatus("AI could not load. Check your connection, then retry.");
      }
    }
    prepareModels();
    return () => { cancelled = true; engineRef.current?.handLandmarker.close(); };
  }, [view, modelAttempt]);

  useEffect(() => {
    const updateVoices = () => {
      const available = window.speechSynthesis?.getVoices() ?? [];
      setVoices(available);
      if (!voiceName && available.length) {
        const preferred = available.find((voice) => /fil|philipp|tagalog/i.test(`${voice.lang} ${voice.name}`))
          ?? available.find((voice) => voice.lang.startsWith("en")) ?? available[0];
        setVoiceName(preferred.name);
      }
    };
    updateVoices();
    window.speechSynthesis?.addEventListener("voiceschanged", updateVoices);
    navigator.serviceWorker?.register(publicAsset("/service-worker.js"), { scope: `${PUBLIC_BASE}/` }).catch(() => undefined);
    return () => window.speechSynthesis?.removeEventListener("voiceschanged", updateVoices);
  }, [voiceName]);

  const speak = useCallback((text: string) => {
    if (!text.trim() || !("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text.trim());
    const selected = voices.find((voice) => voice.name === voiceName);
    if (selected) utterance.voice = selected;
    utterance.rate = speechRate;
    window.speechSynthesis.speak(utterance);
  }, [speechRate, voiceName, voices]);

  const stopCamera = useCallback(() => {
    if (animationRef.current !== null) cancelAnimationFrame(animationRef.current);
    animationRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraRunning(false);
    lastVideoTimeRef.current = -1;
    lastProcessedAtRef.current = 0;
    setLiveLetter("—");
    setStability(0);
    setModelStatus(modelsReady ? "Camera paused · AI ready" : "Preparing private AI models…");
  }, [modelsReady]);

  const drawLandmarks = (landmarks: NormalizedLandmark[], width: number, height: number) => {
    const overlay = overlayRef.current;
    if (!overlay) return;
    if (overlay.width !== width || overlay.height !== height) { overlay.width = width; overlay.height = height; }
    const context = overlay.getContext("2d");
    if (!context) return;
    context.clearRect(0, 0, width, height);
    context.strokeStyle = "#68e0ac"; context.fillStyle = "#effbf5"; context.lineWidth = Math.max(2, width / 320);
    for (const [start, end] of CONNECTIONS) {
      context.beginPath();
      context.moveTo(landmarks[start].x * width, landmarks[start].y * height);
      context.lineTo(landmarks[end].x * width, landmarks[end].y * height);
      context.stroke();
    }
    for (const point of landmarks) {
      context.beginPath();
      context.arc(point.x * width, point.y * height, Math.max(2.5, width / 230), 0, Math.PI * 2);
      context.fill();
    }
  };

  const landmarkVector = (landmarks: NormalizedLandmark[]) => {
    const wrist = landmarks[0];
    const values: number[] = [];
    let scale = 0;
    for (const point of landmarks) {
      const x = point.x - wrist.x, y = point.y - wrist.y, z = point.z - wrist.z;
      values.push(x, y, z); scale = Math.max(scale, Math.hypot(x, y));
    }
    return new Float32Array(values.map((value) => scale > 0 ? value / scale : value));
  };

  const imageTensor = (tf: typeof import("@tensorflow/tfjs-core"), source: HTMLCanvasElement, landmarks: NormalizedLandmark[]) => {
    const xs = landmarks.map((point) => point.x * source.width), ys = landmarks.map((point) => point.y * source.height);
    const padding = Math.min(source.width, source.height) / 12;
    const left = Math.max(0, Math.min(...xs) - padding), top = Math.max(0, Math.min(...ys) - padding);
    const right = Math.min(source.width, Math.max(...xs) + padding), bottom = Math.min(source.height, Math.max(...ys) + padding);
    const sourceWidth = Math.max(1, right - left), sourceHeight = Math.max(1, bottom - top);
    let output = imageCanvasRef.current;
    if (!output) {
      output = document.createElement("canvas"); output.width = 224; output.height = 224;
      imageCanvasRef.current = output;
    }
    const context = output.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("Image canvas is unavailable");
    context.fillStyle = "#000"; context.fillRect(0, 0, 224, 224);
    const scale = Math.min(224 / sourceWidth, 224 / sourceHeight), width = sourceWidth * scale, height = sourceHeight * scale;
    context.drawImage(source, left, top, sourceWidth, sourceHeight, (224 - width) / 2, (224 - height) / 2, width, height);
    const rgba = context.getImageData(0, 0, 224, 224).data, rgb = new Float32Array(224 * 224 * 3);
    for (let input = 0, target = 0; input < rgba.length; input += 4) { rgb[target++] = rgba[input]; rgb[target++] = rgba[input + 1]; rgb[target++] = rgba[input + 2]; }
    return tf.tensor4d(rgb, [1, 224, 224, 3], "float32");
  };

  const handleNoHand = (now: number) => {
    const state = temporalRef.current;
    state.handWasPresent = false; state.latestImage = null; state.previousLandmarks = null; state.history = []; state.candidate = null; state.candidateStarted = 0; state.candidateFrames = 0;
    if (!state.noHandStarted) state.noHandStarted = now;
    const elapsed = now - state.noHandStarted;
    if (elapsed >= 300) state.repeatRearmed = true;
    if (elapsed >= AUTO_SPACE_DELAY_MS && !state.autoSpaceAdded) { state.autoSpaceAdded = true; setDraft((value) => value && !value.endsWith(" ") ? `${value} ` : value); }
    setLiveLetter("—"); setConfidence(0); setStability(0); setModelStatus("No hand detected");
    const overlay = overlayRef.current;
    if (overlay) overlay.getContext("2d")?.clearRect(0, 0, overlay.width, overlay.height);
  };

  const processHand = (landmarks: NormalizedLandmark[], source: HTMLCanvasElement, now: number) => {
    const engine = engineRef.current;
    if (!engine) return;
    const state = temporalRef.current;
    state.noHandStarted = 0; state.autoSpaceAdded = false;
    if (!state.handWasPresent) { state.handWasPresent = true; state.history = []; state.latestImage = null; state.previousLandmarks = null; }
    const vector = landmarkVector(landmarks);
    const handShapeChanged = !state.previousLandmarks || vector.reduce(
      (sum, value, index) => sum + Math.abs(value - (state.previousLandmarks?.[index] ?? value)), 0,
    ) / vector.length >= HAND_SHAPE_REFRESH_THRESHOLD;
    if (!state.latestImage || state.requestNumber % IMAGE_MODEL_INTERVAL === 0 || handShapeChanged) {
      const input = imageTensor(engine.tf, source, landmarks), output = asTensor(engine.imageModel.predict(input));
      state.latestImage = Array.from(output.dataSync()); input.dispose(); output.dispose();
    }
    state.previousLandmarks = vector;
    const landmarkInput = engine.tf.tensor2d(vector, [1, 63], "float32");
    const landmarkOutput = asTensor(engine.landmarkModel.predict(landmarkInput));
    const landmarkProbabilities = Array.from(landmarkOutput.dataSync()); landmarkInput.dispose(); landmarkOutput.dispose();
    state.requestNumber += 1;
    const hybrid = LABELS.map((_, index) => IMAGE_WEIGHT * (state.latestImage?.[index] ?? 0) + LANDMARK_WEIGHT * landmarkProbabilities[index]);
    correctParallelUConfusedAsR(hybrid, vector);
    correctClosedHandConfusedAsS(hybrid, vector);
    state.history.push(hybrid); if (state.history.length > SMOOTHING_FRAMES) state.history.shift();
    const smoothed = LABELS.map((_, index) => state.history.reduce((sum, values) => sum + values[index], 0) / state.history.length);
    const bestIndex = smoothed.reduce((best, value, index) => value > smoothed[best] ? index : best, 0);
    const bestConfidence = smoothed[bestIndex], letter = bestConfidence >= 0.4 ? LABELS[bestIndex] : "—";
    if (letter !== "—") {
      if (state.candidate !== letter) { state.candidate = letter; state.candidateStarted = now; state.candidateFrames = 1; }
      else state.candidateFrames += 1;
    } else {
      state.candidate = null; state.candidateStarted = 0; state.candidateFrames = 0;
    }
    let progress = state.candidate ? Math.min((now - state.candidateStarted) / STABLE_HOLD_MS, state.candidateFrames / MIN_CONFIRMATION_FRAMES, 1) : 0;
    if (state.candidate === state.lastCommitted && !state.repeatRearmed) progress = 0;
    if (state.candidate && progress >= 1 && (state.candidate !== state.lastCommitted || state.repeatRearmed)) {
      const committed = state.candidate;
      state.lastCommitted = committed; state.repeatRearmed = false; state.candidateStarted = now; state.candidateFrames = 0; progress = 0;
      setConfirmedLetter(committed); setDraft((value) => value + committed);
    }
    setLiveLetter(letter); setConfidence(bestConfidence); setStability(progress); setModelStatus("Hand detected · processing locally");
  };

  const recognitionLoop = (frameTime: number) => {
    const video = videoRef.current, engine = engineRef.current;
    if (!video || !engine || !streamRef.current) return;
    const analysisInterval = 1000 / TARGET_ANALYSIS_FPS;
    if (
      frameTime - lastProcessedAtRef.current >= analysisInterval &&
      video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
      video.currentTime !== lastVideoTimeRef.current
    ) {
      lastProcessedAtRef.current = frameTime;
      lastVideoTimeRef.current = video.currentTime;
      const sourceWidth = video.videoWidth || 640, sourceHeight = video.videoHeight || 480;
      const width = Math.min(ANALYSIS_WIDTH, sourceWidth), height = Math.max(1, Math.round(sourceHeight * width / sourceWidth));
      let canvas = processCanvasRef.current;
      if (!canvas) { canvas = document.createElement("canvas"); processCanvasRef.current = canvas; }
      if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
      const context = canvas.getContext("2d", { willReadFrequently: true });
      if (context) {
        context.save(); context.translate(width, 0); context.scale(-1, 1); context.drawImage(video, 0, 0, width, height); context.restore();
        try {
          const result = engine.handLandmarker.detectForVideo(canvas, frameTime);
          if (result.landmarks.length) { drawLandmarks(result.landmarks[0], width, height); processHand(result.landmarks[0], canvas, frameTime); }
          else handleNoHand(frameTime);
        } catch (error) { console.error(error); setModelStatus("Recognition paused after an AI error"); }
      }
    }
    animationRef.current = requestAnimationFrame(recognitionLoop);
  };

  const startCamera = async () => {
    if (!modelsReady || cameraRunning) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 480 }, height: { ideal: 360 }, frameRate: { ideal: 24, max: 30 } }, audio: false });
      streamRef.current = stream; temporalRef.current = freshTemporalState();
      lastVideoTimeRef.current = -1; lastProcessedAtRef.current = 0;
      const video = videoRef.current; if (!video) return;
      video.srcObject = stream; await video.play();
      setCameraRunning(true); setModelStatus("Camera ready · show one hand"); animationRef.current = requestAnimationFrame(recognitionLoop);
    } catch (error) { console.error(error); setModelStatus("Camera permission is needed to recognize signs"); }
  };

  useEffect(() => () => {
    if (animationRef.current !== null) cancelAnimationFrame(animationRef.current);
    streamRef.current?.getTracks().forEach((track) => track.stop());
    engineRef.current?.handLandmarker.close();
  }, []);

  const chooseMode = (nextMode: "sign" | "type") => { setMode(nextMode); if (nextMode === "type") stopCamera(); };
  const addMessage = (sender: Message["sender"], text: string) => {
    const clean = text.trim(); if (!clean) return;
    setMessages((current) => [...current, { sender, text: clean }]);
    if (sender === "signer") setDraft(""); else setTypedReply("");
  };

  if (view === "home") {
    return (
      <main className="site home-site" id="top">
        <nav className="nav shell" aria-label="Main navigation">
          <a className="brand" href={publicAsset("/")} aria-label="Gestivo home"><img src={publicAsset("/gestivo-logo.png")} alt="" /><span>Gestivo</span></a>
          <div className="nav-links"><a href="#features">How it works</a><a href="#downloads">Downloads</a><a href="#about">About</a></div>
          <a className="nav-cta" href={publicAsset("/recognizer/")}>Use online</a>
        </nav>

        <section className="hero shell">
          <div className="hero-copy">
            <p className="eyebrow"><span /> Filipino Sign Language assistant</p>
            <h1>One place for every Gestivo experience.</h1>
            <p className="lede">Recognize FSL in your browser, install the Android app, or download the complete Windows desktop version—all from one organized home.</p>
            <div className="hero-actions"><a className="button button-primary" href={publicAsset("/recognizer/")}>Use Gestivo online</a><a className="button button-secondary" href="#downloads">Choose a download</a></div>
            <p className="privacy-note">No account. No video uploads. Recognition happens on your device.</p>
          </div>
          <div className="product-card" aria-label="Gestivo recognition preview">
            <div className="camera-stage demo-stage"><div className="camera-topline"><span className="status"><i /> Private recognition</span><span className="live-label">FSL</span></div><div className="hand-frame" aria-hidden="true"><span className="corner corner-one" /><span className="corner corner-two" /><span className="corner corner-three" /><span className="corner corner-four" /><div className="gesture-mark">G</div></div><div className="prediction"><span>Detected letter</span><strong>G</strong><span>96% confidence</span></div></div>
            <div className="transcript"><span className="transcript-label">Your message</span><p>Good morning</p><a className="mini-link" href={publicAsset("/recognizer/")}>Open recognizer →</a></div>
          </div>
        </section>

        <section className="choice-section" id="features">
          <div className="shell"><div className="section-heading"><p className="eyebrow"><span /> Start here</p><h2>Choose what you need.</h2><p>Each option has its own focused screen, so the website stays fast and easy to understand.</p></div>
            <div className="choice-grid">
              <article><span className="choice-number">01</span><h3>Use online</h3><p>Open only the live recognizer and conversation tools in your browser.</p><a href={publicAsset("/recognizer/")}>Open recognizer →</a></article>
              <article><span className="choice-number">02</span><h3>Android app</h3><p>Install Gestivo on Android for a mobile, offline-ready experience.</p><a href={ANDROID_DOWNLOAD}>Download APK →</a></article>
              <article><span className="choice-number">03</span><h3>Windows desktop</h3><p>Get the complete desktop experience with local AI recognition.</p><a href={WINDOWS_DOWNLOAD}>Download for Windows →</a></article>
            </div>
          </div>
        </section>

        <section className="download-hub shell" id="downloads">
          <div className="download-intro"><p className="eyebrow"><span /> Downloads</p><h2>Install Gestivo on your device.</h2><p>Choose the version made for your device. The online recognizer needs no installation.</p></div>
          <div className="download-cards">
            <article><div className="platform-icon">A</div><div><small>ANDROID 8.0+</small><h3>Gestivo for Android</h3><p>Mobile interface, camera recognition, conversation mode, and offline AI.</p></div><a className="button button-primary" href={ANDROID_DOWNLOAD}>Download Android APK</a></article>
            <article><div className="platform-icon">W</div><div><small>WINDOWS 10/11</small><h3>Gestivo for Windows</h3><p>Full desktop application with recognition and conversation tools.</p></div><a className="button button-secondary" href={WINDOWS_DOWNLOAD}>Download Windows app</a></article>
          </div>
        </section>

        <section className="about-section" id="about"><div className="shell about-grid"><div><p className="eyebrow"><span /> Built for inclusion</p><h2>Communication should not depend on who can hear or speak.</h2></div><div className="feature-grid"><article><strong>26</strong><span>FSL alphabet classes</span></article><article><strong>2</strong><span>AI models working together</span></article><article><strong>0</strong><span>Camera frames uploaded</span></article><article><strong>3</strong><span>Ways to use Gestivo</span></article></div></div></section>
        <section className="privacy-section shell"><h2>Private by design.</h2><p>Gestivo processes camera frames on your device. The website does not create an account, record video, or send camera images to a server.</p><div><span>✓ On-device AI</span><span>✓ No sign-in</span><span>✓ No video storage</span></div></section>
        <footer className="footer shell"><a className="brand" href="#top"><img src={publicAsset("/gestivo-logo.png")} alt="" /><span>Gestivo</span></a><p>Filipino Sign Language research prototype.</p><a href="#top">Back to top ↑</a></footer>
      </main>
    );
  }

  return (
    <main className={`site recognizer-site text-${textScale}`}>
      <nav className="nav shell recognizer-nav" aria-label="Recognizer navigation">
        <a className="brand" href={publicAsset("/")}><img src={publicAsset("/gestivo-logo.png")} alt="" /><span>Gestivo</span></a>
        <span className="route-label">Online recognizer</span>
        <a className="nav-cta" href={publicAsset("/")}>← Back home</a>
      </nav>
      <header className="recognizer-header shell"><div><p className="eyebrow"><span /> Live workspace</p><h1>Sign, compose, and speak.</h1><p>Start the camera when the AI is ready. Your video stays inside this browser.</p></div><div className={`readiness ${modelsReady ? "ready" : ""}`}><i /><span>{modelStatus}</span></div></header>
      <section className="recognizer-workspace">
        <div className="translator-shell shell">
          <div className="conversation-panel">
            <div className="panel-title"><div><span>Conversation</span><small>{messages.length ? `${messages.length} messages` : "Ready when you are"}</small></div><button type="button" onClick={() => setMessages([])} disabled={!messages.length}>Clear</button></div>
            <div className="messages" aria-live="polite">{!messages.length && <div className="empty-conversation"><strong>Your conversation will appear here.</strong><span>Sign a message or type a reply to begin.</span></div>}{messages.map((message, index) => <div className={`message ${message.sender}`} key={`${message.sender}-${index}`}><p>{message.text}</p><small>{message.sender === "signer" ? "Signed by you" : "Typed reply"}</small></div>)}</div>
            <div className="composer"><div className="mode-switch" role="tablist" aria-label="Message input mode"><button type="button" className={mode === "sign" ? "active" : ""} onClick={() => chooseMode("sign")}>Sign → Speech</button><button type="button" className={mode === "type" ? "active" : ""} onClick={() => chooseMode("type")}>Type a reply</button></div>
              {mode === "sign" ? <><textarea value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="Recognized letters will appear here" aria-label="Recognized sign message" /><div className="composer-actions"><button type="button" onClick={() => setDraft((value) => `${value} `)}>Space</button><button type="button" aria-label="Add question mark" onClick={() => setDraft((value) => `${value.trimEnd()}?`)}>?</button><button type="button" aria-label="Add exclamation mark" onClick={() => setDraft((value) => `${value.trimEnd()}!`)}>!</button><button type="button" onClick={() => setDraft((value) => value.slice(0, -1))}>Delete</button><button type="button" onClick={() => { setDraft(""); setConfirmedLetter("—"); temporalRef.current = freshTemporalState(); }}>Clear</button><span /><button type="button" onClick={() => speak(draft)}>Speak</button><button className="primary-small" type="button" onClick={() => addMessage("signer", draft)}>Add message</button></div></> : <><textarea value={typedReply} onChange={(event) => setTypedReply(event.target.value)} placeholder="Type the other person's reply" aria-label="Typed reply" /><div className="composer-actions"><span /><button type="button" onClick={() => speak(typedReply)}>Speak</button><button className="primary-small" type="button" onClick={() => addMessage("speaker", typedReply)}>Add reply</button></div></>}
            </div>
          </div>
          <div className="recognition-panel">
            <div className="camera-live"><video ref={videoRef} muted playsInline aria-label="Front camera preview" /><canvas ref={overlayRef} aria-hidden="true" />{!cameraRunning && <div className="camera-placeholder"><img src={publicAsset("/gestivo-logo.png")} alt="" /><strong>{modelsReady ? "Camera paused" : "Loading private AI"}</strong><span>{modelStatus}</span></div>}<div className="camera-badge"><i className={cameraRunning ? "on" : ""} />{modelStatus}</div><div className="live-letter">{liveLetter}</div></div>
            <button className="camera-button" type="button" disabled={!modelsReady && !modelFailed} onClick={modelFailed ? () => setModelAttempt((value) => value + 1) : cameraRunning ? stopCamera : startCamera}>{modelFailed ? "Retry AI loading" : cameraRunning ? "Pause camera" : modelsReady ? "Start camera" : "Loading AI…"}</button>
            <div className="metrics"><div><span>Confirmed</span><strong>{confirmedLetter}</strong></div><div><span>Confidence</span><strong>{Math.round(confidence * 100)}%</strong></div></div>
            <div className="stability"><div><span>Hold sign steady</span><span>{Math.round(stability * 100)}%</span></div><progress value={stability} max="1" /></div><p className="camera-help">Hold one sign steadily. Remove your hand briefly before repeating the same letter.</p>
          </div>
        </div>
        <div className="settings shell"><label>Voice<select value={voiceName} onChange={(event) => setVoiceName(event.target.value)}>{voices.map((voice) => <option value={voice.name} key={`${voice.name}-${voice.lang}`}>{voice.name} · {voice.lang}</option>)}</select></label><label>Speech rate<input type="range" min="0.7" max="1.3" step="0.1" value={speechRate} onChange={(event) => setSpeechRate(Number(event.target.value))} /><span>{speechRate.toFixed(1)}×</span></label><label>Text size<select value={textScale} onChange={(event) => setTextScale(event.target.value)}><option value="normal">Normal</option><option value="large">Large</option><option value="extra">Extra large</option></select></label></div>
      </section>
      <footer className="footer shell"><a className="brand" href={publicAsset("/")}><img src={publicAsset("/gestivo-logo.png")} alt="" /><span>Gestivo</span></a><p>Private, on-device FSL recognition.</p><a href={publicAsset("/")}>Home</a></footer>
    </main>
  );
}
