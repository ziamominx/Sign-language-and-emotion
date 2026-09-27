/* Arena shared library: camera + perception loop, HUD, API helpers.
 *
 * The perception loop runs MediaPipe hands/face/pose natively in the page
 * (same WASM stack as the communicator site) and posts a compact rolling
 * landmark window to /api/arena/perceive-style endpoints. Latency and FPS
 * are measured, never faked.
 */
const ArenaSettings = {
  defaults: {mirror: true, confidence: 0.6, sensitivity: 'medium',
             rate: 0.9, volume: 1, roundSeconds: 60, reducedMotion: false},
  load() {
    try { return {...this.defaults, ...JSON.parse(localStorage.getItem('arena.settings') || '{}')}; }
    catch { return {...this.defaults}; }
  },
  save(settings) { localStorage.setItem('arena.settings', JSON.stringify(settings)); },
};

const ArenaAPI = {
  async post(path, body) {
    const response = await fetch(path, {method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body || {})});
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
    return data;
  },
  async get(path) {
    const response = await fetch(path);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
    return data;
  },
};

function arenaToast(message, ms = 2600) {
  let toast = document.querySelector('.toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.className = 'toast';
    document.body.append(toast);
  }
  toast.textContent = message;
  toast.style.opacity = '1';
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => { toast.style.opacity = '0'; }, ms);
}

/* --------------------------------------------------------------------------
 * ArenaCamera: video + native MediaPipe + rolling landmark window + HUD.
 * -------------------------------------------------------------------------- */
class ArenaCamera {
  constructor({video, canvas, hud, windowSize = 24, onFrame} = {}) {
    this.video = video;
    this.canvas = canvas;
    this.hud = hud;
    this.windowSize = windowSize;
    this.onFrame = onFrame || (() => {});
    this.stream = null;
    this.running = false;
    this.landmarker = {hands: null, face: null, pose: null};
    this.window = [];           // [{hands, face, pose, at}]
    this.fps = 0;
    this.latency = 0;
    this.detection = {hand: false, face: false, pose: false};
    this._lastVideoTime = -1;
    this._frameCount = 0;
    this._fpsAt = 0;
    this._raf = null;
  }

  async start() {
    if (this.running) return;
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('Camera needs localhost or HTTPS. Open the site at 127.0.0.1.');
    }
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {width: {ideal: 1280}, height: {ideal: 720}, frameRate: {ideal: 30}, facingMode: 'user'},
    });
    this.video.srcObject = this.stream;
    await this.video.play();
    await this._loadLandmarkers();
    this.running = true;
    this._loop();
  }

  stop() {
    this.running = false;
    if (this._raf) cancelAnimationFrame(this._raf);
    this._raf = null;
    this.stream?.getTracks().forEach(track => track.stop());
    this.stream = null;
    this.video.srcObject = null;
    this.window = [];
    this._renderHud(false);
  }

  async _loadLandmarkers() {
    const vision = await import('/static/wasm/vision_bundle.mjs');
    const fileset = await vision.FilesetResolver.forVisionTasks('/static/wasm');
    const make = (Create, modelPath, options) => Create.createFromOptions(fileset, {
      runningMode: 'VIDEO',
      baseOptions: {modelAssetPath: modelPath, delegate: 'GPU'},
      ...options,
    });
    const load = async (create) => {
      try { return await create('GPU'); }
      catch { return await create('CPU'); }
    };
    this.landmarker.hands = await load((delegate) => make(vision.HandLandmarker,
      '/static/models/hand_landmarker.task',
      {numHands: 2, minHandDetectionConfidence: 0.5, minHandPresenceConfidence: 0.5,
       minTrackingConfidence: 0.5, delegate}));
    this.landmarker.face = await load((delegate) => make(vision.FaceLandmarker,
      '/static/models/face_landmarker.task',
      {numFaces: 1, minFaceDetectionConfidence: 0.35, minFacePresenceConfidence: 0.45,
       minTrackingConfidence: 0.45, delegate}));
    this.landmarker.pose = await load((delegate) => make(vision.PoseLandmarker,
      '/static/models/pose_landmarker_lite.task',
      {numPoses: 1, minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5,
       minTrackingConfidence: 0.5, delegate}));
    this.canvas.width = 640;
    this.canvas.height = Math.round(640 * this.video.videoHeight / this.video.videoWidth);
  }

  _loop = () => {
    if (!this.running) return;
    requestAnimationFrame(this._loop);
    if (this.video.readyState < 2 || this.video.currentTime === this._lastVideoTime) return;
    this._lastVideoTime = this.video.currentTime;
    const started = performance.now();
    this._frameCount += 1;
    const now = performance.now();
    if (now - this._fpsAt >= 1000) {
      this.fps = Math.round(this._frameCount * 1000 / (now - this._fpsAt));
      this._frameCount = 0;
      this._fpsAt = now;
    }
    const ctx = this.canvas.getContext('2d', {alpha: false});
    const settings = ArenaSettings.load();
    ctx.save();
    if (settings.mirror) {
      ctx.translate(this.canvas.width, 0);
      ctx.scale(-1, 1);
    }
    ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
    ctx.restore();

    let hands = [], face = [], pose = [];
    try {
      const handResult = this.landmarker.hands?.detectForVideo(this.canvas, now);
      hands = (handResult?.landmarks || []).map(points => points.map(p => [p.x, p.y]));
      const faceResult = this.landmarker.face?.detectForVideo(this.canvas, now);
      face = faceResult?.faceLandmarks?.[0]?.map(p => [p.x, p.y]) || [];
      const poseResult = this.landmarker.pose?.detectForVideo(this.canvas, now);
      pose = poseResult?.landmarks?.[0]?.map(p => [p.x, p.y]) || [];
    } catch (error) {
      console.warn('landmark frame failed', error);
    }
    this.detection = {hand: hands.length > 0, face: face.length > 400,
                      pose: pose.length > 20};
    this.window.push({hands, face, pose, at: Date.now()});
    while (this.window.length > this.windowSize) this.window.shift();
    this._drawOverlay(hands, face, pose);
    this.latency = Math.round(performance.now() - started);
    this._renderHud(true);
    this.onFrame({hands, face, pose, fps: this.fps, latency: this.latency,
                  detection: this.detection});
  };

  _drawOverlay(hands, face, pose) {
    const ctx = this.canvas.getContext('2d');
    const W = this.canvas.width, H = this.canvas.height;
    const line = (points, a, b, color, width) => {
      ctx.strokeStyle = color; ctx.lineWidth = width;
      ctx.beginPath();
      ctx.moveTo(points[a][0] * W, points[a][1] * H);
      ctx.lineTo(points[b][0] * W, points[b][1] * H);
      ctx.stroke();
    };
    for (const hand of hands) {
      ctx.strokeStyle = '#5eead4cc'; ctx.lineWidth = 2;
      for (const [a, b] of window.HAND_CONNECTIONS || []) line(hand, a, b, '#5eead4cc', 2);
      ctx.fillStyle = '#eef2f9';
      for (const [x, y] of hand) {
        ctx.beginPath(); ctx.arc(x * W, y * H, 3, 0, Math.PI * 2); ctx.fill();
      }
    }
    if (face.length > 400) {
      ctx.strokeStyle = '#818cf855'; ctx.lineWidth = 1;
      for (const [x, y] of face) {
        ctx.beginPath(); ctx.arc(x * W, y * H, 1, 0, Math.PI * 2); ctx.stroke();
      }
    }
    if (pose.length > 20) {
      const bones = [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[0,11],[0,12]];
      for (const [a, b] of bones) if (pose[a] && pose[b]) line(pose, a, b, '#f472b6cc', 2.5);
      ctx.fillStyle = '#f472b6';
      for (const index of [0, 11, 12, 13, 14, 15, 16, 23, 24]) {
        const [x, y] = pose[index];
        ctx.beginPath(); ctx.arc(x * W, y * H, 4, 0, Math.PI * 2); ctx.fill();
      }
    }
  }

  _renderHud(active) {
    if (!this.hud) return;
    const d = this.detection;
    const warn = !active ? '⚠ Camera off' :
      (!d.hand && !d.face && !d.pose) ? '⚠ Move into camera frame' : null;
    this.hud.innerHTML = `
      <span class="pill"><span class="live-dot ${active ? 'on' : ''}"></span> ${active ? 'CAMERA ACTIVE' : 'CAMERA OFF'}</span>
      <span class="pill">FPS: <b>${this.fps}</b></span>
      <span class="pill">LATENCY: <b>${this.latency} ms</b></span>
      <span class="pill">HAND: <b>${d.hand ? 'DETECTED' : '—'}</b></span>
      <span class="pill">FACE: <b>${d.face ? 'DETECTED' : '—'}</b></span>
      <span class="pill">POSE: <b>${d.pose ? 'DETECTED' : '—'}</b></span>
      ${warn ? `<span class="warn-pill">${warn}</span>` : ''}
    `;
    const banner = this.video.closest('.camera-shell')?.querySelector('.warn-banner');
    if (banner) {
      banner.hidden = !(active && !d.hand && !d.face && !d.pose);
      banner.textContent = '⚠ Move into the camera frame';
    }
  }

  snapshot() {
    return {frames: this.window.map(f => ({hands: f.hands, face: f.face, pose: f.pose}))};
  }
}

/* Shared nav */
function arenaNav(active) {
  const links = [['/', 'HOME'], ['/communicator', 'COMMUNICATOR'],
                 ['/charades', 'CHARADES'], ['/memes', 'MEME CHALLENGE'],
                 ['/history', 'HISTORY'], ['/settings', 'SETTINGS'], ['/lab', 'AI LAB']];
  return `<nav class="arena-nav">
    <span class="brand"><span class="dot"></span> AI SIGN &amp; EXPRESSION ARENA</span>
    ${links.map(([href, label]) =>
      `<a href="${href}" class="${href === active ? 'active' : ''}">${label}</a>`).join('')}
  </nav>`;
}

function speak(text, rate) {
  if (!('speechSynthesis' in window)) return false;
  const settings = ArenaSettings.load();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = 'en-US';
  utterance.rate = rate || settings.rate || 0.9;
  utterance.volume = settings.volume ?? 1;
  window.speechSynthesis.speak(utterance);
  return true;
}

window.ArenaSettings = ArenaSettings;
window.ArenaAPI = ArenaAPI;
window.ArenaCamera = ArenaCamera;
window.arenaNav = arenaNav;
window.arenaToast = arenaToast;
window.speak = speak;
