/* Native in-browser MediaPipe tracking for smooth, low-latency overlays.
 *
 * Hands and face run on the GPU/WASM stack inside the page (single
 * requestAnimationFrame loop over <video>), so the overlay no longer pays
 * JPEG upload + Flask + Python tracking round trips. Only the pose tracker
 * (needed by word recognition) still uses the server.
 */
const HAND_CONNECTIONS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17]];

/* The word model treats 0-4 as the thumb chain and 5-20 as finger chains. */
const OPEN_PALM_PAIRS = [[8, 6], [12, 10], [16, 14], [20, 18]];

function openPalmScore(landmarks) {
  if (!landmarks || landmarks.length !== 21) return 0;
  const wrist = landmarks[0];
  let extended = 0;
  for (const [tip, pip] of OPEN_PALM_PAIRS) {
    const tipDistance = Math.hypot(landmarks[tip][0] - wrist[0], landmarks[tip][1] - wrist[1]);
    const pipDistance = Math.hypot(landmarks[pip][0] - wrist[0], landmarks[pip][1] - wrist[1]);
    if (tipDistance > pipDistance * 1.18) extended += 1;
  }
  return extended / OPEN_PALM_PAIRS.length;
}

class NativeTracker {
  constructor({video, onHands, onFace} = {}) {
    this.video = video;
    this.onHands = onHands || (() => {});
    this.onFace = onFace || (() => {});
    this.ready = false;
    this.running = false;
    this.frame = null;
    this.frameContext = null;
    this.lastVideoTime = -1;
    this.landmarkerCount = 0;
  }

  async start() {
    if (this.running) return;
    const vision = await import('/static/wasm/vision_bundle.mjs');
    const fileset = await vision.FilesetResolver.forVisionTasks('/static/wasm');
    const options = {
      runningMode: 'VIDEO',
      numHands: 2,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
      baseOptions: {
        modelAssetPath: '/static/models/hand_landmarker.task',
        delegate: 'GPU',
      },
    };
    const makeHandLandmarker = () => vision.HandLandmarker.createFromOptions(fileset, options);
    const makeFaceLandmarker = () => vision.FaceLandmarker.createFromOptions(fileset, {
      runningMode: 'VIDEO',
      numFaces: 1,
      minFaceDetectionConfidence: 0.35,
      minFacePresenceConfidence: 0.45,
      minTrackingConfidence: 0.45,
      outputFaceBlendshapes: false,
      baseOptions: {
        modelAssetPath: '/static/models/face_landmarker.task',
        delegate: 'GPU',
      },
    });
    try {
      this.hands = await makeHandLandmarker();
    } catch (error) {
      console.warn('GPU hand tracking unavailable, using CPU:', error);
      options.baseOptions.delegate = 'CPU';
      this.hands = await makeHandLandmarker();
    }
    try {
      this.face = await makeFaceLandmarker();
    } catch (error) {
      console.warn('GPU face tracking unavailable, using CPU:', error);
      this.face = null;
    }
    const widest = Math.max(this.video.videoWidth || 640, 480);
    this.frame = document.createElement('canvas');
    this.frame.width = Math.max(480, Math.min(640, widest));
    this.frame.height = Math.round(this.frame.width * (this.video.videoHeight || 480) /
      (this.video.videoWidth || 640));
    this.frameContext = this.frame.getContext('2d', {alpha: false, willReadFrequently: true});
    this.ready = true;
    this.running = true;
    this.loop();
  }

  stop() {
    this.running = false;
    if (this.hands) { this.hands.close(); this.hands = null; }
    if (this.face) { this.face.close(); this.face = null; }
    this.ready = false;
  }

  loop = () => {
    if (!this.running || !this.video || this.video.readyState < 2) {
      if (this.running) requestAnimationFrame(this.loop);
      return;
    }
    if (this.video.currentTime === this.lastVideoTime) {
      if (this.running) requestAnimationFrame(this.loop);
      return;
    }
    this.lastVideoTime = this.video.currentTime;
    try {
      // Track on the mirrored frame exactly like the old server endpoint: the
      // video element displays mirrored (selfie view), so landmarks land in
      // display coordinates, and handedness labels follow MediaPipe's
      // mirrored-input convention ("Right" label = the viewer's right hand).
      this.frameContext.save();
      this.frameContext.translate(this.frame.width, 0);
      this.frameContext.scale(-1, 1);
      this.frameContext.drawImage(this.video, 0, 0, this.frame.width, this.frame.height);
      this.frameContext.restore();
      const timestamp = performance.now();
      const handResult = this.hands.detectForVideo(this.frame, timestamp);
      const hands = (handResult?.landmarks || []).map(landmarks =>
        landmarks.map(point => [point.x, point.y]));
      this.onHands(hands, handResult);
      if (this.face) {
        const faceResult = this.face.detectForVideo(this.frame, timestamp);
        const face = faceResult?.faceLandmarks?.[0]?.map(point => [point.x, point.y]) || [];
        this.onFace(face);
      }
    } catch (error) {
      console.warn('Native tracking frame failed:', error);
    }
    if (this.running) requestAnimationFrame(this.loop);
  }
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = {NativeTracker, openPalmScore, HAND_CONNECTIONS};
} else {
  window.NativeTracker = NativeTracker;
  window.openPalmScore = openPalmScore;
  window.HAND_CONNECTIONS = HAND_CONNECTIONS;
}
