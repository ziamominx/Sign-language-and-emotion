const $ = selector => document.querySelector(selector);
const camera = $('#camera');
const frame = $('#camera-frame');
const startButton = $('#camera-button');
const pauseButton = $('#capture-button');
const status = $('#status-message');
const words = [];
let spelled = '';
let recognitionMode = 'words';
let alphabetReady = false;
const canvas = document.createElement('canvas');
canvas.width = 640;
canvas.height = 480;
const context = canvas.getContext('2d', {alpha: false});
const poseCanvas = document.createElement('canvas');
const poseContext = poseCanvas.getContext('2d', {alpha: false});
const emotionCanvas = document.createElement('canvas');
const emotionContext = emotionCanvas.getContext('2d', {alpha: false});
const overlay = $('#hand-overlay');
const overlayContext = overlay.getContext('2d');
let stream = null, modelReady = false, paused = false, timer = null, poseTimer = null, handTimer = null;
let samples = [], sampling = false, analyzing = false, controller = null, generation = 0;
let nativeTracking = false, nativeTracker = null, finishHolding = false;
let poseTracking = false, poseController = null, latestPose = null, latestPoseAt = 0, lastWordAnalysisAt = 0;
let recording = null, trainingBusy = false;
let targetHands = [], displayedHands = [], targetFace = [], displayedFace = [];
let emotionReady = false, emotionBusy = false, emotionController = null, lastEmotionAt = 0;
let emotionLabel = '', emotionScore = 0, emotionTentative = false, emotionVisible = false;
const stability = new SignStability();
const letterStability = new LetterStability();
const finishGesture = new FinishGesture();
const HAND_BONES = window.HAND_CONNECTIONS;
const FACE_PATHS = [
  [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10],
  [33,160,158,133,153,144,33], [362,385,387,263,373,380,362],
  [70,63,105,66,107], [336,296,334,293,300],
  [61,40,37,0,267,270,291,321,314,17,84,91,61],
  [168,6,197,195,5,4,1],
];
const FACE_DOTS = [...new Set(FACE_PATHS.flat())];

function fitOverlay() {
  const rect = containedVideoRect(frame.clientWidth, frame.clientHeight,
    camera.videoWidth, camera.videoHeight);
  if (!rect) return;
  overlay.style.left = `${rect.left}px`;
  overlay.style.top = `${rect.top}px`;
  overlay.style.width = `${rect.width}px`;
  overlay.style.height = `${rect.height}px`;
  overlay.width = nativeTracker?.frame?.width || 640;
  overlay.height = nativeTracker?.frame?.height || 480;
}

function renderHandOverlay(now) {
  if (!stream) return;
  const dt = lastOverlayFrame ? Math.min(100, now - lastOverlayFrame) : 16;
  lastOverlayFrame = now;
  const handFraction = 1 - Math.exp(-dt / 14);
  const faceFraction = 1 - Math.exp(-dt / 6);
  displayedHands = easeHands(displayedHands, targetHands, handFraction);
  displayedFace = targetFace.length ? easeHands(
    displayedFace.length ? [displayedFace] : [], [targetFace], faceFraction)[0] : [];
  overlayContext.clearRect(0, 0, overlay.width, overlay.height);
  if (displayedFace.length) {
    overlayContext.strokeStyle = 'rgba(225,240,220,.8)';
    overlayContext.lineWidth = 1;
    for (const path of FACE_PATHS) {
      overlayContext.beginPath();
      path.forEach((index, position) => {
        const [x, y] = displayedFace[index];
        if (position === 0) overlayContext.moveTo(x * overlay.width, y * overlay.height);
        else overlayContext.lineTo(x * overlay.width, y * overlay.height);
      });
      overlayContext.stroke();
    }
    overlayContext.fillStyle = 'rgba(240,248,232,.85)';
    for (const index of FACE_DOTS) {
      const [x, y] = displayedFace[index];
      overlayContext.beginPath();
      overlayContext.arc(x * overlay.width, y * overlay.height, 1.25, 0, Math.PI * 2);
      overlayContext.fill();
    }
    const xs = displayedFace.map(point => point[0]), ys = displayedFace.map(point => point[1]);
    const x1 = Math.max(0, Math.min(...xs) * overlay.width - 6);
    const y1 = Math.max(0, Math.min(...ys) * overlay.height - 6);
    const x2 = Math.min(overlay.width, Math.max(...xs) * overlay.width + 6);
    const y2 = Math.min(overlay.height, Math.max(...ys) * overlay.height + 6);
    overlayContext.strokeStyle = 'rgba(240,248,232,.55)';
    overlayContext.strokeRect(x1, y1, x2 - x1, y2 - y1);
    if (emotionVisible && $('#show-emotion').checked && emotionLabel && displayedFace.length > 400) {
      // Compact expression tag above the head, in the style of the older
      // desktop app — no large panel on the side.
      const caption = emotionTentative ? `${emotionLabel}?` : emotionLabel;
      overlayContext.font = '700 13px Manrope, Arial, sans-serif';
      const width = overlayContext.measureText(caption).width + 16;
      const left = Math.max(4, Math.min(overlay.width - width - 4,
        (x1 + x2 - width) / 2));
      const top = Math.max(4, y1 - 28);
      overlayContext.fillStyle = 'rgba(12,17,14,.78)';
      overlayContext.beginPath();
      overlayContext.roundRect(left, top, width, 22, 11);
      overlayContext.fill();
      overlayContext.strokeStyle = 'rgba(201,224,107,.55)';
      overlayContext.lineWidth = 1;
      overlayContext.stroke();
      overlayContext.fillStyle = emotionTentative ? '#dbe6cf' : '#dff29c';
      overlayContext.fillText(caption, left + 8, top + 15);
    }
  }
  overlayContext.lineWidth = 2;
  overlayContext.strokeStyle = '#f2f9e7';
  overlayContext.fillStyle = '#f2f9e7';
  for (const hand of displayedHands) {
    if (hand.length !== 21) continue;
    for (const [a, b] of HAND_BONES) {
      overlayContext.beginPath();
      overlayContext.moveTo(hand[a][0] * overlay.width, hand[a][1] * overlay.height);
      overlayContext.lineTo(hand[b][0] * overlay.width, hand[b][1] * overlay.height);
      overlayContext.stroke();
    }
    hand.forEach(([x, y], index) => {
      overlayContext.beginPath();
      overlayContext.arc(x * overlay.width, y * overlay.height,
        [4,8,12,16,20].includes(index) ? 4 : 2.7, 0, Math.PI * 2);
      overlayContext.fill();
    });
  }
  overlayFrame = requestAnimationFrame(renderHandOverlay);
}

function resetHandOverlay() {
  targetHands = [];
  displayedHands = [];
  targetFace = [];
  displayedFace = [];
  overlayContext.clearRect(0, 0, overlay.width, overlay.height);
}

/* The old /api/track returned display-space hands plus model-space hands and
 * the finish gesture. The native tracker produces the same shapes locally. */
function onNativeHands(hands, handResult) {
  if (!stream || paused) return;
  targetHands = [...hands].sort((a, b) => a[0][0] - b[0][0]);
  const emptyHand = () => Array.from({length: 21}, () => [0, 0]);
  const classifications = handResult?.handedness || [];
  const modelHands = {left: null, right: null};
  hands.forEach((hand, index) => {
    const label = classifications[index]?.[0]?.categoryName;
    if (label === 'Left') modelHands.right = hand.map(point => [point[0], point[1]]);
    if (label === 'Right') modelHands.left = hand.map(point => [point[0], point[1]]);
  });
  if (recognitionMode === 'words' && latestPose && Date.now() - latestPoseAt < 250) {
    const mapped = points => letterboxLandmarks(points, camera.videoWidth, camera.videoHeight);
    const points = [...(latestPose.some(([x, y]) => x !== 0 || y !== 0)
      ? mapped(latestPose) : latestPose),
      ...(modelHands.left ? mapped(modelHands.left) : emptyHand()),
      ...(modelHands.right ? mapped(modelHands.right) : emptyHand())];
    if (points.length === 75) ingestWordPoints(points);
  }
  const finishOpen = hands.length === 2 && hands.every(hand => openPalmScore(hand) === 1);
  observeFinishGesture(finishOpen);
}

function onNativeFace(face) {
  if (!stream || paused) return;
  targetFace = face;
}

function ingestWordPoints(points) {
  const now = Date.now();
  if (recording) {
    if (now >= recording.readyAt) {
      recording.frames.push(points);
      $('#record-count').textContent = `${recording.frames.length}/18 frames`;
      if (recording.frames.length === 18) {
        const completed = recording;
        recording = null;
        $('#record-progress').hidden = true;
        samples = [];
        saveRecording(completed);
      }
    }
    return;
  }
  if (trainingBusy) return;
  samples.push(points);
  if (samples.length > 18) samples.shift();
  if (samples.length === 18 && !analyzing && !finishHolding &&
      now - lastWordAnalysisAt >= 280) {
    lastWordAnalysisAt = now;
    recognizeLive([...samples]);
  }
}

function observeFinishGesture(finishOpen) {
  if (recording || trainingBusy) {
    finishHolding = false;
    finishGesture.reset();
    $('#finish-indicator').hidden = true;
    return;
  }
  const hasMessage = Boolean(messageText());
  const wasFinishing = finishHolding;
  finishHolding = Boolean(finishOpen && hasMessage);
  if (finishHolding && !wasFinishing) {
    controller?.abort();
    controller = null;
    analyzing = false;
  }
  const result = finishGesture.observe(finishHolding, hasMessage);
  $('#finish-indicator').hidden = !finishHolding;
  $('#finish-progress').textContent = `${Math.round(result.progress * 100)}%`;
  if (finishHolding) {
    samples = [];
    stability.resetCandidate();
    letterStability.resetCandidate();
    $('#live-guess').textContent = result.finished ? 'Message complete' : 'Finishing message';
    $('#live-detail').textContent = 'Hold both hands open to speak';
    if (result.finished) showSentence();
  }
}

function showSentence() {
  const text = messageText().trim();
  if (!text) return;
  const sentence = text[0].toUpperCase() + text.slice(1) + (/[.!?]$/.test(text) ? '' : '.');
  $('#sentence-output').textContent = sentence;
  $('#sentence-area').hidden = false;
  announce(speak(sentence, true) ? 'Speaking your complete message.' : 'Sentence ready. Speech is unavailable in this browser.');
}

function announce(text, error = false) {
  status.textContent = text;
  status.classList.toggle('error', error);
}

function renderWords() {
  $('#sentence-area').hidden = true;
  $('#message').replaceChildren();
  if (!words.length && !spelled) {
    const empty = document.createElement('span');
    empty.className = 'message-empty';
    empty.textContent = 'Recognized signs will appear here automatically.';
    $('#message').append(empty);
  }
  for (const word of words) {
    const chip = document.createElement('span');
    chip.className = 'word-chip';
    chip.textContent = word;
    $('#message').append(chip);
  }
  if (spelled) {
    const chip = document.createElement('span');
    chip.className = 'word-chip spelling';
    chip.textContent = spelled;
    chip.setAttribute('aria-label', `Spelling ${spelled}`);
    $('#message').append(chip);
  }
  for (const button of ['#speak-button', '#undo-button', '#clear-button']) $(button).disabled = !words.length && !spelled;
}

function messageText() {
  return [...words, ...(spelled ? [spelled] : [])].join(' ');
}

function speak(text, replace = false) {
  if (!('speechSynthesis' in window)) return false;
  if (replace) window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = 'en-US';
  utterance.rate = 0.9;
  window.speechSynthesis.speak(utterance);
  return true;
}

function addWord(label, automatic = false) {
  words.push(label);
  renderWords();
  stability.markAccepted(label);
  if (automatic && $('#auto-speak').checked) speak(label);
  announce(`${automatic ? 'Recognized' : 'Added'} “${label}”. Keep signing to continue.`);
}

function addLetter(label, automatic = false) {
  if (label === 'del') {
    spelled = spelled.slice(0, -1);
  } else if (label === 'space') {
    if (spelled) {
      words.push(spelled);
      if ($('#auto-speak').checked) speak(spelled);
      spelled = '';
    }
  } else {
    spelled += label;
  }
  letterStability.markAccepted(label);
  renderWords();
  announce(automatic ? `Added letter ${label}. Lower your hand before repeating it.` : `Updated spelling: ${spelled || 'empty'}.`);
}

function renderSuggestions(items = []) {
  $('#suggestions').replaceChildren();
  if (!items.length) {
    const empty = document.createElement('p');
    empty.className = 'suggestions-empty';
    empty.textContent = 'Live suggestions appear while you sign.';
    $('#suggestions').append(empty);
  }
  for (const item of items) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'suggestion';
    button.setAttribute('aria-label', `Add ${item.label} to message`);
    const label = document.createElement('span');
    label.textContent = item.label;
    const score = document.createElement('small');
    score.textContent = item.source === 'trained' ? 'Trained model match  + ADD'
      : item.source === 'personal' ? 'Saved coordinate match  + ADD'
      : `${Math.round(item.score * 100)}% model score  + ADD`;
    button.append(label, score);
    button.addEventListener('click', () => recognitionMode === 'words' ? addWord(item.label) : addLetter(item.label));
    $('#suggestions').append(button);
  }
}

function clearRecognition() {
  samples = [];
  recording = null;
  $('#record-progress').hidden = true;
  latestPose = null;
  latestPoseAt = 0;
  lastWordAnalysisAt = 0;
  stability.resetCandidate();
  letterStability.resetCandidate();
  controller?.abort();
  poseController?.abort();
  emotionController?.abort();
  emotionVisible = false;
  $('#emotion-live').textContent = '';
  emotionController = null;
  emotionBusy = false;
  poseController = null;
  poseTracking = false;
  finishHolding = false;
  finishGesture.reset();
  $('#finish-indicator').hidden = true;
  resetHandOverlay();
  controller = null;
  generation++;
  analyzing = false;
  updateTrainingControls();
}

function updateTrainingControls() {
  const button = $('#record-example');
  button.disabled = !stream || paused || recognitionMode !== 'words' || trainingBusy;
  button.textContent = recording ? 'Cancel recording' : 'Record example';
  const trainButton = $('#train-model');
  if (trainButton) trainButton.disabled = trainingBusy;
  if (stream && $('#training-status').textContent === 'Start the camera to record an example.') {
    $('#training-status').textContent = 'Enter a sign label and record it three times.';
  }
}

function renderPersonalSigns(signs) {
  const list = $('#personal-signs');
  list.replaceChildren();
  for (const sign of signs) {
    const item = document.createElement('li');
    const name = document.createElement('strong');
    name.textContent = sign.label;
    const count = document.createElement('span');
    count.className = sign.ready ? 'ready' : '';
    count.textContent = sign.ready ? ` · ${sign.examples} examples · ready`
      : ` · ${sign.examples}/3 examples`;
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.textContent = 'Remove';
    remove.setAttribute('aria-label', `Remove saved sign ${sign.label}`);
    remove.addEventListener('click', async () => {
      if (!window.confirm(`Remove all saved examples for “${sign.label}”?`)) return;
      const response = await fetch(`/api/personal-signs/${encodeURIComponent(sign.label)}`, {method: 'DELETE'});
      const data = await response.json();
      if (!response.ok) {
        $('#training-status').textContent = data.error || 'Could not remove the saved sign.';
        return;
      }
      loadPersonalSigns();
      $('#training-status').textContent = `Removed “${sign.label}”.`;
    });
    item.append(name, count, remove);
    list.append(item);
  }
}

async function loadPersonalSigns() {
  try {
    const response = await fetch('/api/personal-signs');
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Could not load your saved signs');
    renderPersonalSigns(data.signs);
    renderTrainedStatus(data);
  } catch (error) {
    $('#training-status').textContent = error.message;
  }
}

function renderTrainedStatus(data) {
  const progress = $('#train-progress');
  const button = $('#train-model');
  if (!progress || !button) return;
  if (data.trained) {
    const accuracy = typeof data.trained_accuracy === 'number' && data.trained_accuracy > 0
      ? ` · ${Math.round(data.trained_accuracy * 100)}% validation accuracy` : '';
    progress.textContent = `Trained model active${accuracy}`;
  } else {
    progress.textContent = data.signs?.length
      ? 'Not trained yet — record at least two signs, then train.'
      : 'Record examples, then train your own model.';
  }
  button.textContent = 'Train on my signs';
  button.disabled = trainingBusy;
}

async function pollTraining() {
  try {
    const response = await fetch('/api/personal-signs/train/status');
    const state = await response.json();
    if (!response.ok) throw new Error('Training status unavailable');
    if (state.running) {
      setTimeout(pollTraining, 800);
      return;
    }
    trainingBusy = false;
    updateTrainingControls();
    if (state.result?.ready) {
      const percent = state.result.validation_accuracy
        ? ` · ${Math.round(state.result.validation_accuracy * 100)}% validation accuracy`
        : '';
      $('#train-progress').textContent = `Trained on ${state.result.examples} examples${percent}`;
      $('#training-status').textContent = `Model trained on ${state.result.classes} signs. Sign again to test it.`;
    } else {
      $('#train-progress').textContent = 'Training failed.';
      $('#training-status').textContent = state.result?.error
        || state.result?.reason || 'Training could not run.';
    }
    loadPersonalSigns();
  } catch (error) {
    trainingBusy = false;
    updateTrainingControls();
    $('#train-progress').textContent = 'Training failed.';
    announce(error.message, true);
  }
}

async function startTraining() {
  if (trainingBusy) return;
  trainingBusy = true;
  updateTrainingControls();
  $('#train-progress').textContent = 'Training on your saved signs…';
  try {
    const response = await fetch('/api/personal-signs/train', {method: 'POST'});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Training could not start');
    pollTraining();
  } catch (error) {
    trainingBusy = false;
    updateTrainingControls();
    $('#train-progress').textContent = 'Training failed.';
    announce(error.message, true);
  }
}

async function saveRecording(record) {
  trainingBusy = true;
  updateTrainingControls();
  $('#training-status').textContent = `Saving “${record.label}” coordinates…`;
  try {
    const response = await fetch('/api/personal-signs', {method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({label: record.label, frames: record.frames})});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Could not save this example');
    loadPersonalSigns();
    $('#training-status').textContent = data.saved.ready
      ? `“${data.saved.label}” is ready. Sign it again to test recognition.`
      : `Saved example ${data.saved.examples}/3 for “${data.saved.label}”. Record it again.`;
  } catch (error) {
    $('#training-status').textContent = error.message;
  } finally {
    trainingBusy = false;
    updateTrainingControls();
  }
}

function updateControls() {
  startButton.textContent = stream ? 'Stop camera' : 'Start live translation';
  startButton.disabled = !modelReady && !stream;
  pauseButton.disabled = !stream;
  pauseButton.textContent = paused ? 'Resume translation' : 'Pause translation';
  $('#camera-state').textContent = !stream ? 'Camera off' : paused ? 'Paused' : 'Translating live';
  frame.classList.toggle('active', !!stream);
  frame.classList.toggle('paused', paused);
  updateTrainingControls();
}

function setMode(next) {
  if (next === recognitionMode || (next === 'letters' && !alphabetReady)) return;
  clearRecognition();
  recognitionMode = next;
  updateTrainingControls();
  stability.rearm();
  letterStability.rearm();
  for (const [mode, selector] of [['words', '#mode-words'], ['letters', '#mode-letters']]) {
    const button = $(selector);
    button.classList.toggle('selected', mode === next);
    button.setAttribute('aria-pressed', String(mode === next));
  }
  $('#letter-controls').hidden = next !== 'letters';
  $('#framing-hint').textContent = next === 'letters'
    ? 'Show one hand clearly in the center of the camera. Hold each letter, then lower your hand before the next.'
    : 'Keep your upper body and hands visible. Sign one word at a time and pause briefly between signs.';
  $('#mode-hint').textContent = next === 'letters'
    ? 'Experimental fingerspelling: hold one letter until it appears, then lower your hand before the next. Hold both hands open to speak the full message. J and Z need motion and are not added automatically.'
    : 'ASL words uses the pretrained 2,000-word model. Lower your hands briefly to repeat a word. Hold both hands open for 1.5 seconds to speak the full message.';
  $('#live-guess').textContent = stream ? 'Watching for a sign' : 'Camera off';
  $('#live-detail').textContent = next === 'letters' ? 'Show one hand to spell a letter' : 'Sign one word at a time';
  renderSuggestions();
  announce(next === 'letters' ? 'Fingerspelling mode is ready. Show one hand to the camera.' : 'ASL word mode is ready.');
}

async function checkAlphabet() {
  try {
    const response = await fetch('/api/alphabet/status');
    const data = await response.json();
    alphabetReady = response.ok && data.ready;
    $('#mode-letters').disabled = !alphabetReady;
    $('#mode-letters').title = alphabetReady ? 'Use the local alphabet model' : (data.error || 'Local alphabet model unavailable');
  } catch {
    $('#mode-letters').title = 'Local alphabet model unavailable';
  }
}

async function checkEmotion() {
  try {
    const response = await fetch('/api/emotion/status');
    const data = await response.json();
    emotionReady = response.ok && data.ready;
    if (!emotionReady) {
      $('#show-emotion').checked = false;
      $('#show-emotion').disabled = true;
      $('#show-emotion').title = data.error || 'Facial expression model unavailable';
    }
  } catch {
    $('#show-emotion').checked = false;
    $('#show-emotion').disabled = true;
    $('#show-emotion').title = 'Facial expression model unavailable';
  }
}

async function checkModel() {
  try {
    const response = await fetch('/api/status');
    const data = await response.json();
    if (!response.ok || !data.ready) throw new Error(data.error || 'Model unavailable');
    modelReady = true;
    $('#model-pill').textContent = `${data.labels.toLocaleString()} signs loaded`;
    $('#model-pill').classList.add('ready');
    updateControls();
    announce('Model ready. Start live translation to begin.');
  } catch (error) {
    $('#model-pill').textContent = 'Model unavailable';
    $('#model-pill').classList.add('error');
    announce(`Could not load the ASL model: ${error.message}`, true);
  }
}

async function startCamera() {
  if (!modelReady || stream) return;
  if (!navigator.mediaDevices?.getUserMedia) {
    announce('Camera access needs localhost or a secure HTTPS connection.', true);
    return;
  }
  startButton.disabled = true;
  announce('Requesting camera access…');
  try {
    stream = await navigator.mediaDevices.getUserMedia({audio: false, video: {width: {ideal: 1280}, height: {ideal: 720}, aspectRatio: {ideal: 16 / 9}, frameRate: {ideal: 30}, facingMode: 'user'}});
    camera.srcObject = stream;
    await camera.play();
    poseCanvas.width = 480;
    poseCanvas.height = Math.round(480 * camera.videoHeight / camera.videoWidth);
    emotionCanvas.width = 960;
    emotionCanvas.height = Math.round(960 * camera.videoHeight / camera.videoWidth);
    fitOverlay();
    lastOverlayFrame = 0;
    overlayFrame = requestAnimationFrame(renderHandOverlay);
    startNativeTracking();
    stream.getVideoTracks()[0].addEventListener('ended', stopCamera, {once: true});
    $('#camera-placeholder').hidden = true;
    paused = false;
    updateControls();
    $('#live-guess').textContent = recognitionMode === 'letters' ? 'Watching for a hand' : 'Watching for a sign';
    $('#live-detail').textContent = recognitionMode === 'letters' ? 'Show one hand to spell a letter' : 'Sign one word at a time';
    announce(recognitionMode === 'letters'
      ? 'Live fingerspelling started. Show one hand in the camera area.'
      : 'Live translation started. Sign in the camera area.');
    timer = setInterval(sampleFrame, 110);
    poseTimer = setInterval(trackPose, 100);
  } catch (error) {
    stream?.getTracks().forEach(track => track.stop());
    stream = null;
    camera.srcObject = null;
    updateControls();
    announce(`Camera unavailable: ${error.message}`, true);
  }
}

function stopCamera() {
  if (timer) clearInterval(timer);
  if (poseTimer) clearInterval(poseTimer);
  timer = null;
  poseTimer = null;
  stopNativeTracking();
  clearRecognition();
  if (overlayFrame !== null) cancelAnimationFrame(overlayFrame);
  overlayFrame = null;
  stability.rearm();
  letterStability.rearm();
  stream?.getTracks().forEach(track => track.stop());
  stream = null;
  camera.srcObject = null;
  paused = false;
  if (!trainingBusy) $('#training-status').textContent = 'Start the camera to record an example.';
  $('#camera-placeholder').hidden = false;
  $('#live-guess').textContent = 'Camera off';
  $('#live-detail').textContent = 'Start live translation';
  renderSuggestions();
  updateControls();
  announce('Camera stopped. Your message is still here.');
}

async function sampleFrame() {
  if (!stream || paused || sampling || camera.readyState < 2) return;
  sampling = true;
  try {
    if (recognitionMode === 'words') return;
    const scale = Math.min(canvas.width / camera.videoWidth, canvas.height / camera.videoHeight);
    const width = camera.videoWidth * scale, height = camera.videoHeight * scale;
    context.fillStyle = '#000';
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.drawImage(camera, (canvas.width - width) / 2, (canvas.height - height) / 2, width, height);
    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.65));
    if (!blob || !stream || paused) return;
    if (!analyzing && !finishHolding) recognizeLetter(blob);
  } catch (error) {
    announce(`Could not read camera: ${error.message}`, true);
  } finally {
    sampling = false;
  }
}

let legacyTracking = false, legacyCanvas = null;
let lastPoseAt = 0;

async function startNativeTracking() {
  if (nativeTracker) { stopNativeTracking(); }
  nativeTracker = new NativeTracker({video: camera, onHands: onNativeHands, onFace: onNativeFace});
  nativeTracking = true;
  try {
    await nativeTracker.start();
    fitOverlay();
  } catch (error) {
    console.error('Native tracking failed to start:', error);
    announce('Smooth hand overlay unavailable; falling back to server tracking.', true);
    nativeTracking = false;
    handTimer = setInterval(legacyTrackHands, 70);
  }
}

function stopNativeTracking() {
  nativeTracking = false;
  if (handTimer) { clearInterval(handTimer); handTimer = null; }
  nativeTracker?.stop();
  nativeTracker = null;
}

/* Server-side fallback used only when the WASM tracker cannot start. */
async function legacyTrackHands() {
  if (!stream || paused || legacyTracking || camera.readyState < 2) return;
  legacyTracking = true;
  const run = generation;
  const legacyCanvas = legacyCanvas || document.createElement('canvas');
  legacyCanvas.width = 480;
  legacyCanvas.height = Math.round(480 * camera.videoHeight / camera.videoWidth);
  const legacyContext = legacyCanvas.getContext('2d', {alpha: false});
  legacyContext.drawImage(camera, 0, 0, legacyCanvas.width, legacyCanvas.height);
  const blob = await new Promise(resolve => legacyCanvas.toBlob(resolve, 'image/jpeg', 0.7));
  if (!blob || run !== generation || !stream || paused) {
    legacyTracking = false;
    return;
  }
  const form = new FormData();
  form.append('frame', blob, 'hands.jpg');
  try {
    const response = await fetch('/api/track', {method: 'POST', body: form});
    if (!response.ok) return;
    const data = await response.json();
    if (run !== generation || !stream || paused) return;
    targetHands = (data.hands || []).sort((a, b) => a[0][0] - b[0][0]);
    if (recognitionMode === 'words' && latestPose && Date.now() - latestPoseAt < 250) {
      const emptyHand = () => Array.from({length: 21}, () => [0, 0]);
      const mapped = points => letterboxLandmarks(points, camera.videoWidth, camera.videoHeight);
      const points = [...(latestPose.some(([x, y]) => x !== 0 || y !== 0)
        ? mapped(latestPose) : latestPose),
        ...(data.model_hands?.left ? mapped(data.model_hands.left) : emptyHand()),
        ...(data.model_hands?.right ? mapped(data.model_hands.right) : emptyHand())];
      if (points.length === 75) ingestWordPoints(points);
    }
    observeFinishGesture(Boolean(data.finish_gesture));
  } catch {
    if (run === generation) resetHandOverlay();
  } finally {
    if (run === generation) legacyTracking = false;
  }
}

async function trackPose() {
  if (!stream || paused || poseTracking || camera.readyState < 2) return;
  poseTracking = true;
  const run = generation;
  poseContext.drawImage(camera, 0, 0, poseCanvas.width, poseCanvas.height);
  const blob = await new Promise(resolve => poseCanvas.toBlob(resolve, 'image/jpeg', 0.65));
  if (!blob || run !== generation || !stream || paused) {
    poseTracking = false;
    return;
  }
  const form = new FormData();
  form.append('frame', blob, 'pose.jpg');
  poseController = new AbortController();
  try {
    const response = await fetch('/api/pose-track', {method: 'POST', body: form,
      signal: poseController.signal});
    if (!response.ok) return;
    const data = await response.json();
    if (run !== generation || !stream || paused) return;
    latestPose = data.pose || Array.from({length: 33}, () => [0, 0]);
    latestPoseAt = Date.now();
  } catch (error) {
    if (error.name !== 'AbortError' && run === generation) latestPose = null;
  } finally {
    if (run === generation) {
      poseTracking = false;
      poseController = null;
    }
  }
}

async function recognizeEmotion() {
  emotionBusy = true;
  const run = generation;
  emotionContext.drawImage(camera, 0, 0, emotionCanvas.width, emotionCanvas.height);
  const blob = await new Promise(resolve => emotionCanvas.toBlob(resolve, 'image/jpeg', 0.65));
  if (!blob || run !== generation || !stream || paused) {
    emotionBusy = false;
    return;
  }
  const form = new FormData();
  form.append('frame', blob, 'face.jpg');
  emotionController = new AbortController();
  try {
    const response = await fetch('/api/emotion', {method: 'POST', body: form,
      signal: emotionController.signal});
    const data = await response.json();
    if (run !== generation || !stream || paused || !$('#show-emotion').checked) return;
    if (!response.ok) throw new Error(data.error || 'Expression analysis unavailable');
    emotionVisible = Boolean(data.visible);
    emotionLabel = data.label || '';
    emotionScore = data.score || 0;
    emotionTentative = Boolean(data.tentative);
    $('#emotion-live').textContent = emotionVisible
      ? `Estimated facial expression: ${emotionLabel}${emotionTentative ? ', tentative' : ''}` : '';
  } catch (error) {
    if (error.name !== 'AbortError' && run === generation) {
      emotionVisible = false;
      $('#emotion-live').textContent = '';
    }
  } finally {
    if (run === generation) {
      emotionBusy = false;
      emotionController = null;
    }
  }
}

async function recognizeLive(clip) {
  analyzing = true;
  const run = generation;
  controller = new AbortController();
  try {
    const response = await fetch('/api/live-landmarks', {method: 'POST',
      headers: {'Content-Type': 'application/json'}, body: JSON.stringify({frames: clip}),
      signal: controller.signal});
    const data = await response.json();
    if (run !== generation) return;
    if (!response.ok) throw new Error(data.error || 'Recognition failed');
    if (!data.visible) {
      stability.observe(data);
      $('#live-guess').textContent = 'Waiting for a sign';
      $('#live-detail').textContent = data.guidance || 'Keep your upper body and hands in view';
      renderSuggestions();
      announce(data.guidance || 'Live camera is running. Show a sign when ready.');
      return;
    }
    const [first] = data.suggestions;
    $('#live-guess').textContent = first.label;
    $('#live-detail').textContent = data.source === 'trained'
      ? `Trained model match · ${data.processing_ms} ms analysis`
      : data.source === 'personal'
      ? `Saved coordinate match · ${data.processing_ms} ms analysis`
      : `${Math.round(first.score * 100)}% model score · ${data.processing_ms} ms analysis`;
    renderSuggestions(data.suggestions);
    const accepted = stability.observe(data);
    if (data.uncertain) {
      announce('Unsure of this sign. Try signing clearly or choose a suggestion.');
      return;
    }
    if (finishHolding) return;
    if (accepted) {
      addWord(accepted, true);
    } else if (stability.candidate && stability.candidate !== stability.lastAccepted) {
      announce('Checking the sign across another moment…');
    }
  } catch (error) {
    if (error.name !== 'AbortError' && run === generation) announce(error.message, true);
  } finally {
    if (run === generation) {
      analyzing = false;
      controller = null;
    }
  }
}

async function recognizeLetter(blob) {
  analyzing = true;
  const run = generation;
  const form = new FormData();
  form.append('frame', blob, 'hand.jpg');
  controller = new AbortController();
  try {
    const response = await fetch('/api/alphabet/predict', {method: 'POST', body: form, signal: controller.signal});
    const data = await response.json();
    if (run !== generation) return;
    if (!response.ok) throw new Error(data.error || 'Fingerspelling recognition failed');
    if (finishHolding) return;
    const accepted = letterStability.observe(data);
    if (!data.visible) {
      $('#live-guess').textContent = 'Waiting for a hand';
      $('#live-detail').textContent = data.guidance || 'Show one hand to spell';
      renderSuggestions();
      announce(data.guidance || 'Show one hand to spell a letter.');
      return;
    }
    const first = data.suggestions[0];
    $('#live-guess').textContent = first.label;
    $('#live-detail').textContent = `${Math.round(first.score * 100)}% model score · ${data.processing_ms} ms analysis`;
    renderSuggestions(data.suggestions);
    if (accepted) addLetter(accepted, true);
    else announce(data.uncertain
      ? 'Unsure of this letter. Hold your hand clearly or choose a candidate.'
      : 'Checking the letter across several frames…');
  } catch (error) {
    if (error.name !== 'AbortError' && run === generation) announce(error.message, true);
  } finally {
    if (run === generation) {
      analyzing = false;
      controller = null;
    }
  }
}

startButton.addEventListener('click', () => stream ? stopCamera() : startCamera());
$('#record-example').addEventListener('click', () => {
  if (recording) {
    recording = null;
    $('#record-progress').hidden = true;
    $('#training-status').textContent = 'Recording cancelled.';
    updateTrainingControls();
    return;
  }
  if (!stream || paused || recognitionMode !== 'words' || trainingBusy) return;
  const label = $('#sign-label').value.trim();
  if (!/^[A-Za-z][A-Za-z '\-]{0,39}$/.test(label)) {
    $('#training-status').textContent = 'Enter a short sign name using letters and spaces.';
    $('#sign-label').focus();
    return;
  }
  samples = [];
  controller?.abort();
  recording = {label, readyAt: Date.now() + 800, frames: []};
  $('#record-progress').hidden = false;
  $('#record-count').textContent = 'Get ready';
  $('#training-status').textContent = `Get ready to sign “${label}”, then hold the movement in view.`;
  updateTrainingControls();
});
const trainButton = $('#train-model');
if (trainButton) trainButton.addEventListener('click', startTraining);
$('#mode-words').addEventListener('click', () => setMode('words'));
$('#mode-letters').addEventListener('click', () => setMode('letters'));
$('#show-emotion').addEventListener('change', () => {
  if ($('#show-emotion').checked) lastEmotionAt = 0;
  else {
    emotionController?.abort();
    emotionVisible = false;
    $('#emotion-live').textContent = '';
  }
});
window.addEventListener('resize', fitOverlay);
camera.addEventListener('resize', fitOverlay);
pauseButton.addEventListener('click', () => {
  if (!stream) return;
  paused = !paused;
  clearRecognition();
  updateControls();
  $('#live-guess').textContent = paused ? 'Translation paused'
    : recognitionMode === 'letters' ? 'Watching for a hand' : 'Watching for a sign';
  $('#live-detail').textContent = paused ? 'Resume when you are ready'
    : recognitionMode === 'letters' ? 'Show one hand to spell a letter' : 'Sign one word at a time';
  announce(paused ? 'Live translation paused.' : 'Live translation resumed.');
});
$('#speak-button').addEventListener('click', () => {
  if (!messageText()) return;
  announce(speak(messageText(), true) ? 'Speaking your message.' : 'This browser does not support speech output.');
});
$('#letter-space').addEventListener('click', () => addLetter('space'));
$('#letter-delete').addEventListener('click', () => addLetter('del'));
$('#undo-button').addEventListener('click', () => {
  if (spelled) spelled = spelled.slice(0, -1);
  else words.pop();
  renderWords();
  announce('Removed the last character or word.');
});
$('#clear-button').addEventListener('click', () => {
  words.length = 0;
  spelled = '';
  renderWords();
  announce('Message cleared.');
});
window.addEventListener('beforeunload', stopCamera);

renderWords();
renderSuggestions();
updateControls();
checkModel();
checkAlphabet();
checkEmotion();
loadPersonalSigns();
