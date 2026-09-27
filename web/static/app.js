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
const trackingCanvas = document.createElement('canvas');
const trackingContext = trackingCanvas.getContext('2d', {alpha: false});
const emotionCanvas = document.createElement('canvas');
const emotionContext = emotionCanvas.getContext('2d', {alpha: false});
const overlay = $('#hand-overlay');
const overlayContext = overlay.getContext('2d');
let stream = null, modelReady = false, paused = false, timer = null;
let samples = [], sampling = false, analyzing = false, controller = null, generation = 0;
let tracking = false, trackingController = null, lastTrackedAt = 0, finishHolding = false;
let targetHands = [], displayedHands = [], overlayFrame = null, lastOverlayFrame = 0;
let emotionReady = false, emotionBusy = false, emotionController = null, lastEmotionAt = 0;
const stability = new SignStability();
const letterStability = new LetterStability();
const finishGesture = new FinishGesture();
const HAND_BONES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],
  [13,17],[0,17],[17,18],[18,19],[19,20]];

function fitOverlay() {
  const rect = containedVideoRect(frame.clientWidth, frame.clientHeight,
    camera.videoWidth, camera.videoHeight);
  if (!rect) return;
  overlay.style.left = `${rect.left}px`;
  overlay.style.top = `${rect.top}px`;
  overlay.style.width = `${rect.width}px`;
  overlay.style.height = `${rect.height}px`;
  overlay.width = trackingCanvas.width;
  overlay.height = trackingCanvas.height;
}

function renderHandOverlay(now) {
  if (!stream) return;
  const dt = lastOverlayFrame ? Math.min(100, now - lastOverlayFrame) : 16;
  lastOverlayFrame = now;
  const fraction = 1 - Math.exp(-dt / 45);
  displayedHands = easeHands(displayedHands, targetHands, fraction);
  overlayContext.clearRect(0, 0, overlay.width, overlay.height);
  overlayContext.lineWidth = 2.5;
  overlayContext.strokeStyle = '#d6ed76';
  overlayContext.fillStyle = '#f2f9d2';
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
  overlayContext.clearRect(0, 0, overlay.width, overlay.height);
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
    score.textContent = `${Math.round(item.score * 100)}% model score  + ADD`;
    button.append(label, score);
    button.addEventListener('click', () => recognitionMode === 'words' ? addWord(item.label) : addLetter(item.label));
    $('#suggestions').append(button);
  }
}

function clearRecognition() {
  samples = [];
  stability.resetCandidate();
  letterStability.resetCandidate();
  controller?.abort();
  trackingController?.abort();
  emotionController?.abort();
  emotionController = null;
  emotionBusy = false;
  trackingController = null;
  tracking = false;
  finishHolding = false;
  finishGesture.reset();
  $('#finish-indicator').hidden = true;
  resetHandOverlay();
  controller = null;
  generation++;
  analyzing = false;
}

function updateControls() {
  startButton.textContent = stream ? 'Stop camera' : 'Start live translation';
  startButton.disabled = !modelReady && !stream;
  pauseButton.disabled = !stream;
  pauseButton.textContent = paused ? 'Resume translation' : 'Pause translation';
  $('#camera-state').textContent = !stream ? 'Camera off' : paused ? 'Paused' : 'Translating live';
  frame.classList.toggle('active', !!stream);
  frame.classList.toggle('paused', paused);
}

function setMode(next) {
  if (next === recognitionMode || (next === 'letters' && !alphabetReady)) return;
  clearRecognition();
  recognitionMode = next;
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
    stream = await navigator.mediaDevices.getUserMedia({audio: false, video: {width: {ideal: 960}, height: {ideal: 720}, aspectRatio: {ideal: 4 / 3}, facingMode: 'user'}});
    camera.srcObject = stream;
    await camera.play();
    trackingCanvas.width = Math.max(320, Math.ceil(120 * camera.videoWidth / camera.videoHeight));
    trackingCanvas.height = Math.round(trackingCanvas.width * camera.videoHeight / camera.videoWidth);
    emotionCanvas.width = 480;
    emotionCanvas.height = Math.round(480 * camera.videoHeight / camera.videoWidth);
    fitOverlay();
    lastOverlayFrame = 0;
    overlayFrame = requestAnimationFrame(renderHandOverlay);
    stream.getVideoTracks()[0].addEventListener('ended', stopCamera, {once: true});
    $('#camera-placeholder').hidden = true;
    paused = false;
    updateControls();
    $('#emotion-panel').hidden = !emotionReady || !$('#show-emotion').checked;
    $('#live-guess').textContent = recognitionMode === 'letters' ? 'Watching for a hand' : 'Watching for a sign';
    $('#live-detail').textContent = recognitionMode === 'letters' ? 'Show one hand to spell a letter' : 'Sign one word at a time';
    announce(recognitionMode === 'letters'
      ? 'Live fingerspelling started. Show one hand in the camera area.'
      : 'Live translation started. Sign in the camera area.');
    timer = setInterval(sampleFrame, 110);
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
  timer = null;
  clearRecognition();
  if (overlayFrame !== null) cancelAnimationFrame(overlayFrame);
  overlayFrame = null;
  $('#emotion-panel').hidden = true;
  stability.rearm();
  letterStability.rearm();
  stream?.getTracks().forEach(track => track.stop());
  stream = null;
  camera.srcObject = null;
  paused = false;
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
    if (!tracking && Date.now() - lastTrackedAt >= 110) {
      lastTrackedAt = Date.now();
      trackHands();
    }
    if (emotionReady && $('#show-emotion').checked && !emotionBusy &&
        Date.now() - lastEmotionAt >= 3000) {
      lastEmotionAt = Date.now();
      recognizeEmotion();
    }
    const scale = Math.min(canvas.width / camera.videoWidth, canvas.height / camera.videoHeight);
    const width = camera.videoWidth * scale, height = camera.videoHeight * scale;
    context.fillStyle = '#000';
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.drawImage(camera, (canvas.width - width) / 2, (canvas.height - height) / 2, width, height);
    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.65));
    if (!blob || !stream || paused) return;
    samples.push(blob);
    if (samples.length > 16) samples.shift();
    if (!analyzing && !finishHolding) {
      if (recognitionMode === 'letters') recognizeLetter(blob);
      else if (samples.length === 16) recognizeLive([...samples]);
    }
  } catch (error) {
    announce(`Could not read camera: ${error.message}`, true);
  } finally {
    sampling = false;
  }
}

async function trackHands() {
  tracking = true;
  const run = generation;
  trackingContext.drawImage(camera, 0, 0, trackingCanvas.width, trackingCanvas.height);
  const blob = await new Promise(resolve => trackingCanvas.toBlob(resolve, 'image/jpeg', 0.7));
  if (!blob || run !== generation || !stream || paused) {
    tracking = false;
    return;
  }
  const form = new FormData();
  form.append('frame', blob, 'hands.jpg');
  trackingController = new AbortController();
  try {
    const response = await fetch('/api/track', {method: 'POST', body: form, signal: trackingController.signal});
    if (!response.ok) return;
    const data = await response.json();
    if (run !== generation || !stream || paused) return;
    targetHands = (data.hands || []).sort((a, b) => a[0][0] - b[0][0]);
    const hasMessage = Boolean(messageText());
    const wasFinishing = finishHolding;
    finishHolding = Boolean(data.finish_gesture && hasMessage);
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
  } catch (error) {
    if (error.name !== 'AbortError' && run === generation) {
      resetHandOverlay();
      finishHolding = false;
      finishGesture.reset();
      $('#finish-indicator').hidden = true;
    }
  } finally {
    if (run === generation) {
      tracking = false;
      trackingController = null;
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
    $('#emotion-panel').hidden = false;
    $('#emotion-label').textContent = data.visible ? data.label : 'No face detected';
    $('#emotion-detail').textContent = data.visible ? 'Visual estimate · may be wrong' : 'Keep your face in view';
  } catch (error) {
    if (error.name !== 'AbortError' && run === generation) {
      $('#emotion-label').textContent = 'Unavailable';
      $('#emotion-detail').textContent = 'Expression model could not respond';
      $('#emotion-panel').hidden = false;
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
  const form = new FormData();
  clip.forEach((blob, index) => form.append('frames', blob, `frame-${index}.jpg`));
  controller = new AbortController();
  try {
    const response = await fetch('/api/live', {method: 'POST', body: form, signal: controller.signal});
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
    $('#live-detail').textContent = `${Math.round(first.score * 100)}% model score · ${data.processing_ms} ms analysis`;
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
$('#mode-words').addEventListener('click', () => setMode('words'));
$('#mode-letters').addEventListener('click', () => setMode('letters'));
$('#show-emotion').addEventListener('change', () => {
  $('#emotion-panel').hidden = !stream || !$('#show-emotion').checked;
  if ($('#show-emotion').checked) lastEmotionAt = 0;
  else emotionController?.abort();
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
