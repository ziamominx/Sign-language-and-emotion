const camera = document.querySelector('#camera');
const cameraFrame = document.querySelector('#camera-frame');
const cameraButton = document.querySelector('#camera-button');
const captureButton = document.querySelector('#capture-button');
const cameraState = document.querySelector('#camera-state');
const modelPill = document.querySelector('#model-pill');
const placeholder = document.querySelector('#camera-placeholder');
const progress = document.querySelector('#record-progress');
const countdown = document.querySelector('#countdown');
const suggestions = document.querySelector('#suggestions');
const message = document.querySelector('#message');
const speakButton = document.querySelector('#speak-button');
const undoButton = document.querySelector('#undo-button');
const clearButton = document.querySelector('#clear-button');
const statusMessage = document.querySelector('#status-message');

let stream = null;
let modelReady = false;
let capturing = false;
const words = [];

function setStatus(text, isError = false) {
  statusMessage.textContent = text;
  statusMessage.classList.toggle('error', isError);
}

function renderMessage() {
  message.replaceChildren();
  if (!words.length) {
    const empty = document.createElement('span');
    empty.className = 'message-empty';
    empty.textContent = 'Your words will appear here.';
    message.append(empty);
  } else {
    for (const word of words) {
      const chip = document.createElement('span');
      chip.className = 'word-chip';
      chip.textContent = word;
      message.append(chip);
    }
  }
  speakButton.disabled = !words.length;
  undoButton.disabled = !words.length;
  clearButton.disabled = !words.length;
}

function renderSuggestions(items) {
  suggestions.replaceChildren();
  if (!items || !items.length) {
    const empty = document.createElement('p');
    empty.className = 'suggestions-empty';
    empty.textContent = 'Capture a sign to see suggestions.';
    suggestions.append(empty);
    return;
  }
  for (const item of items) {
    const option = document.createElement('button');
    option.type = 'button';
    option.className = 'suggestion';
    option.setAttribute('aria-label', `Add ${item.label} to message`);
    const label = document.createElement('span');
    label.textContent = item.label;
    const score = document.createElement('small');
    score.textContent = `${Math.round(item.score * 100)}% model score  + ADD`;
    option.append(label, score);
    option.addEventListener('click', () => {
      words.push(item.label);
      renderMessage();
      renderSuggestions([]);
      setStatus(`Added “${item.label}”. Capture another sign or speak your message.`);
      captureButton.focus();
    });
    suggestions.append(option);
  }
}

async function checkModel() {
  try {
    const response = await fetch('/api/status');
    const data = await response.json();
    if (!response.ok || !data.ready) throw new Error(data.error || 'Model unavailable');
    modelReady = true;
    modelPill.textContent = `${data.labels.toLocaleString()} signs loaded`;
    modelPill.classList.add('ready');
    captureButton.disabled = !stream;
    setStatus('Model ready. Enable your camera to begin.');
  } catch (error) {
    modelPill.textContent = 'Model unavailable';
    modelPill.classList.add('error');
    setStatus(`Could not load the ASL model: ${error.message}`, true);
  }
}

async function enableCamera() {
  if (stream) return;
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus('Camera access needs localhost or a secure HTTPS connection.', true);
    return;
  }
  cameraButton.disabled = true;
  setStatus('Requesting camera access…');
  try {
    stream = await navigator.mediaDevices.getUserMedia({audio: false, video: {width: {ideal: 960}, height: {ideal: 720}, facingMode: 'user'}});
    camera.srcObject = stream;
    await camera.play();
    cameraFrame.classList.add('active');
    placeholder.hidden = true;
    cameraButton.textContent = 'Camera enabled';
    cameraState.textContent = 'Camera live';
    captureButton.disabled = !modelReady;
    setStatus(modelReady ? 'Frame your upper body and capture one sign.' : 'Camera ready. Waiting for model.');
  } catch (error) {
    cameraButton.disabled = false;
    setStatus(`Camera unavailable: ${error.message}`, true);
  }
}

const sleep = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));

async function captureSign() {
  if (!stream || !modelReady || capturing) return;
  capturing = true;
  captureButton.disabled = true;
  cameraButton.disabled = true;
  progress.hidden = false;
  renderSuggestions([]);
  try {
    setStatus('Get ready. Keep shoulders and hands in view.');
    countdown.textContent = 'READY';
    await sleep(750);
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 480;
    const context = canvas.getContext('2d', {alpha: false});
    const form = new FormData();
    for (let index = 0; index < 24; index++) {
      if (camera.readyState < 2) throw new Error('Camera feed stopped');
      countdown.textContent = Math.max(0, (2.4 - index * 0.1)).toFixed(1);
      const scale = Math.min(canvas.width / camera.videoWidth, canvas.height / camera.videoHeight);
      const width = camera.videoWidth * scale;
      const height = camera.videoHeight * scale;
      context.fillStyle = '#000';
      context.fillRect(0, 0, canvas.width, canvas.height);
      context.drawImage(camera, (canvas.width - width) / 2, (canvas.height - height) / 2, width, height);
      const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.72));
      if (!blob) throw new Error('Could not capture a camera frame');
      form.append('frames', blob, `frame-${index}.jpg`);
      await sleep(100);
    }
    progress.hidden = true;
    setStatus('Analyzing hand and body movement…');
    const response = await fetch('/api/recognize', {method: 'POST', body: form});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Recognition failed');
    renderSuggestions(data.suggestions);
    setStatus(data.uncertain
      ? 'Low certainty. Capture again, or choose a suggestion only if you know it is correct.'
      : 'Choose the intended word from the suggestions. Capture again if none fit.', data.uncertain);
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    progress.hidden = true;
    capturing = false;
    captureButton.disabled = !stream || !modelReady;
    cameraButton.disabled = !!stream;
  }
}

cameraButton.addEventListener('click', enableCamera);
captureButton.addEventListener('click', captureSign);
speakButton.addEventListener('click', () => {
  if (!words.length) return;
  if (!('speechSynthesis' in window)) {
    setStatus('This browser does not support speech output.', true);
    return;
  }
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(words.join(' '));
  utterance.lang = 'en-US';
  utterance.rate = 0.9;
  window.speechSynthesis.speak(utterance);
  setStatus('Speaking your message.');
});
undoButton.addEventListener('click', () => { words.pop(); renderMessage(); setStatus('Removed the last word.'); });
clearButton.addEventListener('click', () => { words.length = 0; renderMessage(); renderSuggestions([]); setStatus('Message cleared.'); });
window.addEventListener('beforeunload', () => stream?.getTracks().forEach(track => track.stop()));

renderMessage();
checkModel();
