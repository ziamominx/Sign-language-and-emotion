const $ = selector => document.querySelector(selector);
const camera = $('#camera');
const frame = $('#camera-frame');
const startButton = $('#camera-button');
const pauseButton = $('#capture-button');
const status = $('#status-message');
const words = [];
const canvas = document.createElement('canvas');
canvas.width = 640;
canvas.height = 480;
const context = canvas.getContext('2d', {alpha: false});
let stream = null, modelReady = false, paused = false, timer = null;
let samples = [], sampling = false, analyzing = false, controller = null, generation = 0;
const stability = new SignStability();

function announce(text, error = false) {
  status.textContent = text;
  status.classList.toggle('error', error);
}

function renderWords() {
  $('#message').replaceChildren();
  if (!words.length) {
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
  for (const button of ['#speak-button', '#undo-button', '#clear-button']) $(button).disabled = !words.length;
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
    button.addEventListener('click', () => addWord(item.label));
    $('#suggestions').append(button);
  }
}

function clearRecognition() {
  samples = [];
  stability.resetCandidate();
  controller?.abort();
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
    stream = await navigator.mediaDevices.getUserMedia({audio: false, video: {width: {ideal: 960}, height: {ideal: 720}, facingMode: 'user'}});
    camera.srcObject = stream;
    await camera.play();
    stream.getVideoTracks()[0].addEventListener('ended', stopCamera, {once: true});
    $('#camera-placeholder').hidden = true;
    paused = false;
    updateControls();
    $('#live-guess').textContent = 'Watching for a sign';
    $('#live-detail').textContent = 'Sign one word at a time';
    announce('Live translation started. Sign in the camera area.');
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
  stability.rearm();
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
    const scale = Math.min(canvas.width / camera.videoWidth, canvas.height / camera.videoHeight);
    const width = camera.videoWidth * scale, height = camera.videoHeight * scale;
    context.fillStyle = '#000';
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.drawImage(camera, (canvas.width - width) / 2, (canvas.height - height) / 2, width, height);
    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.65));
    if (!blob || !stream || paused) return;
    samples.push(blob);
    if (samples.length > 16) samples.shift();
    if (samples.length === 16 && !analyzing) recognizeLive([...samples]);
  } catch (error) {
    announce(`Could not read camera: ${error.message}`, true);
  } finally {
    sampling = false;
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

startButton.addEventListener('click', () => stream ? stopCamera() : startCamera());
pauseButton.addEventListener('click', () => {
  if (!stream) return;
  paused = !paused;
  clearRecognition();
  updateControls();
  $('#live-guess').textContent = paused ? 'Translation paused' : 'Watching for a sign';
  $('#live-detail').textContent = paused ? 'Resume when you are ready' : 'Sign one word at a time';
  announce(paused ? 'Live translation paused.' : 'Live translation resumed.');
});
$('#speak-button').addEventListener('click', () => {
  if (!words.length) return;
  announce(speak(words.join(' '), true) ? 'Speaking your message.' : 'This browser does not support speech output.');
});
$('#undo-button').addEventListener('click', () => { words.pop(); renderWords(); announce('Removed the last word.'); });
$('#clear-button').addEventListener('click', () => { words.length = 0; renderWords(); announce('Message cleared.'); });
window.addEventListener('beforeunload', stopCamera);

renderWords();
renderSuggestions();
updateControls();
checkModel();
