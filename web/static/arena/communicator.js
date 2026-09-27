/* Communicator: gesture → text → voice with a stability gate. */
document.querySelector('#nav').innerHTML = arenaNav('/communicator');

const settings = ArenaSettings.load();
document.querySelector('#rate').value = settings.rate;
document.querySelector('#volume').value = settings.volume;

const camera = new ArenaCamera({
  video: document.querySelector('#camera'),
  canvas: document.querySelector('#overlay'),
  hud: document.querySelector('#hud'),
  windowSize: 20,
});

const words = [];
let candidate = null, candidateCount = 0, lastAccepted = null;
let observing = false;
const REQUIRED_REPEATS = 2;

document.querySelector('#start').addEventListener('click', async () => {
  try {
    await camera.start();
    document.querySelector('#shell').classList.add('active');
    document.querySelector('#placeholder').hidden = true;
    document.querySelector('#start').disabled = true;
    document.querySelector('#stop').disabled = false;
    camera.onFrame = () => {
      if (!observing) { observing = true; observe().finally(() => { observing = false; }); }
    };
  } catch (error) {
    arenaToast(`Camera unavailable: ${error.message}`);
  }
});

document.querySelector('#stop').addEventListener('click', () => {
  camera.stop();
  document.querySelector('#shell').classList.remove('active');
  document.querySelector('#placeholder').hidden = false;
  document.querySelector('#start').disabled = false;
  document.querySelector('#stop').disabled = true;
});

async function observe() {
  try {
    const data = await ArenaAPI.post('/api/arena/communicator/observe',
      {...camera.snapshot(), player: 'player_1'});
    renderGesture(data.gesture);
  } catch { /* keep the loop alive on transient errors */ }
}

function renderGesture(gesture) {
  const label = document.querySelector('#detected');
  const confidence = document.querySelector('#confidence');
  const bar = document.querySelector('#confidence-bar');
  const alternatives = document.querySelector('#alternatives');
  if (!gesture) {
    label.textContent = '—'; confidence.textContent = '—';
    bar.style.width = '0%'; alternatives.textContent = '—';
    document.querySelector('#center-label').hidden = true;
    candidate = null; candidateCount = 0;
    return;
  }
  const strong = gesture.confidence >= 0.85;
  const possible = gesture.confidence >= 0.6;
  label.textContent = gesture.label;
  confidence.textContent = `${Math.round(gesture.confidence * 100)}%`;
  confidence.style.color = strong ? 'var(--good)' : possible ? 'var(--warn)' : 'var(--bad)';
  bar.style.width = `${Math.round(gesture.confidence * 100)}%`;
  alternatives.textContent = (gesture.alternatives || [])
    .map(item => `${item.label} ${Math.round(item.confidence * 100)}%`).join(' · ') || '—';
  const center = document.querySelector('#center-label');
  center.hidden = false;
  center.textContent = `${gesture.label} · ${Math.round(gesture.confidence * 100)}%`;

  if (gesture.confidence < 0.6) { candidate = null; candidateCount = 0; return; }
  if (candidate === gesture.label) candidateCount += 1;
  else { candidate = gesture.label; candidateCount = 1; }
  if (candidateCount >= REQUIRED_REPEATS && gesture.label !== lastAccepted) {
    addWord(gesture.label);
    candidate = null; candidateCount = 0;
  }
}

function addWord(word) {
  words.push(word);
  lastAccepted = word;
  renderSentence();
  if (document.querySelector('#auto-speak').checked) speak(word);
}

function renderSentence() {
  const target = document.querySelector('#sentence');
  if (!words.length) {
    target.textContent = 'Say something with your hands…';
    target.classList.add('muted');
  } else {
    target.textContent = words.join(' ');
    target.classList.remove('muted');
  }
}

document.querySelector('#speak-btn').addEventListener('click', () => {
  if (!words.length) return arenaToast('Nothing to speak yet.');
  speak(words.join(' ')) || arenaToast('This browser has no speech synthesis.');
});
document.querySelector('#replay').addEventListener('click', () => {
  if (words.length) speak(words.join(' '));
});
document.querySelector('#undo').addEventListener('click', () => {
  words.pop(); lastAccepted = null; renderSentence();
});
document.querySelector('#clear').addEventListener('click', () => {
  words.length = 0; lastAccepted = null; renderSentence();
});
document.querySelector('#wrong-btn').addEventListener('click', async () => {
  if (!lastAccepted) return arenaToast('Nothing to correct yet.');
  const actual = prompt(`The AI detected "${lastAccepted}". What did you actually sign?`);
  if (!actual) return;
  try {
    await ArenaAPI.post('/api/arena/communicator/correct',
      {player: 'player_1', guessed: lastAccepted, actual: actual.trim().toUpperCase()});
    arenaToast(`Noted — "${actual.trim().toUpperCase()}" stored for your profile.`);
  } catch (error) { arenaToast(error.message); }
});
document.querySelector('#rate').addEventListener('input', event => {
  settings.rate = parseFloat(event.target.value);
  document.querySelector('#rate-value').textContent = `${settings.rate.toFixed(2)}×`;
  ArenaSettings.save(settings);
});
document.querySelector('#volume').addEventListener('input', event => {
  settings.volume = parseFloat(event.target.value);
  ArenaSettings.save(settings);
});

(async function loadVocabulary() {
  try {
    const data = await ArenaAPI.get('/api/arena/status');
    document.querySelector('#vocab').innerHTML = GESTURE_HINTS.map(
      item => `<span class="chip" title="${item.hints}">${item.label}</span>`).join('');
  } catch { /* vocabulary list is decorative */ }
})();

const GESTURE_HINTS = [
  {label: 'HELLO', hints: 'Open palm, wave hand side to side'},
  {label: 'YES', hints: 'Fist nodding up and down'},
  {label: 'NO', hints: 'Index finger wagging side to side'},
  {label: 'PLEASE', hints: 'Flat hand circling on chest'},
  {label: 'WATER', hints: 'Pinch at chin'},
  {label: 'POINT', hints: 'Index finger extended'},
  {label: 'I LOVE YOU', hints: 'Thumb + index + pinky extended'},
];
window.addEventListener('beforeunload', () => camera.stop());
