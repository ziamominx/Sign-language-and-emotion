/* Meme challenge: perform, capture, score, and see the meme. */
document.querySelector('#nav').innerHTML = arenaNav('/memes');

const $ = selector => document.querySelector(selector);
const camera = new ArenaCamera({
  video: $('#camera'), canvas: $('#overlay'), hud: $('#hud'), windowSize: 22,
});
let challenge = null;

$('#start').addEventListener('click', async () => {
  if (!camera.running) {
    try {
      await camera.start();
      $('#shell').classList.add('active');
      $('#placeholder').hidden = true;
      $('#start').textContent = 'Camera ready — perform!';
      $('#capture').disabled = false;
    } catch (error) { return arenaToast(`Camera unavailable: ${error.message}`); }
  }
  arenaToast('Perform the challenge, then press "Capture my act".');
});

$('#capture').addEventListener('click', async () => {
  if (!camera.running) return arenaToast('Start the camera first.');
  $('#capture').disabled = true;
  $('#capture').textContent = 'Analyzing…';
  try {
    const data = await ArenaAPI.post('/api/arena/memes/analyze', camera.snapshot());
    renderResult(data);
  } catch (error) {
    arenaToast(error.message);
  } finally {
    $('#capture').disabled = false;
    $('#capture').textContent = 'Capture my act';
  }
});

$('#next').addEventListener('click', loadChallenge);
$('#next-after').addEventListener('click', async () => {
  await loadChallenge();
  $('#result-panel').hidden = true;
});
$('#try-again').addEventListener('click', () => {
  $('#result-panel').hidden = true;
  arenaToast('Go again — press "Capture my act" when ready.');
});

async function loadChallenge() {
  try {
    const data = await ArenaAPI.get('/api/arena/memes/challenge');
    challenge = data.challenge;
    $('#challenge-prompt').textContent = challenge.prompt;
    $('#challenge-hint').textContent = `Hint: ${challenge.hint}`;
    const label = $('#challenge-label');
    label.hidden = false;
    label.textContent = challenge.prompt;
  } catch (error) { arenaToast(error.message); }
}

function renderResult(data) {
  $('#result-panel').hidden = false;
  const score = data.score;
  $('#score-breakdown').innerHTML = [
    ['Expression', score.expression], ['Pose', score.pose],
    ['Intensity', score.intensity], ['Challenge match', score.challenge_match],
  ].map(([name, value]) => `
    <div class="bar-row"><span class="name">${name}</span>
      <span class="pct">${value}%</span>
      <div class="meter"><i style="width:${value}%"></i></div></div>`).join('');
  $('#final-pct').textContent = `${score.final} / 100`;
  $('#final-bar').style.width = `${score.final}%`;

  const features = $('#features');
  const profileEntries = Object.entries(data.profile || {});
  features.innerHTML = profileEntries.length
    ? profileEntries.map(([tag, value], index) =>
        `<span class="chip ${index === 0 ? 'hot' : ''}">${tag.replace(/_/g, ' ')} ${Math.round(value * 100)}%</span>`).join('')
    : '<span class="chip">Not enough signal — try bigger expressions</span>';
  $('#expression').textContent = data.expression
    ? `${data.expression.label} · ${Math.round(data.expression.confidence * 100)}%` : '—';
  $('#action').textContent = data.action
    ? `${data.action.label} · ${Math.round(data.action.confidence * 100)}%` : '—';

  const match = (data.matches || [])[0];
  const card = $('#meme-card');
  if (match) {
    card.innerHTML = `<div class="art"><img src="${match.image}" alt="${match.label}"></div>
      <div class="label">${match.label} · ${Math.round(match.score * 100)}% match</div>`;
  } else {
    card.innerHTML = '<div class="art">🤷</div><div class="label">No meme matched — try again with a bigger expression</div>';
  }
  if (score.final >= 70) arenaToast(`Great act! ${score.final}/100`);
  else if (score.final >= 40) arenaToast('Decent — exaggerate more for a higher score.');
  else arenaToast('The AI could barely see it. Bigger face, more movement!');
}

loadChallenge();
window.addEventListener('beforeunload', () => camera.stop());
