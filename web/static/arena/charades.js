/* Charades: game loop, AI brain rendering, feedback, and AI actor mode. */
document.querySelector('#nav').innerHTML = arenaNav('/charades');

const camera = new ArenaCamera({
  video: document.querySelector('#camera'),
  canvas: document.querySelector('#overlay'),
  hud: document.querySelector('#hud'),
  windowSize: 24,
});

let aiGuessesYou = true;   // Mode A: human acts, AI guesses
let roundRunning = false;
let observing = false;
let pollTimer = null;

const $ = selector => document.querySelector(selector);

$('#start-camera').addEventListener('click', async () => {
  if (camera.running) {
    camera.stop();
    $('#shell').classList.remove('active');
    $('#placeholder').hidden = false;
    $('#start-camera').textContent = 'Camera on';
    return;
  }
  try {
    await camera.start();
    $('#shell').classList.add('active');
    $('#placeholder').hidden = true;
    $('#start-camera').textContent = 'Camera off';
    camera.onFrame = () => {
      if (roundRunning && aiGuessesYou && !observing) {
        observing = true;
        observe().finally(() => { observing = false; });
      }
    };
  } catch (error) {
    arenaToast(`Camera unavailable: ${error.message}`);
  }
});

$('#mode-toggle').addEventListener('click', () => {
  aiGuessesYou = !aiGuessesYou;
  $('#mode-toggle').textContent = aiGuessesYou ? 'Mode: AI GUESSES' : 'Mode: AI ACTS';
  $('#ai-actor-panel').hidden = aiGuessesYou;
  $('#guess-panel').hidden = !aiGuessesYou || !roundRunning;
});

$('#start-round').addEventListener('click', async () => {
  try {
    const data = await ArenaAPI.post('/api/arena/charades/start',
      {actor_team: $('#actor-team').value});
    roundRunning = true;
    if (!camera.running) $('#start-camera').click();
    renderGame(data.game, data.secret_movie, data.ai_script);
    if (aiGuessesYou) $('#guess-panel').hidden = true;
    arenaToast('Round started — act it out!');
    if (!aiGuessesYou) showAIScript(data.ai_script);
  } catch (error) {
    arenaToast(error.message);
  }
});

$('#correct-btn').addEventListener('click', () => sendFeedback(true));
$('#wrong-btn').addEventListener('click', () => sendFeedback(false));
$('#timeout-btn').addEventListener('click', async () => {
  try { const data = await ArenaAPI.post('/api/arena/charades/timeout');
        endRound(data.game); } catch (error) { arenaToast(error.message); }
});
$('#reset-game').addEventListener('click', async () => {
  try { const data = await ArenaAPI.post('/api/arena/charades/reset');
        renderGame(data.game, null); endRound(data.game); } catch (error) { arenaToast(error.message); }
});
$('#submit-guess').addEventListener('click', () => {
  const guess = $('#human-guess').value.trim();
  if (!guess) return;
  const secret = $('#secret-title').textContent;
  const correct = guess.toLowerCase().replace(/[^a-z]/g, '') ===
                  secret.toLowerCase().replace(/[^a-z]/g, '');
  arenaToast(correct ? '✓ Correct!' : `✕ It was "${secret}"`);
  if (roundRunning) sendFeedback(correct);
});

async function observe() {
  try {
    const data = await ArenaAPI.post('/api/arena/charades/observe', camera.snapshot());
    renderBrain(data.agent);
    if (data.game && !data.game.round_active) { endRound(data.game); return; }
    if (data.ai_guess) renderGuess(data.ai_guess);
    if (data.detections?.length) {
      const top = data.detections[0];
      const center = $('#center-label');
      center.hidden = false;
      center.textContent = `Seeing: ${top.label.toLowerCase()} ${Math.round(top.confidence * 100)}%`;
    }
  } catch { /* transient */ }
}

async function sendFeedback(correct) {
  try {
    const data = await ArenaAPI.post('/api/arena/charades/feedback', {correct});
    const result = data.game.last_result;
    if (correct && result && result.elapsed <= 10) {
      arenaToast(`✓ Correct within 10s — ${result.points} points!`);
    } else {
      arenaToast(correct ? '✓ Correct — 1 point' : '✕ Wrong — point to the acting team');
    }
    endRound(data.game);
  } catch (error) { arenaToast(error.message); }
}

function endRound(game) {
  roundRunning = false;
  renderGame(game, null);
  $('#guess-panel').hidden = true;
  $('#secret').hidden = true;
  $('#center-label').hidden = true;
  if (game.last_result) {
    const result = game.last_result;
    $('#brain-status').textContent =
      `Last round: ${result.outcome.toUpperCase()} — movie was "${result.movie}"` +
      (result.winner ? ` · point to ${result.winner}` : '');
  }
  if (game.finished) {
    const [a, b] = game.teams;
    arenaToast(`Match over — final score ${a.score} : ${b.score}. Reset to play again.`);
  }
}

function renderGame(game, secretMovie, aiScript) {
  const [a, b] = game.teams;
  const teamA = $('#team-a'), teamB = $('#team-b');
  teamA.querySelector('.pts').textContent = a.score;
  teamB.querySelector('.pts').textContent = b.score;
  teamA.querySelector('.small').textContent = `streak ${a.streak}`;
  teamB.querySelector('.small').textContent = `streak ${b.streak}`;
  teamA.classList.toggle('active', game.actor_team === 'TEAM A');
  teamB.classList.toggle('active', game.actor_team === 'TEAM B');
  $('#timer').textContent = game.round_active ? game.time_remaining : '—';
  $('#timer').classList.toggle('low', game.round_active && game.time_remaining <= 10);
  $('#timeout-btn').disabled = !game.round_active;
  $('#start-round').disabled = game.round_active;
  $('#guess-panel').hidden = !aiGuessesYou || !game.round_active;
  $('#actor-line').textContent = game.round_active
    ? `Round ${game.round}/${game.total_rounds} — ${game.actor_team} acts, ${game.guessing_team} guesses.`
    : 'Pick the acting team, then start the round. Only the actor sees the movie.';
  if (secretMovie && game.round_active) {
    $('#secret').hidden = false;
    $('#secret-title').textContent = secretMovie;
    $('#secret-hints').textContent = aiScript
      ? `AI-actor script if you need it: ${aiScript.join(', ')}` : 'Act it out — no words!';
  }
  clearInterval(pollTimer);
  if (game.round_active) {
    pollTimer = setInterval(async () => {
      try {
        const data = await ArenaAPI.get('/api/arena/charades/state');
        const remaining = data.game.time_remaining;
        $('#timer').textContent = game.round_active ? remaining : '—';
        $('#timer').classList.toggle('low', remaining <= 10);
        if (!data.game.round_active) endRound(data.game);
      } catch { /* ignore */ }
    }, 1000);
  }
}

function renderBrain(agent) {
  $('#brain-status').textContent = agent.should_guess
    ? 'Confident — preparing a guess…' : 'Observing. Keep acting!';
  $('#thinking').hidden = !agent.candidates.length;
  const observed = $('#observed');
  observed.innerHTML = agent.concepts.length
    ? agent.concepts.map((item, index) =>
        `<span class="chip ${index === 0 ? 'hot' : ''}">${item.label} · ${Math.round(item.share * 100)}%</span>`).join('')
    : '<span class="chip">—</span>';
  $('#candidates').innerHTML = agent.candidates.length
    ? agent.candidates.map(item => `
      <div class="bar-row">
        <span class="name">${item.title}</span>
        <span class="pct">${Math.round(item.confidence * 100)}%</span>
        <div class="meter"><i style="width:${Math.round(item.confidence * 100)}%"></i></div>
      </div>`).join('')
    : '<p class="small muted" style="margin:.2rem 0">No strong candidates yet.</p>';
}

function renderGuess(guess) {
  $('#guess-panel').hidden = false;
  $('#guess-title').textContent = guess.title;
  $('#guess-bar').style.width = `${Math.round(guess.confidence * 100)}%`;
  $('#guess-alternatives').textContent = 'AI considered: ' + guess.candidates
    .map(item => `${item.title} ${Math.round(item.confidence * 100)}%`).join(' · ');
}

function showAIScript(script) {
  $('#ai-actor-panel').hidden = false;
  $('#ai-script').innerHTML = script.map((concept, index) =>
    `<div class="step"><span class="n">${index + 1}</span><span>${concept}</span></div>`).join('');
}

window.addEventListener('beforeunload', () => { camera.stop(); clearInterval(pollTimer); });
