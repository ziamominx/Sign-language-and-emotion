/* Home page: status, recent sessions, and the guided demo. */
document.querySelector('#nav').innerHTML = arenaNav('/');

async function loadStatus() {
  try {
    const data = await ArenaAPI.get('/api/arena/status');
    document.querySelector('#stat-movies').textContent = data.movies;
    document.querySelector('#stat-memes').textContent = data.memes;
    document.querySelector('#stat-accuracy').textContent =
      data.accuracy === null || data.accuracy === undefined
        ? '—' : `${Math.round(data.accuracy * 100)}%`;
    document.querySelector('#stat-sessions').textContent = data.sessions.length;
    const recent = document.querySelector('#recent');
    if (!data.sessions.length) {
      recent.textContent = 'No sessions yet — start any mode and results appear here.';
      return;
    }
    recent.innerHTML = data.sessions.map(session => {
      const when = new Date(session.created_at * 1000).toLocaleTimeString();
      const detail = session.detail && session.detail.movie ? session.detail.movie : '';
      return `<div class="history-row"><span><span class="outcome ${session.outcome}">${session.mode}</span>
        ${detail ? `<b style="margin-left:.5rem">${detail}</b>` : ''}</span>
        <span class="when">${when}</span></div>`;
    }).join('');
  } catch (error) {
    document.querySelector('#recent').textContent = `Could not load status: ${error.message}`;
  }
}

/* Guided demo: highlights each step and navigates every 25 seconds. */
const Demo = {
  timer: null,
  index: 0,
  steps: () => [...document.querySelectorAll('#demo-steps .step')],
  start() {
    this.stop();
    this.index = 0;
    this.highlight();
    this.timer = setInterval(() => {
      this.index += 1;
      if (this.index >= this.steps().length) { this.stop(); return; }
      this.highlight();
      window.location.href = this.steps()[this.index].dataset.href;
    }, 25000);
    window.location.href = this.steps()[0].dataset.href;
  },
  highlight() {
    this.steps().forEach((step, i) => {
      step.classList.toggle('active', i === this.index);
      step.classList.toggle('done', i < this.index);
    });
  },
  stop() {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    this.steps().forEach(step => step.classList.remove('active', 'done'));
  },
};

document.querySelector('#demo-start')?.addEventListener('click', () => Demo.start());
document.querySelector('#demo-stop')?.addEventListener('click', () => Demo.stop());
document.querySelector('#demo-mode')?.addEventListener('click', () => Demo.start());
loadStatus();
