/* Hold two open hands to speak the collected message once per gesture. */
class FinishGesture {
  constructor(holdMs = 1500) {
    this.holdMs = holdMs;
    this.reset();
  }

  reset() {
    this.startedAt = null;
    this.fired = false;
  }

  observe(open, hasMessage, now = Date.now()) {
    if (!open) {
      this.reset();
      return {progress: 0, finished: false};
    }
    if (!hasMessage) return {progress: 0, finished: false};
    if (this.startedAt === null) this.startedAt = now;
    const progress = Math.min(1, (now - this.startedAt) / this.holdMs);
    if (progress === 1 && !this.fired) {
      this.fired = true;
      return {progress, finished: true};
    }
    return {progress, finished: false};
  }
}

if (typeof module !== 'undefined' && module.exports) module.exports = FinishGesture;
else window.FinishGesture = FinishGesture;
