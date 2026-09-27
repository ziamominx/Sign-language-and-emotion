/* Wait for repeated model agreement before automatically adding an ASL word. */
class SignStability {
  constructor() {
    this.candidate = '';
    this.count = 0;
    this.lastAccepted = '';
    this.lastAcceptedAt = 0;
  }

  resetCandidate() {
    this.candidate = '';
    this.count = 0;
  }

  rearm() {
    this.resetCandidate();
    this.lastAccepted = '';
  }

  markAccepted(label, now = Date.now()) {
    this.lastAccepted = label;
    this.lastAcceptedAt = now;
    this.resetCandidate();
  }

  observe(result, now = Date.now()) {
    if (!result.visible) {
      this.rearm();
      return null;
    }
    const [first, second] = result.suggestions || [];
    if (result.uncertain || !first || !second) {
      this.resetCandidate();
      return null;
    }
    const margin = first.score - second.score;
    const strong = first.score >= 0.75 && margin >= 0.25;
    const moderate = first.score >= 0.55 && margin >= 0.15;
    if (!strong && !moderate) {
      this.resetCandidate();
      return null;
    }
    if (this.candidate === first.label) this.count += 1;
    else {
      this.candidate = first.label;
      this.count = 1;
    }
    const needed = strong ? 2 : 3;
    if (this.count < needed || first.label === this.lastAccepted || now - this.lastAcceptedAt < 1800) return null;
    return first.label;
  }
}

if (typeof module !== 'undefined' && module.exports) module.exports = SignStability;
else window.SignStability = SignStability;
