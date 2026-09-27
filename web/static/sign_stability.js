/* Wait for repeated model agreement before automatically adding an ASL word. */
class SignStability {
  constructor() {
    this.candidate = '';
    this.count = 0;
    this.lastAccepted = '';
  }

  resetCandidate() {
    this.candidate = '';
    this.count = 0;
  }

  rearm() {
    this.resetCandidate();
    this.lastAccepted = '';
  }

  markAccepted(label) {
    this.lastAccepted = label;
    this.resetCandidate();
  }

  observe(result, now = Date.now()) {
    if (!result.visible) {
      this.rearm();
      return null;
    }
    const [first, second] = result.suggestions || [];
    if (result.uncertain || !first || (!second && first.source !== 'personal')) {
      this.resetCandidate();
      return null;
    }
    const margin = first.score - (second?.score || 0);
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
    if (this.count < needed || first.label === this.lastAccepted) return null;
    return first.label;
  }
}

if (typeof module !== 'undefined' && module.exports) module.exports = SignStability;
else window.SignStability = SignStability;
