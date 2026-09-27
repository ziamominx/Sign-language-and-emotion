/* Accept clear, repeated fingerspelling guesses only once per hand presentation. */
class LetterStability {
  constructor() {
    this.candidate = '';
    this.count = 0;
    this.acceptedSinceRelease = new Set();
  }

  resetCandidate() {
    this.candidate = '';
    this.count = 0;
  }

  rearm() {
    this.resetCandidate();
    this.acceptedSinceRelease.clear();
  }

  markAccepted(label) {
    if (typeof label === 'string') this.acceptedSinceRelease.add(label);
    this.resetCandidate();
  }

  observe(result, now = Date.now()) {
    if (!result || !result.visible) {
      this.rearm();
      return null;
    }
    const [first, second] = result.suggestions || [];
    if (result.uncertain || !first || !second || typeof first.label !== 'string' ||
        !Number.isFinite(first.score) || !Number.isFinite(second.score)) {
      this.resetCandidate();
      return null;
    }
    const label = first.label;
    const autoAllowed = /^[A-Z]$/.test(label) && label !== 'J' && label !== 'Z';
    if (!autoAllowed || first.score < 0.85 || first.score - second.score < 0.30) {
      this.resetCandidate();
      return null;
    }
    if (this.candidate === label) this.count += 1;
    else {
      this.candidate = label;
      this.count = 1;
    }
    if (this.count < 4 || this.acceptedSinceRelease.has(label)) return null;
    this.markAccepted(label, now);
    return label;
  }
}

if (typeof module !== 'undefined' && module.exports) module.exports = LetterStability;
else window.LetterStability = LetterStability;
