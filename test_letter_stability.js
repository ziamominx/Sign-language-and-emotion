const {test} = require('node:test');
const assert = require('node:assert/strict');
const LetterStability = require('./web/static/letter_stability.js');

const result = (label, score = 0.9, runnerUp = 0.1) => ({
  visible: true,
  uncertain: false,
  suggestions: [{label, score}, {label: 'other', score: runnerUp}],
});

test('four consecutive strong guesses accept one letter', () => {
  const policy = new LetterStability();
  for (let i = 0; i < 3; i += 1) assert.equal(policy.observe(result('A'), i), null);
  assert.equal(policy.observe(result('A'), 3), 'A');
  assert.equal(policy.observe(result('A'), 4), null);
  assert.equal(policy.observe(result('A'), 5), null);
  for (let i = 0; i < 3; i += 1) assert.equal(policy.observe(result('B'), i + 6), null);
  assert.equal(policy.observe(result('B'), 9), 'B');
  for (let i = 0; i < 5; i += 1) assert.equal(policy.observe(result('A'), i + 10), null);
});

test('hand absence re-arms a repeated letter', () => {
  const policy = new LetterStability();
  for (let i = 0; i < 4; i += 1) policy.observe(result('A'), i);
  assert.equal(policy.observe({visible: false}, 4), null);
  for (let i = 0; i < 3; i += 1) assert.equal(policy.observe(result('A'), i + 5), null);
  assert.equal(policy.observe(result('A'), 8), 'A');
});

test('weak, ambiguous, and changing guesses restart agreement', () => {
  const policy = new LetterStability();
  assert.equal(policy.observe(result('A', 0.84), 0), null);
  assert.equal(policy.observe(result('A', 0.9, 0.65), 1), null);
  assert.equal(policy.observe(result('A'), 2), null);
  assert.equal(policy.observe(result('B'), 3), null);
  assert.equal(policy.observe(result('A'), 4), null);
  assert.equal(policy.observe({...result('A'), uncertain: true}, 5), null);
  for (let i = 0; i < 3; i += 1) assert.equal(policy.observe(result('A'), i + 6), null);
  assert.equal(policy.observe(result('A'), 9), 'A');
});

test('motion letters and controls are never automatic', () => {
  const policy = new LetterStability();
  for (const label of ['J', 'Z', 'del', 'space']) {
    for (let i = 0; i < 5; i += 1) assert.equal(policy.observe(result(label), i), null);
  }
  for (let i = 0; i < 3; i += 1) assert.equal(policy.observe(result('C'), i), null);
  assert.equal(policy.observe(result('C'), 3), 'C');
});

test('manual choice marks a letter as accepted until release', () => {
  const policy = new LetterStability();
  policy.observe(result('A'), 0);
  policy.markAccepted('A');
  for (let i = 0; i < 6; i += 1) assert.equal(policy.observe(result('A'), i + 1), null);
  policy.observe({visible: false}, 8);
  for (let i = 0; i < 3; i += 1) assert.equal(policy.observe(result('A'), i + 9), null);
  assert.equal(policy.observe(result('A'), 12), 'A');
});
