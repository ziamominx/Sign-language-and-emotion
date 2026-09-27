const {test} = require('node:test');
const assert = require('node:assert/strict');
const SignStability = require('./web/static/sign_stability.js');

const result = (label, score = 0.82, runnerUp = 0.08) => ({
  visible: true,
  uncertain: false,
  suggestions: [{label, score}, {label: 'other', score: runnerUp}],
});

test('stable sign is added once, even while held', () => {
  const policy = new SignStability();
  assert.equal(policy.observe(result('HELLO'), 2000), null);
  assert.equal(policy.observe(result('HELLO'), 2700), 'HELLO');
  policy.markAccepted('HELLO', 2700);
  assert.equal(policy.observe(result('HELLO'), 3500), null);
  assert.equal(policy.observe(result('HELLO'), 4500), null);
});

test('a repeated sign is possible after the hands leave the view', () => {
  const policy = new SignStability();
  policy.observe(result('HELLO'), 2000);
  assert.equal(policy.observe(result('HELLO'), 2700), 'HELLO');
  policy.markAccepted('HELLO', 2700);
  assert.equal(policy.observe({visible: false}, 3500), null);
  assert.equal(policy.observe(result('HELLO'), 4700), null);
  assert.equal(policy.observe(result('HELLO'), 5400), 'HELLO');
});

test('weak and changing guesses do not enter the message', () => {
  const policy = new SignStability();
  assert.equal(policy.observe(result('A', 0.51, 0.2), 2000), null);
  assert.equal(policy.observe(result('B'), 2700), null);
  assert.equal(policy.observe(result('A'), 3400), null);
  assert.equal(policy.observe({visible: true, uncertain: true, suggestions: []}, 4100), null);
  assert.equal(policy.observe(result('A'), 4800), null);
});

test('moderate guesses require three matching windows and different words can follow', () => {
  const policy = new SignStability();
  const moderate = result('A', 0.61, 0.2);
  assert.equal(policy.observe(moderate, 2000), null);
  assert.equal(policy.observe(moderate, 2700), null);
  assert.equal(policy.observe(moderate, 3400), 'A');
  policy.markAccepted('A', 3400);
  assert.equal(policy.observe(result('B'), 4100), null);
  assert.equal(policy.observe(result('B'), 4400), 'B');
});
