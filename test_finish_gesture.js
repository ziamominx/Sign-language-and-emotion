const test = require('node:test');
const assert = require('node:assert/strict');
const FinishGesture = require('./web/static/finish_gesture');

test('two open hands speak once after a deliberate hold', () => {
  const gesture = new FinishGesture(1500);
  assert.equal(gesture.observe(true, true, 100).finished, false);
  assert.equal(gesture.observe(true, true, 900).finished, false);
  assert.equal(gesture.observe(true, true, 1600).finished, true);
  assert.equal(gesture.observe(true, true, 1900).finished, false);
  gesture.observe(false, true, 2000);
  assert.equal(gesture.observe(true, true, 2100).finished, false);
  assert.equal(gesture.observe(true, true, 3600).finished, true);
});

test('no message cannot trigger finish', () => {
  const gesture = new FinishGesture();
  gesture.observe(true, false, 0);
  assert.equal(gesture.observe(true, true, 2000).finished, false);
});
