const {test} = require('node:test');
const assert = require('node:assert/strict');
const {containedVideoRect, easeHands, letterboxLandmarks} = require('./web/static/hand_geometry');

test('overlay matches a 4:3 video in a wide camera panel', () => {
  const rect = containedVideoRect(1000, 500, 640, 480);
  assert.ok(Math.abs(rect.left - 166.6666666667) < 1e-6);
  assert.equal(rect.top, 0);
  assert.ok(Math.abs(rect.width - 666.6666666667) < 1e-6);
  assert.equal(rect.height, 500);
});

test('overlay matches pillarboxed and landscape video exactly', () => {
  assert.deepEqual(containedVideoRect(500, 800, 1280, 720),
    {left: 0, top: 259.375, width: 500, height: 281.25});
  assert.equal(containedVideoRect(500, 800, 0, 720), null);
});

test('landmarks ease toward a stable target without a one-frame jump', () => {
  assert.deepEqual(easeHands([[[0, 0]]], [[[1, 1]]], 0.5), [[[0.5, 0.5]]]);
  assert.deepEqual(easeHands([], [[[1, 1]]], 0.5), [[[1, 1]]]);
});

test('recognition landmarks match the original letterboxed 640 by 480 camera frames', () => {
  assert.deepEqual(letterboxLandmarks([[0, 0], [0.5, 0.5], [1, 1]], 1280, 720),
    [[0, 0.125], [0.5, 0.5], [1, 0.875]]);
  assert.deepEqual(letterboxLandmarks([[0, 0], [1, 1]], 640, 480),
    [[0, 0], [1, 1]]);
});
