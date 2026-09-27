/* Match a contain-fitted video and ease landmark motion between detections. */
function containedVideoRect(boxWidth, boxHeight, videoWidth, videoHeight) {
  if (!boxWidth || !boxHeight || !videoWidth || !videoHeight) return null;
  const scale = Math.min(boxWidth / videoWidth, boxHeight / videoHeight);
  const width = Math.min(boxWidth, videoWidth * scale);
  const height = Math.min(boxHeight, videoHeight * scale);
  return {left: (boxWidth - width) / 2, top: (boxHeight - height) / 2, width, height};
}

function easeHands(current, target, fraction) {
  if (current.length !== target.length || current.some((hand, i) => hand.length !== target[i].length)) {
    return target.map(hand => hand.map(point => [...point]));
  }
  return target.map((hand, h) => hand.map((point, p) => [
    current[h][p][0] + (point[0] - current[h][p][0]) * fraction,
    current[h][p][1] + (point[1] - current[h][p][1]) * fraction,
  ]));
}

/* The word model was trained on 640×480 frames with the camera letterboxed. */
function letterboxLandmarks(points, videoWidth, videoHeight) {
  const rect = containedVideoRect(640, 480, videoWidth, videoHeight);
  if (!rect) return [];
  return points.map(([x, y]) => [
    (rect.left + x * rect.width) / 640,
    (rect.top + y * rect.height) / 480,
  ]);
}

if (typeof module !== 'undefined' && module.exports) module.exports = {containedVideoRect, easeHands, letterboxLandmarks};
