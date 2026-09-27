"""Example-based recognition of isolated signs from MediaPipe hand landmarks.

This is a personal vocabulary recognizer, not a pretrained ASL translator.
Examples are kept locally as JSON so users can inspect and remove them.
"""

import json
from pathlib import Path

import numpy as np


FRAMES = 24
MIN_EXAMPLES = 3
MAX_DISTANCE = 0.30
MIN_MARGIN = 1.25


def _resample(points):
    old = np.linspace(0, 1, len(points))
    new = np.linspace(0, 1, FRAMES)
    return np.stack([np.interp(new, old, points[:, i]) for i in range(points.shape[1])], axis=1)


def features(clip):
    """Normalize a clip while retaining hand motion and two-hand spacing."""
    if len(clip) < 8:
        raise ValueError("A sign needs at least eight tracked frames")
    counts = [len(frame) for frame in clip]
    hands = max(set(counts), key=counts.count)
    if hands not in (1, 2) or counts.count(hands) < len(clip) * 0.8:
        raise ValueError("Keep one or two hands visible throughout the sign")
    clip = [frame for frame in clip if len(frame) == hands]
    frames = []
    for frame in clip:
        pts = np.asarray(frame, dtype=np.float32)
        if pts.shape != (hands, 21, 2) or not np.isfinite(pts).all():
            raise ValueError("Expected 21 finite (x, y) landmarks per hand")
        frames.append(pts[np.argsort(pts[:, 0, 0])])
    data = np.stack(frames)
    anchor = np.median(data[:3, 0, 0], axis=0)
    palm = np.linalg.norm(data[:3, 0, 5] - data[:3, 0, 17], axis=1).mean()
    if palm < 0.015:
        raise ValueError("Hand is too small or obscured")
    normalized = (data - anchor) / palm
    return _resample(normalized.reshape(len(data), -1)).astype(np.float32), hands


class SignLibrary:
    def __init__(self, directory="sign_examples"):
        self.directory = Path(directory)
        self.examples = {}
        self.reload()

    def reload(self):
        self.examples = {}
        if not self.directory.exists():
            return
        for path in sorted(self.directory.glob("*.json")):
            try:
                obj = json.loads(path.read_text(encoding="utf-8"))
                if obj.get("version") != 1 or not isinstance(obj.get("label"), str):
                    continue
                shape = np.asarray(obj["features"], dtype=np.float32)
                if shape.shape != (FRAMES, obj["hands"] * 42) or not np.isfinite(shape).all():
                    continue
                self.examples.setdefault(obj["label"], []).append((shape, obj["hands"]))
            except (OSError, ValueError, KeyError, TypeError):
                continue

    def add(self, label, clip):
        label = label.strip().upper()
        if not label or len(label) > 40 or not all(c.isalnum() or c in " -'" for c in label):
            raise ValueError("Use a short letter or word label")
        shape, hands = features(clip)
        self.directory.mkdir(parents=True, exist_ok=True)
        number = 1
        while (self.directory / f"{label.replace(' ', '_')}_{number:04}.json").exists():
            number += 1
        path = self.directory / f"{label.replace(' ', '_')}_{number:04}.json"
        path.write_text(json.dumps({"version": 1, "label": label, "hands": hands,
                                    "features": shape.tolist()}), encoding="utf-8")
        self.examples.setdefault(label, []).append((shape, hands))
        return path

    def recognize(self, clip):
        try:
            shape, hands = features(clip)
        except ValueError:
            return None
        distances = []
        for label, examples in self.examples.items():
            matches = [float(np.linalg.norm(shape.reshape(FRAMES, hands, 21, 2) -
                        sample.reshape(FRAMES, hands, 21, 2), axis=-1).mean())
                       for sample, count in examples if count == hands]
            if len(matches) >= MIN_EXAMPLES:
                distances.append((min(matches), label))
        distances.sort()
        if not distances or distances[0][0] > MAX_DISTANCE:
            return None
        if len(distances) > 1 and distances[1][0] <= max(
                distances[0][0] * MIN_MARGIN, distances[0][0] + 0.08):
            return None
        return distances[0][1]
