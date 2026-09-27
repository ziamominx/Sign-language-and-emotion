"""Local, trainable coordinate templates for a signer's own vocabulary."""

import json
import re
from pathlib import Path
from threading import RLock

import numpy as np


STORE_PATH = Path(__file__).resolve().parent / ".models" / "personal_signs.json"
FRAMES = 18
MIN_EXAMPLES = 3
MAX_EXAMPLES = 12
MAX_LABELS = 100
LABEL_PATTERN = re.compile(r"^[A-Za-z][A-Za-z '\-]{0,39}$")


def validate_clip(frames):
    try:
        points = np.asarray(frames, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("Expected numeric landmark coordinates") from exc
    if (points.ndim != 3 or not 12 <= len(points) <= 32 or
            points.shape[1:] != (75, 2) or not np.isfinite(points).all()):
        raise ValueError("Expected 12–32 finite frames of 75 x/y landmarks")
    if np.any(points < 0) or np.any(points > 1):
        raise ValueError("Landmark coordinates must be between 0 and 1")
    pose = np.any(points[:, 11:17] != 0, axis=(1, 2)).mean()
    hands = np.any(points[:, 33:] != 0, axis=(1, 2)).mean()
    if min(pose, hands) < 0.7:
        raise ValueError("Keep your upper body and signing hands visible while recording")
    return points[np.linspace(0, len(points) - 1, FRAMES).astype(int)]


def coordinate_features(points):
    """Normalize to shoulder scale while retaining hand shape and body location."""
    points = np.asarray(points, dtype=np.float32) * [4 / 3, 1]
    center = (points[:, 11] + points[:, 12]) / 2
    shoulder_scale = np.maximum(np.linalg.norm(points[:, 11] - points[:, 12], axis=1), 0.08)
    features = [((points[:, [0, 13, 14, 15, 16]] - center[:, None]) /
                 shoulder_scale[:, None, None] * 0.5).reshape(len(points), -1)]
    for start in (33, 54):
        hand = points[:, start:start + 21]
        present = np.any(hand != 0, axis=(1, 2)).astype(np.float32)
        wrist = hand[:, 0]
        palm_scale = np.maximum(np.linalg.norm(hand[:, 9] - wrist, axis=1), 0.025)
        shape = np.clip((hand - wrist[:, None]) / palm_scale[:, None, None], -5, 5)
        location = np.clip((wrist - center) / shoulder_scale[:, None], -5, 5)
        features += [shape.reshape(len(points), -1) * 0.7,
                     location * present[:, None] * 2, present[:, None] * 3]
    return np.concatenate(features, axis=1).astype(np.float32)


def distance(left, right):
    """Small temporal shifts tolerate a sign starting a frame or two early."""
    return min(float(np.mean((left[max(shift, 0):FRAMES + min(shift, 0)] -
                              right[max(-shift, 0):FRAMES - max(shift, 0)]) ** 2))
               for shift in range(-2, 3))


class PersonalSigns:
    def __init__(self, path=STORE_PATH):
        self.path = Path(path)
        self.lock = RLock()
        self.examples = {}
        self.templates = {}
        self.thresholds = {}
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("version") != 1:
                raise ValueError("Unsupported personal sign data version")
            self.examples = data["examples"]
            for label in self.examples:
                self._refresh(label)

    def _refresh(self, label):
        templates = [coordinate_features(clip) for clip in self.examples[label]]
        self.templates[label] = templates
        if len(templates) >= MIN_EXAMPLES:
            within = [min(distance(template, other) for j, other in enumerate(templates)
                          if i != j) for i, template in enumerate(templates)]
            self.thresholds[label] = float(np.clip(np.percentile(within, 75) * 1.8,
                                                    0.035, 0.18))

    def summary(self):
        with self.lock:
            return [{"label": label, "examples": len(clips),
                     "ready": len(clips) >= MIN_EXAMPLES}
                    for label, clips in sorted(self.examples.items())]

    def add(self, label, frames):
        label = " ".join(str(label).strip().lower().split())
        if not LABEL_PATTERN.fullmatch(label):
            raise ValueError("Use a short sign name with letters, spaces, apostrophes or hyphens")
        clip = validate_clip(frames)
        with self.lock:
            if label not in self.examples and len(self.examples) >= MAX_LABELS:
                raise ValueError("Personal sign limit reached")
            clips = self.examples.setdefault(label, [])
            if len(clips) >= MAX_EXAMPLES:
                raise ValueError("This sign already has the maximum number of examples")
            clips.append(np.round(clip, 4).tolist())
            self._refresh(label)
            self._save()
            return {"label": label, "examples": len(clips),
                    "ready": len(clips) >= MIN_EXAMPLES}

    def remove(self, label):
        with self.lock:
            if label not in self.examples:
                raise KeyError(label)
            del self.examples[label]
            self.templates.pop(label, None)
            self.thresholds.pop(label, None)
            self._save()

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"version": 1, "examples": self.examples},
                                        separators=(",", ":")), encoding="utf-8")
        temporary.replace(self.path)

    def match(self, frames):
        try:
            query = coordinate_features(validate_clip(frames))
        except ValueError:
            return None
        with self.lock:
            ready = {label: templates for label, templates in self.templates.items()
                     if len(templates) >= MIN_EXAMPLES}
            if not ready:
                return None
            ranked = []
            for label, templates in ready.items():
                closest = min(distance(query, template) for template in templates)
                ranked.append((closest, self.thresholds[label], label))
            ranked.sort()
            best, threshold, label = ranked[0]
            if best > threshold:
                return None
            if len(ranked) > 1 and ranked[1][0] < best + max(0.015, best * 0.3):
                return None
            return {"label": label, "distance": round(best, 4),
                    "threshold": round(threshold, 4)}
