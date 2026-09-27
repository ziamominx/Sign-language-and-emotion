"""Rule-based temporal perception over landmark sequences.

The frontend tracks hands (21 points x2), face (478 points), and body pose
(33 points) natively and posts compact landmark frames here. This module
turns a rolling window of those frames into observable features — finger
extension, mouth openness, eye widening, arm elevation, motion energy —
and classifies gestures, actions, and expressions with confidence scores.

Everything is geometry over normalized coordinates, so the same rules work
for any camera resolution. This is the engine's 'eyes': later, a trained
sequence model can replace `classify_*` without touching the agents.
"""

import numpy as np

# Landmark indices (MediaPipe conventions)
FACE = {
    "left_eye_outer": 33, "left_eye_inner": 133, "right_eye_outer": 263,
    "right_eye_inner": 362, "left_eye_top": 159, "left_eye_bottom": 145,
    "right_eye_top": 386, "right_eye_bottom": 374, "mouth_left": 61,
    "mouth_right": 291, "upper_lip": 13, "lower_lip": 14, "nose": 1,
    "chin": 152, "forehead": 10, "left_brow": 105, "right_brow": 334,
    "left_cheek": 234, "right_cheek": 454,
}
POSE = {"left_shoulder": 11, "right_shoulder": 12, "left_elbow": 13,
        "right_elbow": 14, "left_wrist": 15, "right_wrist": 16,
        "left_hip": 23, "right_hip": 24, "nose": 0}
HAND_TIPS = [4, 8, 12, 16, 20]
HAND_PIPS = [2, 5, 6, 9, 10, 13, 14, 17, 18]


def _finite(points):
    return points is not None and len(points) > 0


def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


# --------------------------------------------------------------------------
# Frame-level features
# --------------------------------------------------------------------------

def hand_features(hand):
    """Finger extension pattern + palm orientation for one 21-point hand."""
    if not _finite(hand) or len(hand) < 21:
        return None
    hand = np.asarray(hand, dtype=float)
    wrist = hand[0]
    scale = max(_dist(hand[5], hand[17]), 0.02)  # palm width reference
    fingers = []
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        tip_d = _dist(hand[tip], wrist)
        pip_d = _dist(hand[pip], wrist)
        fingers.append(tip_d > pip_d * 1.15)
    thumb = _dist(hand[4], hand[17]) > scale * 1.4
    return {
        "extended": fingers + [thumb],
        "count": int(sum(fingers)) + int(thumb),
        "spread": float(np.mean([_dist(hand[t], wrist) / scale for t in HAND_TIPS])),
    }


def face_features(face):
    """Observable expression features from face landmarks (478-point mesh)."""
    if not _finite(face) or len(face) < 468:
        return None
    face = np.asarray(face, dtype=float)
    eye_scale = max(_dist(face[FACE["left_eye_outer"]], face[FACE["right_eye_outer"]]), 0.02)

    def eye_open(top, bottom):
        return _dist(face[top], face[bottom]) / eye_scale

    mouth_open = _dist(face[FACE["upper_lip"]], face[FACE["lower_lip"]]) / eye_scale
    mouth_width = _dist(face[FACE["mouth_left"]], face[FACE["mouth_right"]]) / eye_scale
    brow_height = ((_dist(face[FACE["left_brow"]], face[FACE["left_eye_outer"]]) +
                    _dist(face[FACE["right_brow"]], face[FACE["right_eye_outer"]])) / 2) / eye_scale
    # Smile: mouth corners rise above lip center
    lip_mid_y = (face[FACE["upper_lip"]][1] + face[FACE["lower_lip"]][1]) / 2
    corner_lift = (lip_mid_y - (face[FACE["mouth_left"]][1] + face[FACE["mouth_right"]][1]) / 2) / eye_scale
    head_tilt = float(np.degrees(np.arctan2(
        face[FACE["left_cheek"]][1] - face[FACE["right_cheek"]][1],
        face[FACE["right_cheek"]][0] - face[FACE["left_cheek"]][0])))
    return {
        "left_eye": eye_open(FACE["left_eye_top"], FACE["left_eye_bottom"]),
        "right_eye": eye_open(FACE["right_eye_top"], FACE["right_eye_bottom"]),
        "eyes_wide": max(eye_open(FACE["left_eye_top"], FACE["left_eye_bottom"]),
                         eye_open(FACE["right_eye_top"], FACE["right_eye_bottom"])),
        "mouth_open": mouth_open,
        "mouth_width": mouth_width,
        "brow_height": brow_height,
        "smile": corner_lift,
        "head_tilt": head_tilt,
    }


def pose_features(pose):
    """Body orientation and arm elevation from 33-point pose landmarks."""
    if not _finite(pose) or len(pose) < 25:
        return None
    pose = np.asarray(pose, dtype=float)
    left_sh = pose[POSE["left_shoulder"]]
    right_sh = pose[POSE["right_shoulder"]]
    shoulder_w = max(_dist(left_sh, right_sh), 0.03)
    wrists = {"left": pose[POSE["left_wrist"]], "right": pose[POSE["right_wrist"]]}
    elbows = {"left": pose[POSE["left_elbow"]], "right": pose[POSE["right_elbow"]]}
    head_y = pose[POSE["nose"]][1]

    def rel_wrist(side):
        w = wrists[side]
        return {"above_head": (head_y - w[1]) / shoulder_w,
                "above_shoulder": (left_sh[1] - w[1]) / shoulder_w,
                "spread": abs(w[0] - (left_sh[0] + right_sh[0]) / 2) / shoulder_w}

    def rel_elbow(side):
        e = elbows[side]
        return (left_sh[1] - e[1]) / shoulder_w

    return {
        "shoulder_width": float(shoulder_w),
        "wrists": {s: rel_wrist(s) for s in ("left", "right")},
        "elbows_up": {s: rel_elbow(s) for s in ("left", "right")},
        "head_y": float(head_y),
        "torso": pose[11:25].tolist(),
    }


# --------------------------------------------------------------------------
# Sequence features (temporal)
# --------------------------------------------------------------------------

def motion_energy(series):
    """Mean per-frame movement of a tracked point set across the window."""
    if len(series) < 2:
        return 0.0
    points = [np.asarray(frame, dtype=float) for frame in series if _finite(frame)]
    if len(points) < 2:
        return 0.0
    total = 0.0
    for prev, curr in zip(points, points[1:]):
        n = min(len(prev), len(curr))
        if n == 0:
            continue
        total += float(np.mean(np.hypot(prev[:n, 0] - curr[:n, 0],
                                        prev[:n, 1] - curr[:n, 1])))
    return total / (len(points) - 1)


def wave_score(hand_series):
    """Side-to-side oscillation of the palm — a wave has >= 2 direction flips."""
    xs = []
    for frame in hand_series:
        hand = hand_features(frame) if _finite(frame) else None
        if hand is None:
            continue
        frame = np.asarray(frame, dtype=float)
        xs.append(float(frame[0][0]))
    if len(xs) < 6:
        return 0.0
    center = np.mean(xs)
    flips = 0
    direction = 0
    for x in xs:
        d = 1 if x > center else -1
        if d != direction:
            flips += 1
            direction = d
    oscillation = float(np.std(xs)) * 10
    return min(1.0, max(0.0, (flips - 3) / 6) * 0.7 + oscillation * 0.5)


def nod_score(pose_series):
    """Vertical head oscillation for YES."""
    if len(pose_series) < 6:
        return 0.0
    ys = [float(np.asarray(p, dtype=float)[POSE["nose"]][1])
          for p in pose_series if _finite(p) and len(p) > 0]
    if len(ys) < 6:
        return 0.0
    center = np.mean(ys)
    flips = 0
    direction = 0
    for y in ys:
        d = 1 if y > center else -1
        if d != direction:
            flips += 1
            direction = d
    return min(1.0, max(0.0, (flips - 3) / 5) * 0.8 + float(np.std(ys)) * 8 * 0.4)


def wag_score(hand_series):
    """Horizontal tip oscillation with a mostly-static palm — NO gesture."""
    if len(hand_series) < 6:
        return 0.0
    frames = [np.asarray(f, dtype=float) for f in hand_series if _finite(f)]
    if len(frames) < 6:
        return 0.0
    tip_xs = [float(f[8][0]) for f in frames]
    wrist_xs = [float(f[0][0]) for f in frames]
    center = np.mean(tip_xs)
    flips = 0
    direction = 0
    for x in tip_xs:
        d = 1 if x > center else -1
        if d != direction:
            flips += 1
            direction = d
    still = float(np.std(wrist_xs)) < 0.02
    return min(1.0, (max(0.0, (flips - 3) / 5) * (0.9 if still else 0.5)) +
               float(np.std(tip_xs)) * 6 * 0.3)


def arms_up_score(pose_series):
    """Both wrists above the head, sustained."""
    values = []
    for pose in pose_series:
        features = pose_features(pose) if _finite(pose) else None
        if features is None:
            continue
        above = sum(1 for s in ("left", "right")
                    if features["wrists"][s]["above_head"] > 0)
        values.append(above)
    if not values:
        return 0.0
    return min(1.0, (sum(1 for v in values if v == 2) / len(values)) *
               (0.6 + 0.4 * float(np.mean(values)) / 2))


def hand_above_head_score(pose_series):
    """Either wrist above the head — celebration/victory signal."""
    values = []
    for pose in pose_series:
        features = pose_features(pose) if _finite(pose) else None
        if features is None:
            continue
        best = max(features["wrists"][s]["above_head"] for s in ("left", "right"))
        values.append(best)
    if not values:
        return 0.0
    return min(1.0, float(np.mean(values)) * 1.2)


def hand_on_face_score(face_series, hand_series):
    """Palm overlapping the face region — facepalm signal."""
    if len(face_series) != len(hand_series):
        pass  # frames are index-aligned by the caller
    scores = []
    for face, hand in zip(face_series, hand_series):
        if not _finite(face) or not _finite(hand) or len(face) < 468 or len(hand) < 21:
            continue
        face = np.asarray(face, dtype=float)
        hand = np.asarray(hand, dtype=float)
        center = (face[FACE["nose"]] + face[FACE["chin"]]) / 2
        face_size = max(_dist(face[FACE["forehead"]], face[FACE["chin"]]), 0.05)
        palm = hand[9]
        distance = _dist(palm, center) / face_size
        scores.append(max(0.0, 1.0 - distance * 1.6))
    return float(np.mean(scores)) if scores else 0.0


def hands_on_head_score(pose_series, hand_series):
    """Both hands near the top of the head — shock/disbelief signal."""
    if len(pose_series) != len(hand_series):
        pass
    scores = []
    for pose, hand in zip(pose_series, hand_series):
        if not _finite(pose) or not _finite(hand) or len(pose) < 25:
            continue
        pose = np.asarray(pose, dtype=float)
        hand = np.asarray(hand, dtype=float)
        head = pose[POSE["nose"]]
        shoulder_w = max(_dist(pose[POSE["left_shoulder"]], pose[POSE["right_shoulder"]]), 0.03)
        for point in (hand[8], hand[12]):
            if _dist(point, head) / shoulder_w < 1.1:
                scores.append(1.0)
                break
        else:
            scores.append(0.0)
    return float(np.mean(scores)) if scores else 0.0


def jump_score(pose_series):
    """Vertical torso translation — celebration jump."""
    if len(pose_series) < 8:
        return 0.0
    ys = []
    for pose in pose_series:
        features = pose_features(pose) if _finite(pose) else None
        if features is None:
            continue
        ys.append(features["head_y"])
    if len(ys) < 8:
        return 0.0
    return min(1.0, float(np.ptp(ys)) * 6)


def shrug_score(pose_series):
    """Shoulders rising toward ears with forearms out."""
    values = []
    for pose in pose_series:
        features = pose_features(pose) if _finite(pose) else None
        if features is None:
            continue
        elbows = (features["elbows_up"]["left"] + features["elbows_up"]["right"]) / 2
        spread = (features["wrists"]["left"]["spread"] +
                  features["wrists"]["right"]["spread"]) / 2
        values.append(min(1.0, elbows * 0.8 + min(1.0, spread) * 0.3))
    return float(np.mean(values)) if values else 0.0


# --------------------------------------------------------------------------
# Classifiers
# --------------------------------------------------------------------------

def classify_hand_gesture(hand_series, face_series=None, pose_series=None):
    """Classify a communicator gesture from a hand landmark window.

    Returns dict with label, confidence, and runner-up possibilities.
    """
    if not hand_series:
        return None
    latest = hand_series[-1]
    current = hand_features(latest)
    if current is None:
        return None

    # Temporal scores
    wave = wave_score(hand_series)
    wag = wag_score(hand_series)
    nod = nod_score(pose_series or [])

    candidates = []
    open_count = current["count"]
    if wave > 0.45 and open_count >= 4:
        candidates.append(("HELLO", 0.55 + wave * 0.4))
    if wag > 0.4 and open_count <= 2:
        candidates.append(("NO", 0.5 + wag * 0.45))
    if nod > 0.4:
        candidates.append(("YES", 0.5 + nod * 0.45))
    if current["extended"] == [True, True, False, False, False] and current["count"] == 2:
        # index+thumb pinch-ish → WATER-ish hint (vocabulary hint system below)
        candidates.append(("WATER", 0.5))
    if all(current["extended"]) and current["count"] == 5:
        candidates.append(("PLEASE", 0.55))
        candidates.append(("HOME", 0.5))
    if current["extended"] == [True, False, False, False, True] :
        candidates.append(("I LOVE YOU", 0.6))
    if current["extended"] == [False, True, False, False, False]:
        candidates.append(("POINT", 0.6))
    candidates.sort(key=lambda item: -item[1])
    if not candidates:
        return None
    label, confidence = candidates[0]
    return {"label": label, "confidence": round(min(0.97, confidence), 2),
            "alternatives": [{"label": l, "confidence": round(c, 2)}
                             for l, c in candidates[1:3]]}


def classify_action(hand_series, pose_series, face_series=None):
    """Classify a charades action from pose + hand motion."""
    if not pose_series:
        return None
    scores = {
        "CELEBRATING": arms_up_score(pose_series) * 0.7 + jump_score(pose_series) * 0.3,
        "FIGHTING": float(np.mean([motion_energy([h]) for h in hand_series[-6:]])) * 12
                    if hand_series else 0.0,
        "CRYING": (hand_on_face_score(face_series or [], hand_series) * 0.6 +
                   _sad_face(face_series or []) * 0.4),
        "DANCING": motion_energy(pose_series) * 6,
        "EXERCISING": motion_energy(pose_series) * 5 * 0.8,
        "LIFTING": arms_up_score(pose_series) * 0.6,
        "RIDING": float(np.mean([0.4])) if pose_series else 0.0,
        "HELPING": shrug_score(pose_series) * 0.5,
        "WANDERING": motion_energy(pose_series) * 3,
    }
    ranked = sorted(scores.items(), key=lambda item: -item[1])[:3]
    label, confidence = ranked[0]
    if confidence < 0.35:
        return None
    return {"label": label, "confidence": round(min(0.95, confidence), 2),
            "alternatives": [{"label": l, "confidence": round(c, 2)}
                             for l, c in ranked[1:]]}


def _sad_face(face_series):
    """Inverted smile signal — frown-like expression."""
    scores = [face_features(f) for f in face_series if _finite(f)]
    scores = [f for f in scores if f]
    if not scores:
        return 0.0
    return float(np.mean([max(0.0, -f["smile"]) * 8 for f in scores]))


def classify_expression(face_series, hand_series=None, pose_series=None):
    """Classify the observable expression from face (+hands/pose) features."""
    frames = [face_features(f) for f in face_series if _finite(f)]
    frames = [f for f in frames if f]
    if len(frames) < 3:
        return None
    avg = {key: float(np.mean([f[key] for f in frames]))
           for key in frames[0]}
    latest = frames[-1]
    hands = [hand_features(h) for h in (hand_series or []) if _finite(h)]
    hands = [h for h in hands if h]
    hands_up = (hands_on_head_score(pose_series or [], hand_series or [])
                if hand_series and pose_series else 0.0)
    motion = motion_energy(face_series) * 10

    scores = {
        "surprise": (min(1.0, avg["eyes_wide"] * 6) * 0.4 +
                     min(1.0, avg["mouth_open"] * 5) * 0.4 + hands_up * 0.2),
        "excitement": (min(1.0, avg["smile"] * 6) * 0.4 + motion * 0.2 +
                       min(1.0, avg["mouth_open"] * 4) * 0.2 +
                       (hand_above_head_score(pose_series or []) * 0.2 if pose_series else 0)),
        "smile": min(1.0, avg["smile"] * 6),
        "sadness": _sad_face(face_series) * 0.7 + (0.3 if avg["eyes_wide"] < 0.25 else 0),
        "anger": max(0.0, (0.28 - avg["brow_height"]) * 10) * 0.6 +
                 max(0.0, (0.15 - avg["smile"]) * 4) * 0.4,
        "confusion": max(0.0, abs(latest["head_tilt"]) - 4) / 20 * 0.5 +
                     max(0.0, (0.3 - avg["mouth_open"]) * 2) * 0.2,
        "neutral": 0.35,
    }
    # Suppression: neutral loses when anything expressive fires
    expressive = max(v for k, v in scores.items() if k != "neutral")
    if expressive > 0.45:
        scores["neutral"] *= 0.4
    ranked = sorted(scores.items(), key=lambda item: -item[1])[:3]
    label, confidence = ranked[0]
    if confidence < 0.3:
        return {"label": "uncertain", "confidence": round(confidence, 2),
                "alternatives": [{"label": l, "confidence": round(c, 2)}
                                 for l, c in ranked]}
    return {"label": label, "confidence": round(min(0.96, confidence), 2),
            "alternatives": [{"label": l, "confidence": round(c, 2)}
                             for l, c in ranked[1:]]}


def expression_intensity(expression, face_series):
    """How strongly the expression is performed (0-1)."""
    frames = [face_features(f) for f in face_series if _finite(f)]
    frames = [f for f in frames if f]
    if not frames or expression in (None, "uncertain"):
        return 0.0
    avg = {key: float(np.mean([f[key] for f in frames])) for key in frames[0]}
    label = expression["label"]
    if label == "surprise":
        return float(min(1.0, (avg["eyes_wide"] * 6 + avg["mouth_open"] * 5) / 2))
    if label in ("excitement", "smile"):
        return float(min(1.0, avg["smile"] * 6))
    if label == "sadness":
        return float(min(1.0, -avg["smile"] * 8))
    if label == "anger":
        return float(min(1.0, (0.28 - avg["brow_height"]) * 10))
    if label == "confusion":
        return float(min(1.0, abs(avg["head_tilt"]) / 18))
    return float(min(1.0, expression.get("confidence", 0.5)))
