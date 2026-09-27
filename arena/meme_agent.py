"""Meme agent: match an observed expression/action profile to a meme.

The agent turns perception features into a tag profile with intensities
("eyes_wide 0.9", "hands_raised 0.8"), then scores every meme in the
dataset against that profile. The best match and the challenge's
expected tags drive the game score.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "memes.json"

TAG_ALIASES = {"anger": "angry", "fear_like": "fear", "confused": "confusion",
               "hands_on_head": "hands_raised", "eyes_sideways": "confusion"}


def load_memes(path=DEFAULT_DATA):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["memes"], data["challenges"]


def tag_profile(expression, features):
    """Fuse perception outputs into {tag: intensity}."""
    profile = {}
    if not expression or expression.get("label") in (None, "uncertain"):
        return profile
    label = expression["label"]
    confidence = float(expression.get("confidence", 0))
    tags = {label: max(confidence, 0.4)}
    for alias, canonical in TAG_ALIASES.items():
        if alias in tags:
            tags[canonical] = max(tags.pop(alias), tags.get(canonical, 0))
    if label == "surprise":
        tags["eyes_wide"] = min(1.0, features.get("eyes_wide", 0) * 6)
        tags["mouth_open"] = min(1.0, features.get("mouth_open", 0) * 5)
    if label in ("smile", "excitement"):
        tags["smile"] = min(1.0, features.get("smile", 0) * 6)
    if label == "sadness":
        tags["frown"] = min(1.0, max(0.0, -features.get("smile", 0)) * 8)
    if label == "anger":
        tags["eyebrows_down"] = min(1.0, max(0.0, (0.28 - features.get("brow_height", 0.3)) * 10))
    if label == "confusion":
        tags["head_tilt"] = min(1.0, abs(features.get("head_tilt", 0)) / 18)
        tags["eyebrow_raised"] = tags["head_tilt"] * 0.8
    return {tag: round(float(value), 2) for tag, value in tags.items() if value > 0.15}


def with_pose_tags(profile, action):
    """Add pose-derived tags (hands_raised, facepalm, jump)."""
    profile = dict(profile)
    if action:
        label = action.get("label", "").lower()
        confidence = float(action.get("confidence", 0))
        if label == "celebrating":
            profile["hands_raised"] = max(profile.get("hands_raised", 0), confidence)
        if label == "crying":
            profile["facepalm"] = max(profile.get("facepalm", 0), confidence * 0.8)
            profile["hand_on_face"] = confidence * 0.8
    return profile


def match_memes(profile, memes, top=3):
    """Score every meme: fraction of its tags present, weighted by intensity."""
    results = []
    for meme in memes:
        if not meme["tags"]:
            continue
        hits, total = 0.0, 0.0
        for tag in meme["tags"]:
            total += 1.0
            value = profile.get(tag, 0.0)
            if value > 0.2:
                hits += min(1.0, value)
        coverage = hits / total if total else 0.0
        # Profile intensity: how strongly the user performed what matched
        performed = sum(profile.get(tag, 0.0) for tag in meme["tags"]) / len(meme["tags"])
        score = 0.65 * coverage + 0.35 * min(1.0, performed)
        results.append({"id": meme["id"], "label": meme["label"],
                        "image": meme["image"], "score": round(score, 3),
                        "intensity": meme["intensity"],
                        "matched_tags": [tag for tag in meme["tags"] if profile.get(tag, 0) > 0.2]})
    results.sort(key=lambda item: -item["score"])
    return results[:top]


def challenge_score(profile, challenge, expression, action):
    """Score a challenge attempt across expression, pose, intensity, match."""
    expected = [TAG_ALIASES.get(tag, tag) for tag in challenge["expect_tags"]]
    hits = sum(1 for tag in expected if profile.get(tag, 0) > 0.3)
    expression_score = float(expression.get("confidence", 0)) * 100 if expression else 0.0
    pose_score = (float(action.get("confidence", 0)) * 100 if action
                  else min(100.0, max(profile.values()) * 100 if profile else 0.0))
    intensity = float(np_mean(profile)) * 100 if profile else 0.0
    match_score = hits / len(expected) * 100 if expected else 0.0
    final = round(0.35 * match_score + 0.25 * expression_score +
                  0.2 * pose_score + 0.2 * intensity)
    return {"expression": round(expression_score), "pose": round(pose_score),
            "intensity": round(intensity), "challenge_match": round(match_score),
            "final": max(0, min(100, final))}


def np_mean(profile):
    return float(sum(profile.values()) / len(profile)) if profile else 0.0


class MemeAgent:
    def __init__(self, data_path=DEFAULT_DATA):
        self.memes, self.challenges = load_memes(data_path)
        self.challenge_index = 0

    def current_challenge(self):
        return self.challenges[self.challenge_index % len(self.challenges)]

    def next_challenge(self):
        self.challenge_index = (self.challenge_index + 1) % len(self.challenges)
        return self.current_challenge()

    def analyze(self, expression, features, action):
        profile = with_pose_tags(tag_profile(expression, features or {}), action)
        matches = match_memes(profile, self.memes)
        challenge = self.current_challenge()
        return {"profile": profile, "matches": matches,
                "challenge": challenge,
                "score": challenge_score(profile, challenge, expression, action),
                "expression": expression, "action": action}
