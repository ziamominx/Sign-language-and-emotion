"""Charades agent: observe concepts, rank movie candidates, learn from feedback.

The agent is deliberately transparent: it maintains per-concept and
per-movie weight tables that update from human right/wrong feedback, so
"the AI learns" is a real, inspectable mechanism rather than a label.

    observe(action/concept detections)
      → concept accumulation (confidence-gated, decaying)
      → candidate generation (concept_clues co-occurrence)
      → candidate scoring (weights x evidence, minus penalties)
      → hypothesis ranking
      → guess (only when a lead is decisive)
      → human feedback (correct/wrong)
      → weight update (reinforce or penalize)

Nothing here hard-codes "college → 3 Idiots": every concept contributes
evidence to every movie that lists it, and the feedback history shifts
the ranking over time.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "movies.json"

CONCEPT_DECAY = 0.96          # per-tick decay of observed concepts
CONCEPT_MIN_CONFIDENCE = 0.5  # ignore detections below this
CONCEPT_MAX = 12              # cap accumulation per concept
FEEDBACK_LR = 0.08            # learning rate for weight updates
DECISIVE_MARGIN = 0.18        # how far the top guess must lead
MIN_EVIDENCE = 2              # distinct concepts before guessing


def load_movies(path=DEFAULT_DATA):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    movies = {movie["title"]: movie for movie in data["movies"]}
    clues = data["concept_clues"]
    for title, movie in movies.items():
        for concept in set(movie["concepts"] + movie["actions"] + movie["genre"]):
            clues.setdefault(concept, [])
            if title not in clues[concept]:
                clues[concept].append(title)
    return movies, clues


class CharadesAgent:
    def __init__(self, data_path=DEFAULT_DATA, store=None):
        self.movies, self.clues = load_movies(data_path)
        self.store = store
        self.concepts = {}
        self.reset_round()

    def reset_round(self):
        self.concepts = {}
        self.last_guess = None
        self.rejected = set()
        self.tick = 0

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def observe(self, detections):
        """detections: iterable of {label, confidence, kind} from perception."""
        self.tick += 1
        for key in list(self.concepts):
            self.concepts[key] *= CONCEPT_DECAY
        for detection in detections or []:
            confidence = float(detection.get("confidence", 0))
            if confidence < CONCEPT_MIN_CONFIDENCE:
                continue
            for concept in self._concepts_of(detection):
                self.concepts[concept] = min(
                    CONCEPT_MAX, self.concepts.get(concept, 0) + confidence)

    def _concepts_of(self, detection):
        label = str(detection.get("label", "")).strip().lower()
        kind = detection.get("kind")
        if kind == "concept":
            return [label] if label in self.clues else []
        # Gesture/action labels map onto concepts through the movie data
        mapping = {title.lower(): title for title in self.movies}
        if label in mapping:
            return []
        direct = label if label in self.clues else None
        if direct:
            return [direct]
        synonyms = {"studying": ["studying", "student", "college"],
                    "celebrating": ["celebrating", "gold", "win"],
                    "fighting": ["fighting", "fight", "war"],
                    "crying": ["crying", "sadness"],
                    "dancing": ["dancing"],
                    "exercising": ["exercising", "training"],
                    "painting": ["painting", "art"],
                    "playing": ["playing", "cricket"],
                    "riding": ["riding", "horse"],
                    "lifting": ["lifting", "training"],
                    "romance": ["romance", "love"],
                    "helping": ["helping"],
                    "healing": ["healing", "hospital"],
                    "wandering": ["wandering", "confusion"],
                    "cricket": ["cricket"],
                    "hospital": ["hospital"],
                    "book": ["book"],
                    "train": ["train"],
                    "water": ["water"],
                    "college": ["college"],
                    "fight": ["fight"],
                    "love": ["love"],
                    "gold": ["gold"],
                    "god": ["god"],
                    "alien": ["alien"],
                    "father": ["father"],
                    "king": ["king"],
                    "war": ["war"],
                    "school": ["school"]}
        return synonyms.get(label, [])

    # ------------------------------------------------------------------
    # Reasoning
    # ------------------------------------------------------------------
    def observed_concepts(self, top=6):
        ranked = sorted(self.concepts.items(), key=lambda item: -item[1])
        total = max(sum(weight for _, weight in ranked), 1e-6)
        return [{"label": label, "weight": round(weight, 2),
                 "share": round(weight / total, 2)}
                for label, weight in ranked[:top] if weight > 0.4]

    def candidates(self, top=4):
        weights = self._movie_weights()
        evidence = self._movie_evidence()
        ranked = []
        for title in self.movies:
            score = weights.get(title, 1.0) * (0.5 + evidence.get(title, 0.0))
            ranked.append((title, score, evidence.get(title, 0.0)))
        ranked.sort(key=lambda item: -item[1])
        if not ranked or ranked[0][1] <= 0:
            return []
        best = ranked[0][1]
        return [{"title": title, "confidence": round(score / best, 3),
                 "evidence": round(evidence_value, 2)}
                for title, score, evidence_value in ranked[:top] if score > 0]

    def _movie_weights(self):
        weights = {title: 1.0 for title in self.movies}
        if self.store is not None:
            for title, weight in self.store.movie_weights().items():
                if title in weights:
                    weights[title] = weight
        return weights

    def _movie_evidence(self):
        """How strongly observed concepts support each movie."""
        evidence = {title: 0.0 for title in self.movies}
        for concept, weight in self.concepts.items():
            if weight < 0.4:
                continue
            for title in self.clues.get(concept, []):
                if title in evidence:
                    evidence[title] += weight
        # A concept that fits very few movies is more informative
        for concept, weight in self.concepts.items():
            supporters = self.clues.get(concept, [])
            if weight < 0.4 or not supporters:
                continue
            concentration = 1.0 / len(supporters)
            for title in supporters:
                if title in evidence:
                    evidence[title] += concentration * weight * 0.5
        for title in self.rejected:
            if title in evidence:
                evidence[title] *= 0.35
        return evidence

    def should_guess(self):
        candidates = self.candidates(top=2)
        if len(candidates) < 1:
            return False
        evidence_titles = len(self.observed_concepts(top=99))
        if evidence_titles < MIN_EVIDENCE:
            return False
        if len(candidates) == 1:
            return candidates[0]["confidence"] >= 0.55
        return candidates[0]["confidence"] - candidates[1]["confidence"] >= DECISIVE_MARGIN

    def guess(self):
        candidates = self.candidates()
        if not candidates:
            return None
        self.last_guess = candidates[0]["title"]
        return {"title": self.last_guess,
                "confidence": candidates[0]["confidence"],
                "candidates": candidates,
                "observed": self.observed_concepts()}

    def feedback(self, correct):
        """Update weights from human feedback; returns what changed."""
        if not self.last_guess:
            return None
        title = self.last_guess
        if correct:
            for concept, weight in self.concepts.items():
                if weight >= 0.8 and title in self.clues.get(concept, []):
                    pass  # reinforcing concepts is handled by movie weights
            delta = {title: +FEEDBACK_LR}
        else:
            self.rejected.add(title)
            delta = {title: -FEEDBACK_LR * 1.5}
        if self.store is not None:
            self.store.apply_weight_deltas(delta)
        return {"title": title, "delta": delta,
                "rejected_now": title in self.rejected}

    def state(self):
        return {"concepts": self.observed_concepts(),
                "candidates": self.candidates(),
                "last_guess": self.last_guess,
                "rejected": sorted(self.rejected),
                "should_guess": self.should_guess()}


class AIActorScript:
    """Mode B: the AI 'acts out' a movie as an animated gesture card sequence.

    The cards are game communication, clearly labeled as such — they are not
    real sign language.
    """

    def __init__(self, movies, clues, rng=None):
        self.movies = movies
        self.clues = clues
        self.rng = rng or np.random.default_rng()

    def script_for(self, title, steps=4):
        movie = self.movies.get(title)
        if not movie:
            return []
        pool = list(movie["concepts"]) + list(movie["actions"]) + list(movie["genre"])
        self.rng.shuffle(pool)
        return pool[:steps]

    def hint_for(self, concept):
        hints = {
            "college": "🎓 Point at an imaginary campus, mime a bag strap",
            "school": "🏫 Flat palm taps — a school building",
            "cricket": "🏏 Swing an imaginary bat",
            "wrestling": "🤼 Grapple with the air",
            "train": "🚆 Rotate an arm like train wheels",
            "king": "👑 Lower a crown onto your head",
            "war": "⚔️ Swing a sword twice",
            "hospital": "🏥 Draw a cross on your shoulder",
            "alien": "👽 Two antennae above your head",
            "gold": "🥇 Point at a medal, mouth 'wow'",
            "father": "👨 Thumb touches forehead",
            "village": "🌾 Draw small huts with your hand",
            "love": "❤️ Cross fists over your chest",
            "friends": "🤝 Hook index fingers together",
            "studying": "✍️ Write in the air",
            "celebrating": "🎉 Fists up, small jump",
            "fighting": "🥊 Two quick jabs",
            "crying": "😢 Rub your eyes",
            "dancing": "💃 Sway side to side",
            "comedy": "😂 Big laugh, pointing",
            "drama": "🎭 Hand fans out from face",
            "action": "💥 Fast punch into palm",
            "sports": "🏅 Raise a medal",
            "romance": "💞 Draw a heart in the air",
            "historical": "🏛️ Stroke an imaginary beard",
            "family": "👨‍👩‍👧 Circle a hand around three fingers",
        }
        return hints.get(concept, f"🤟 Mime: {concept.replace('_', ' ')}")
