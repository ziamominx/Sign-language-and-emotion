"""Arena engine tests: perception, agents, learning, game, store, API."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from arena.charades_agent import CharadesAgent
from arena.game_engine import CharadesGame
from arena.meme_agent import MemeAgent, match_memes, tag_profile
from arena.perception import (classify_action, classify_expression,
                              face_features, hand_features)
from arena.perception import arms_up_score, motion_energy
from arena.store import ArenaStore


def hand_frame(fingers_up=5, x=0.5, y=0.5):
    """Synthetic 21-point hand; fingers_up counts extended fingers plus thumb."""
    points = np.zeros((21, 2))
    points[0] = [x, y]
    # Four fingers: tip far from wrist when extended, pip closer always
    for index, (tip, pip) in enumerate(((8, 6), (12, 10), (16, 14), (20, 18))):
        extended = index < fingers_up - 1
        points[tip] = [x, y - (0.12 if extended else 0.03)]
        points[pip] = [x, y - 0.05]
    points[4] = [x - (0.15 if fingers_up == 5 else 0.04), y]
    points[5] = [x, y - 0.04]
    points[17] = [x + 0.05, y - 0.02]
    return points.tolist()


def pose_frame(arms='down', jitter=0.0):
    points = np.zeros((33, 2))
    points[11] = [0.4, 0.4]   # left shoulder
    points[12] = [0.6, 0.4]   # right shoulder
    points[13] = [0.35, 0.55]  # left elbow
    points[14] = [0.65, 0.55]  # right elbow
    # y grows downward: 'up' means wrists ABOVE the head (y=0.2)
    points[15] = [0.35, 0.15 if arms == 'up' else 0.7]
    points[16] = [0.65, 0.15 if arms == 'up' else 0.7]
    points[23] = [0.42, 0.8]
    points[24] = [0.58, 0.8]
    points[0] = [0.5, 0.2]
    if jitter:
        points += np.random.default_rng(0).normal(0, jitter, points.shape)
    return points.tolist()


def face_frame(smile=0.0, mouth_open=0.05, eye=0.03):
    """Synthetic 478-point face. Mouth corners sit at lip-center height minus
    smile, so smile=0 is a neutral mouth and smile>0 lifts the corners."""
    face = np.zeros((478, 2))
    face[33] = [0.42, 0.45]
    face[263] = [0.58, 0.45]  # inter-eye distance = 0.16
    face[159] = [0.45, 0.42]
    face[145] = [0.45, 0.42 + eye]
    face[386] = [0.55, 0.42]
    face[374] = [0.55, 0.42 + eye]
    corner_y = 0.58 + mouth_open / 2 - smile
    face[61] = [0.45, corner_y]
    face[291] = [0.55, corner_y]
    face[13] = [0.5, 0.58]
    face[14] = [0.5, 0.58 + mouth_open]
    face[1] = [0.5, 0.52]
    face[152] = [0.5, 0.75]
    face[10] = [0.5, 0.35]
    face[105] = [0.45, 0.40]
    face[334] = [0.55, 0.40]
    face[234] = [0.35, 0.5]
    face[454] = [0.65, 0.5]
    return face.tolist()


class PerceptionTests(unittest.TestCase):
    def test_hand_features_counts_extended_fingers(self):
        features = hand_features(hand_frame(fingers_up=5))
        self.assertEqual(features["count"], 5)  # four fingers + thumb
        features = hand_features(hand_frame(fingers_up=1))
        self.assertEqual(features["count"], 1)  # thumb only

    def test_face_features_detects_smile(self):
        neutral = face_features(face_frame(smile=0))
        smiling = face_features(face_frame(smile=0.04))
        self.assertGreater(smiling["smile"], neutral["smile"])

    def test_arms_up_score_rises_with_sustained_hands_up(self):
        down = [pose_frame(arms='down') for _ in range(10)]
        up = [pose_frame(arms='up') for _ in range(10)]
        self.assertLess(arms_up_score(down), 0.2)
        self.assertGreater(arms_up_score(up), 0.5)

    def test_motion_energy_zero_for_static(self):
        static = [pose_frame() for _ in range(8)]
        self.assertAlmostEqual(motion_energy(static), 0.0, places=6)

    def test_expression_classifier_prefers_surprise_for_wide_eyes(self):
        frames = [face_frame(mouth_open=0.12, eye=0.08) for _ in range(8)]
        result = classify_expression(frames)
        self.assertIsNotNone(result)
        self.assertEqual(result["label"], "surprise")

    def test_action_classifier_sees_celebration(self):
        result = classify_action([], [pose_frame(arms='up') for _ in range(12)], [])
        self.assertIsNotNone(result)
        self.assertEqual(result["label"], "CELEBRATING")


class CharadesAgentTests(unittest.TestCase):
    def setUp(self):
        self.agent = CharadesAgent(store=None)
        self.agent.reset_round()

    def observe(self, labels):
        self.agent.observe([{"label": label, "confidence": 0.9} for label in labels])

    def test_concept_evidence_ranks_matching_movie_first(self):
        self.observe(["college", "student", "exam", "engineering"])
        candidates = self.agent.candidates()
        self.assertTrue(candidates)
        self.assertEqual(candidates[0]["title"], "3 Idiots")

    def test_should_not_guess_without_enough_evidence(self):
        self.observe(["college"])
        self.assertFalse(self.agent.should_guess())

    def test_guess_when_decisive(self):
        self.observe(["wrestling", "father", "daughters", "training"])
        self.assertTrue(self.agent.should_guess())
        guess = self.agent.guess()
        self.assertEqual(guess["title"], "Dangal")

    def test_wrong_feedback_demotes_and_alternative_can_lead(self):
        self.observe(["college", "student", "exam", "engineering", "friends"])
        first = self.agent.guess()
        self.assertEqual(first["title"], "3 Idiots")
        self.agent.feedback(False)
        self.assertIn("3 Idiots", self.agent.rejected)
        candidates = self.agent.candidates()
        # The rejected movie is heavily penalized
        self.assertNotEqual(candidates[0]["title"], "3 Idiots")

    def test_decayed_concepts_fade(self):
        self.observe(["cricket"])
        weight_now = self.agent.concepts["cricket"]
        self.agent.observe([])
        self.assertLess(self.agent.concepts["cricket"], weight_now)


class MemeAgentTests(unittest.TestCase):
    def setUp(self):
        self.agent = MemeAgent()

    def test_surprise_profile_matches_surprise_meme(self):
        expression = {"label": "surprise", "confidence": 0.9}
        features = {"eyes_wide": 0.12, "mouth_open": 0.15}
        profile = tag_profile(expression, features)
        self.assertIn("surprise", profile)
        matches = match_memes(profile, self.agent.memes)
        self.assertTrue(matches)
        self.assertIn("surprise", " ".join(matches[0]["matched_tags"]))

    def test_challenge_scoring_bounded(self):
        profile = {"surprise": 0.9, "eyes_wide": 0.8, "mouth_open": 0.7}
        challenge = {"expect_tags": ["surprise", "shock", "eyes_wide", "mouth_open"]}
        score = __import__('arena.meme_agent', fromlist=['challenge_score']).challenge_score(
            profile, challenge, {"label": "surprise", "confidence": 0.9}, None)
        self.assertTrue(0 <= score["final"] <= 100)
        self.assertGreaterEqual(score["challenge_match"], 50)

    def test_cycle_challenges(self):
        first = self.agent.current_challenge()["id"]
        self.agent.next_challenge()
        second = self.agent.current_challenge()["id"]
        self.assertNotEqual(first, second)


class GameEngineTests(unittest.TestCase):
    def setUp(self):
        self.game = CharadesGame()

    def test_correct_awards_fast_bonus(self):
        self.game.start_round("TEAM A", "Dangal")
        self.game.resolve_correct(elapsed_seconds=8)
        self.assertEqual(self.game.scores["TEAM B"], 2)
        self.assertFalse(self.game.round_active)

    def test_wrong_awards_actor_team(self):
        self.game.start_round("TEAM A", "Dangal")
        self.game.resolve_wrong()
        self.assertEqual(self.game.scores["TEAM A"], 1)

    def test_timeout_awards_actor_team(self):
        self.game.start_round("TEAM B", "Sholay")
        self.game.resolve_timeout()
        self.assertEqual(self.game.scores["TEAM B"], 1)

    def test_state_masks_secret(self):
        self.game.start_round("TEAM A", "Dangal")
        self.assertIsNone(self.game.state(masked=True)["secret_movie"])
        self.assertEqual(self.game.state(masked=False)["secret_movie"], "Dangal")

    def test_no_second_round_while_active(self):
        self.game.start_round("TEAM A", "Dangal")
        with self.assertRaises(ValueError):
            self.game.start_round("TEAM B", "Sholay")


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def make_store(self):
        store = ArenaStore(Path(self.directory.name) / "test.sqlite3")
        self.addCleanup(store.close)
        return store

    def test_weight_deltas_persist_and_reset(self):
        store = self.make_store()
        store.apply_weight_deltas({"Dangal": 0.08, "Sholay": -0.12})
        weights = store.movie_weights()
        self.assertGreater(weights["Dangal"], 1.0)
        self.assertLess(weights["Sholay"], 1.0)
        store.reset_learning()
        self.assertEqual(store.movie_weights(), {})

    def test_sessions_and_accuracy(self):
        store = self.make_store()
        store.add_session("charades", "correct", {"movie": "PK"})
        store.add_session("charades", "wrong", {"movie": "PK"})
        store.add_session("meme", "attempt", {"score": 66})
        self.assertEqual(store.accuracy(), 0.5)
        self.assertEqual(len(store.recent_sessions()), 3)


class ArenaApiTests(unittest.TestCase):
    def setUp(self):
        from server import app
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_status_lists_datasets(self):
        data = self.client.get("/api/arena/status").get_json()
        self.assertTrue(data["ok"])
        self.assertGreaterEqual(data["movies"], 10)
        self.assertGreaterEqual(data["memes"], 10)

    def test_charades_start_and_feedback_flow(self):
        response = self.client.post("/api/arena/charades/reset")
        self.assertTrue(response.get_json()["ok"])
        start = self.client.post("/api/arena/charades/start",
                                 json={"actor_team": "TEAM A"}).get_json()
        self.assertIn("secret_movie", start)
        self.assertTrue(start["ai_script"])
        feedback = self.client.post("/api/arena/charades/feedback",
                                    json={"correct": True}).get_json()
        self.assertTrue(feedback["ok"])
        self.assertEqual(feedback["game"]["teams"][0]["name"], "TEAM A")

    def test_meme_analyze_endpoint(self):
        frames = [{"hands": [], "face": [face_frame(mouth_open=0.12, eye=0.08) for _ in range(3)],
                   "pose": []} for _ in range(6)]
        response = self.client.post("/api/arena/memes/analyze", json={"frames": frames})
        data = response.get_json()
        self.assertTrue(data["ok"])
        self.assertIn("score", data)

    def test_communicator_observe_endpoint(self):
        frames = [{"hands": [hand_frame(5)], "face": [], "pose": []} for _ in range(8)]
        response = self.client.post("/api/arena/communicator/observe", json={"frames": frames})
        self.assertTrue(response.get_json()["ok"])

    def test_pages_render(self):
        for path in ("/", "/communicator", "/charades", "/memes", "/history", "/settings", "/lab"):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 200)


if __name__ == "__main__":
    unittest.main()
