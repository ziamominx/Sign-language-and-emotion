import io
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from asl_model import ASLRecognizer, normalize_keypoints
from server import app, extract_keypoints


def landmarks(count, x):
    return SimpleNamespace(landmark=[SimpleNamespace(x=x, y=0.5) for _ in range(count)])


def camera_frames(count=12):
    _, encoded = cv2.imencode(".jpg", np.zeros((240, 320, 3), dtype=np.uint8))
    return [io.BytesIO(encoded.tobytes()) for _ in range(count)]


class ExtractionTests(unittest.TestCase):
    def test_independent_hands_recover_correct_anatomical_sides(self):
        holistic_result = SimpleNamespace(pose_landmarks=landmarks(33, 0.5),
                                          left_hand_landmarks=None, right_hand_landmarks=None)
        hands_result = SimpleNamespace(
            multi_hand_landmarks=[landmarks(21, 0.8), landmarks(21, 0.2)],
            multi_handedness=[SimpleNamespace(classification=[SimpleNamespace(label="Left")]),
                              SimpleNamespace(classification=[SimpleNamespace(label="Right")])])
        holistic = MagicMock()
        holistic.__enter__.return_value.process.return_value = holistic_result
        hands = MagicMock()
        hands.__enter__.return_value.process.return_value = hands_result
        diagnostics = {}
        with patch("server.mp.solutions.holistic.Holistic", return_value=holistic), \
                patch("server.mp.solutions.hands.Hands", return_value=hands) as create_hands:
            points = extract_keypoints(camera_frames(), diagnostics)
        self.assertEqual(points.shape, (12, 75, 2))
        np.testing.assert_allclose(points[:, 33:54, 0], 0.2)
        np.testing.assert_allclose(points[:, 54:75, 0], 0.8)
        self.assertEqual(diagnostics, {"hands": 1.0, "pose": 1.0, "fallback_frames": 12})
        create_hands.assert_called_once()

    def test_fallback_fills_only_the_missing_holistic_side(self):
        holistic_result = SimpleNamespace(pose_landmarks=landmarks(33, 0.5),
                                          left_hand_landmarks=landmarks(21, 0.3),
                                          right_hand_landmarks=None)
        hands_result = SimpleNamespace(
            multi_hand_landmarks=[landmarks(21, 0.9), landmarks(21, 0.8)],
            multi_handedness=[SimpleNamespace(classification=[SimpleNamespace(label="Right")]),
                              SimpleNamespace(classification=[SimpleNamespace(label="Left")])])
        holistic = MagicMock()
        holistic.__enter__.return_value.process.return_value = holistic_result
        hands = MagicMock()
        hands.__enter__.return_value.process.return_value = hands_result
        with patch("server.mp.solutions.holistic.Holistic", return_value=holistic), \
                patch("server.mp.solutions.hands.Hands", return_value=hands) as create_hands:
            points = extract_keypoints(camera_frames())
        np.testing.assert_allclose(points[:, 33:54, 0], 0.3)
        np.testing.assert_allclose(points[:, 54:75, 0], 0.8)
        create_hands.assert_called_once()

    def test_live_visibility_guidance_distinguishes_hands_and_pose(self):
        hands_result = SimpleNamespace(
            multi_hand_landmarks=[landmarks(21, 0.8)],
            multi_handedness=[SimpleNamespace(classification=[SimpleNamespace(label="Left")])])
        for pose, recovered, expected in (
                (landmarks(33, 0.5), False, "Show your signing hands"),
                (None, True, "Step back until your head and shoulders")):
            with self.subTest(pose=pose is not None):
                holistic_result = SimpleNamespace(pose_landmarks=pose,
                                                  left_hand_landmarks=None,
                                                  right_hand_landmarks=None)
                holistic = MagicMock()
                holistic.__enter__.return_value.process.return_value = holistic_result
                hands = MagicMock()
                hands.__enter__.return_value.process.return_value = (
                    hands_result if recovered else SimpleNamespace(
                        multi_hand_landmarks=None, multi_handedness=None))
                with patch("server.mp.solutions.holistic.Holistic", return_value=holistic), \
                        patch("server.mp.solutions.hands.Hands", return_value=hands):
                    response = app.test_client().post(
                        "/api/live",
                        data={"frames": [(frame, f"frame-{i}.jpg")
                                         for i, frame in enumerate(camera_frames())]},
                        content_type="multipart/form-data")
                self.assertEqual(response.status_code, 200)
                data = response.get_json()
                self.assertFalse(data["visible"])
                self.assertIn(expected, data["guidance"])
                self.assertEqual(data["visibility"]["hands"], float(recovered))
                self.assertEqual(data["visibility"]["pose"], float(pose is not None))


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = ASLRecognizer()

    def test_checkpoint_and_label_mapping(self):
        self.assertEqual(len(self.model.labels), 2000)
        points = np.full((24, 75, 2), 0.5, dtype=np.float32)
        choices = self.model.predict(points)
        self.assertEqual(len(choices), 3)
        self.assertTrue(all(choice["label"] for choice in choices))
        self.assertGreaterEqual(choices[0]["score"], choices[1]["score"])

    def test_keypoint_shape_validation(self):
        with self.assertRaises(ValueError):
            normalize_keypoints(np.zeros((24, 42, 2)))

    def test_website_and_blank_clip_rejection(self):
        client = app.test_client()
        self.assertEqual(client.get("/").status_code, 200)
        self.assertEqual(client.get("/api/status").get_json()["labels"], 2000)
        _, encoded = cv2.imencode(".jpg", np.zeros((240, 320, 3), dtype=np.uint8))
        data = {"frames": [(io.BytesIO(encoded.tobytes()), f"frame-{i}.jpg") for i in range(12)]}
        response = client.post("/api/recognize", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 422)
        self.assertIn("visible", response.get_json()["error"])

        live_data = {"frames": [(io.BytesIO(encoded.tobytes()), f"frame-{i}.jpg") for i in range(12)]}
        live_response = client.post("/api/live", data=live_data, content_type="multipart/form-data")
        self.assertEqual(live_response.status_code, 200)
        self.assertFalse(live_response.get_json()["visible"])
        self.assertEqual(live_response.get_json()["suggestions"], [])

    def test_live_recognition_returns_model_predictions(self):
        client = app.test_client()
        choices = [{"label": "hello", "score": 0.81}, {"label": "help", "score": 0.09}]
        def extracted(_, diagnostics=None):
            diagnostics.update({"hands": 1.0, "pose": 1.0, "fallback_frames": 0})
            return np.zeros((16, 75, 2))

        with patch("server.extract_keypoints", side_effect=extracted) as extract:
            with patch("server.get_recognizer") as get_model:
                get_model.return_value.predict.return_value = choices
                data = {"frames": [(io.BytesIO(b"frame"), f"frame-{i}.jpg") for i in range(16)]}
                response = client.post("/api/live", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["visible"])
        self.assertFalse(response.get_json()["uncertain"])
        self.assertEqual(response.get_json()["suggestions"], choices)
        self.assertEqual(response.get_json()["visibility"]["hands"], 1.0)
        extract.assert_called_once()


if __name__ == "__main__":
    unittest.main()
