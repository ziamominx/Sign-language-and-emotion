"""API contract tests for the optional, experimental fingerspelling mode."""

import io
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from server import app


def frame_data():
    _, encoded = cv2.imencode(".jpg", np.zeros((240, 320, 3), dtype=np.uint8))
    return {"frame": (io.BytesIO(encoded.tobytes()), "frame.jpg", "image/jpeg")}


class AlphabetApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_status_reports_missing_optional_model_without_server_error(self):
        with patch("server.get_alphabet_recognizer", side_effect=FileNotFoundError("local checkpoint missing")):
            response = self.client.get("/api/alphabet/status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {
            "ready": False, "experimental": True, "error": "local checkpoint missing"})

    def test_status_reports_ready_alphabet_model(self):
        model = SimpleNamespace(labels=tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ("del", "space"))
        with patch("server.get_alphabet_recognizer", return_value=model):
            response = self.client.get("/api/alphabet/status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {
            "ready": True, "labels": 28, "experimental": True})

    def test_predict_validates_frame_and_optional_checkpoint(self):
        response = self.client.post("/api/alphabet/predict", data={},
                                    content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        invalid = self.client.post("/api/alphabet/predict",
                                   data={"frame": (io.BytesIO(b"not an image"), "frame.jpg")},
                                   content_type="multipart/form-data")
        self.assertEqual(invalid.status_code, 422)
        with patch("server.get_alphabet_recognizer", side_effect=FileNotFoundError("checkpoint missing")):
            absent = self.client.post("/api/alphabet/predict", data=frame_data(),
                                      content_type="multipart/form-data")
        self.assertEqual(absent.status_code, 503)
        self.assertIn("checkpoint missing", absent.get_json()["error"])

    def test_no_hand_returns_visible_false_without_predicting(self):
        model = MagicMock()
        hands = MagicMock()
        hands.__enter__.return_value.process.return_value = SimpleNamespace(
            multi_hand_landmarks=None)
        with patch("server.get_alphabet_recognizer", return_value=model), \
                patch("server.mp.solutions.hands.Hands", return_value=hands):
            response = self.client.post("/api/alphabet/predict", data=frame_data(),
                                        content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertFalse(data["visible"])
        self.assertTrue(data["uncertain"])
        self.assertEqual(data["suggestions"], [])
        self.assertIn("Show one hand", data["guidance"])
        self.assertGreaterEqual(data["processing_ms"], 0)
        model.predict.assert_not_called()

    def test_visible_hand_returns_ranked_suggestions_without_committing(self):
        points = [SimpleNamespace(x=0.25, y=0.5, z=0.1) for _ in range(21)]
        choices = [{"label": "A", "score": 0.8},
                   {"label": "B", "score": 0.1},
                   {"label": "C", "score": 0.05}]
        model = MagicMock()
        model.predict.return_value = choices
        hands = MagicMock()
        hands.__enter__.return_value.process.return_value = SimpleNamespace(
            multi_hand_landmarks=[SimpleNamespace(landmark=points)])
        with patch("server.get_alphabet_recognizer", return_value=model), \
                patch("server.mp.solutions.hands.Hands", return_value=hands) as create_hands:
            response = self.client.post("/api/alphabet/predict", data=frame_data(),
                                        content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["visible"])
        self.assertFalse(data["uncertain"])
        self.assertEqual(data["suggestions"], choices)
        self.assertNotIn("transcript", data)
        self.assertIs(model.predict.call_args.args[0], points)
        create_hands.assert_called_once()
        self.assertTrue(create_hands.call_args.kwargs["static_image_mode"])


if __name__ == "__main__":
    unittest.main()
