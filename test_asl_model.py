import io
import unittest

import cv2
import numpy as np

from asl_model import ASLRecognizer, normalize_keypoints
from server import app


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


if __name__ == "__main__":
    unittest.main()
