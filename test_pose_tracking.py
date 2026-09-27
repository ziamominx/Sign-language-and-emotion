"""The lightweight body stream used with hand landmarks for word inference."""

import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from server import app


class PoseTrackingTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_returns_33_body_points_or_none(self):
        _, encoded = cv2.imencode('.jpg', np.zeros((240, 320, 3), dtype=np.uint8))
        for landmarks, expected in (([SimpleNamespace(x=.4, y=.6)] * 33, 33),
                                    (None, None)):
            with self.subTest(present=landmarks is not None):
                tracker = SimpleNamespace(process=lambda _: SimpleNamespace(
                    pose_landmarks=SimpleNamespace(landmark=landmarks) if landmarks else None))
                with patch('server.get_pose_tracker', return_value=tracker):
                    response = self.client.post('/api/pose-track', data={
                        'frame': (io.BytesIO(encoded.tobytes()), 'pose.jpg')},
                        content_type='multipart/form-data')
                self.assertEqual(response.status_code, 200)
                points = response.get_json()['pose']
                self.assertEqual(len(points) if points else None, expected)


if __name__ == '__main__':
    unittest.main()
