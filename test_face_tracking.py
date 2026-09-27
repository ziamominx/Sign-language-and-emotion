"""Face landmarks used by the mirrored live preview."""

import io
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from server import app


def frame():
    _, encoded = cv2.imencode('.jpg', np.zeros((270, 480, 3), dtype=np.uint8))
    return {'frame': (io.BytesIO(encoded.tobytes()), 'frame.jpg', 'image/jpeg')}


class FaceTrackingTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_no_face_returns_no_markers(self):
        tracker = SimpleNamespace(process=lambda _: SimpleNamespace(multi_face_landmarks=[]))
        with patch('server.get_face_tracker', return_value=tracker):
            response = self.client.post('/api/face-track', data=frame(),
                                        content_type='multipart/form-data')
        self.assertEqual(response.get_json(), {'face': None, 'box': None})

    def test_detected_face_returns_points_and_bounds(self):
        points = [SimpleNamespace(x=.2 + i / 1000, y=.3 + i / 2000)
                  for i in range(468)]
        tracker = SimpleNamespace(process=lambda _: SimpleNamespace(
            multi_face_landmarks=[SimpleNamespace(landmark=points)]))
        with patch('server.get_face_tracker', return_value=tracker):
            response = self.client.post('/api/face-track', data=frame(),
                                        content_type='multipart/form-data')
        data = response.get_json()
        self.assertEqual(len(data['face']), 468)
        self.assertEqual(data['face'][0], [.2, .3])
        self.assertEqual(data['box'], [.2, .3, .667, .5335])

    def test_invalid_input_is_rejected(self):
        self.assertEqual(self.client.post('/api/face-track', data={},
                         content_type='multipart/form-data').status_code, 400)

    def test_face_tracker_uses_mirrored_preview_coordinates(self):
        image = np.zeros((270, 480, 3), dtype=np.uint8)
        image[:, :240] = 255
        _, encoded = cv2.imencode('.jpg', image)
        tracker = MagicMock()
        tracker.process.return_value = SimpleNamespace(multi_face_landmarks=[])
        with patch('server.get_face_tracker', return_value=tracker):
            self.client.post('/api/face-track',
                             data={'frame': (io.BytesIO(encoded.tobytes()), 'frame.jpg')},
                             content_type='multipart/form-data')
        mirrored = tracker.process.call_args.args[0]
        self.assertLess(mirrored[20, 20].mean(), 20)
        self.assertGreater(mirrored[20, 460].mean(), 235)


if __name__ == '__main__':
    unittest.main()
