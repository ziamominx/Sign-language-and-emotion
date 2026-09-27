"""Live hand overlay and two-open-palm finish gesture contracts."""

import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from server import _open_palm, app


def hand(open_fingers=True):
    points = [SimpleNamespace(x=0.5, y=0.7) for _ in range(21)]
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        points[pip] = SimpleNamespace(x=0.5, y=0.5)
        points[tip] = SimpleNamespace(x=0.5, y=0.3 if open_fingers else 0.6)
    return SimpleNamespace(landmark=points)


def frame():
    _, encoded = cv2.imencode('.jpg', np.zeros((240, 320, 3), dtype=np.uint8))
    return {'frame': (io.BytesIO(encoded.tobytes()), 'frame.jpg', 'image/jpeg')}


class HandTrackingTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_open_palm_requires_four_extended_fingers(self):
        self.assertTrue(_open_palm(hand().landmark))
        self.assertFalse(_open_palm(hand(False).landmark))

    def test_track_returns_overlay_points_and_two_hand_finish(self):
        tracker = SimpleNamespace(process=lambda _: SimpleNamespace(
            multi_hand_landmarks=[hand(), hand()]))
        with patch('server.get_tracker', return_value=tracker):
            response = self.client.post('/api/track', data=frame(),
                                        content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(len(data['hands']), 2)
        self.assertEqual(len(data['hands'][0]), 21)
        self.assertTrue(data['finish_gesture'])

    def test_one_hand_or_no_hand_cannot_finish(self):
        for landmarks in ([hand()], []):
            tracker = SimpleNamespace(process=lambda _, found=landmarks: SimpleNamespace(
                multi_hand_landmarks=found))
            with patch('server.get_tracker', return_value=tracker):
                response = self.client.post('/api/track', data=frame(),
                                            content_type='multipart/form-data')
            self.assertFalse(response.get_json()['finish_gesture'])

    def test_bad_frames_are_rejected(self):
        self.assertEqual(self.client.post('/api/track', data={},
                         content_type='multipart/form-data').status_code, 400)
        self.assertEqual(self.client.post('/api/track',
                         data={'frame': (io.BytesIO(b'bad'), 'frame.jpg')},
                         content_type='multipart/form-data').status_code, 422)


if __name__ == '__main__':
    unittest.main()
