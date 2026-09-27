"""Facial-expression API must distinguish a detected face from an empty frame."""

import io
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from server import app, choose_expression


def frame():
    _, encoded = cv2.imencode('.jpg', np.zeros((240, 320, 3), dtype=np.uint8))
    return {'frame': (io.BytesIO(encoded.tobytes()), 'frame.jpg', 'image/jpeg')}


class EmotionApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_status_explains_optional_dependency(self):
        with patch('server.find_spec', return_value=None):
            result = self.client.get('/api/emotion/status').get_json()
        self.assertFalse(result['ready'])
        self.assertIn('DeepFace', result['error'])

    def test_missing_dependency_does_not_guess_expression(self):
        with patch('server.find_spec', return_value=None):
            response = self.client.post('/api/emotion', data=frame(),
                                        content_type='multipart/form-data')
        self.assertEqual(response.status_code, 503)

    def test_no_face_returns_no_label_without_running_model(self):
        detector = MagicMock()
        detector.__enter__.return_value.process.return_value = SimpleNamespace(detections=[])
        with patch('server.find_spec', return_value=object()), \
                patch('server.mp.solutions.face_detection.FaceDetection', return_value=detector), \
                patch('server.classify_expression') as classify:
            response = self.client.post('/api/emotion', data=frame(),
                                        content_type='multipart/form-data')
        self.assertEqual(response.get_json(), {'visible': False, 'label': None})
        classify.assert_not_called()

    def test_detected_face_is_cropped_before_classification(self):
        box = SimpleNamespace(xmin=.2, ymin=.2, width=.5, height=.5)
        detection = SimpleNamespace(location_data=SimpleNamespace(relative_bounding_box=box))
        detector = MagicMock()
        detector.__enter__.return_value.process.return_value = SimpleNamespace(
            detections=[detection])
        with patch('server.find_spec', return_value=object()), \
                patch('server.mp.solutions.face_detection.FaceDetection', return_value=detector), \
                patch('server.classify_expression', return_value={
                    'label': 'Happy', 'score': 82.0, 'tentative': False}) as classify:
            response = self.client.post('/api/emotion', data=frame(),
                                        content_type='multipart/form-data')
        self.assertEqual(response.get_json(), {'visible': True, 'label': 'Happy',
                                              'score': 82.0, 'tentative': False})
        crop = classify.call_args.args[0]
        self.assertLess(crop.shape[0], 240)
        self.assertLess(crop.shape[1], 320)

    def test_close_happy_or_sad_score_is_shown_as_tentative_instead_of_neutral(self):
        happy = choose_expression('neutral', {'neutral': 42, 'happy': 36, 'sad': 8})
        self.assertEqual(happy, {'label': 'Happy', 'score': 36.0, 'tentative': True})
        sad = choose_expression('neutral', {'neutral': 40, 'happy': 8, 'sad': 35})
        self.assertEqual(sad['label'], 'Sad')
        neutral = choose_expression('neutral', {'neutral': 80, 'happy': 7, 'sad': 8})
        self.assertEqual(neutral['label'], 'Neutral')


if __name__ == '__main__':
    unittest.main()
