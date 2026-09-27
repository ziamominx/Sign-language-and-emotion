"""Fast landmark stream validation and inference contract."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from server import app


def clip(pose=True, hands=True):
    frame = [[0.5, 0.5] if pose else [0, 0] for _ in range(33)]
    frame += [[0.5, 0.5] if hands else [0, 0] for _ in range(42)]
    return {"frames": [frame for _ in range(18)]}


class LiveLandmarkTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_valid_clip_uses_pretrained_model(self):
        model = SimpleNamespace(predict=lambda points: [
            {"label": "my", "score": 0.8}, {"label": "name", "score": 0.1}])
        with patch('server.get_recognizer', return_value=model) as get_model:
            response = self.client.post('/api/live-landmarks', json=clip())
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()['visible'])
        self.assertEqual(response.get_json()['suggestions'][0]['label'], 'my')
        self.assertEqual(get_model.call_count, 1)

    def test_missing_hands_or_pose_does_not_guess(self):
        for pose, hands in ((False, True), (True, False)):
            with self.subTest(pose=pose, hands=hands):
                with patch('server.get_recognizer') as get_model:
                    response = self.client.post('/api/live-landmarks',
                                                json=clip(pose, hands))
                self.assertEqual(response.status_code, 200)
                self.assertFalse(response.get_json()['visible'])
                get_model.assert_not_called()

    def test_malformed_clip_is_rejected(self):
        self.assertEqual(self.client.post('/api/live-landmarks', json={}).status_code, 400)
        self.assertEqual(self.client.post('/api/live-landmarks',
                         json={"frames": [[[0, 0]] for _ in range(18)]}).status_code, 422)


if __name__ == '__main__':
    unittest.main()
