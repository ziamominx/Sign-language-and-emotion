"""The training API must report state and gate live recognition on quality."""

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from personal_signs import PersonalSigns
from server import app, invalidate_trained_model


def clip(wrist_x=.4, finger_spread=.012, offset=0, droop=0.0):
    frames = np.zeros((18, 75, 2), dtype=np.float32)
    for t in range(18):
        frames[t, 0] = [.5, .2 + droop]
        frames[t, 11] = [.4, .4 + droop]
        frames[t, 12] = [.6, .4 + droop]
        frames[t, 13:17] = [[.38, .5 + droop], [.62, .5 + droop],
                            [.38, .6 + droop], [.62, .6 + droop]]
        x = wrist_x + offset + t * .001
        for point in range(21):
            frames[t, 33 + point] = [x + (point % 5) * finger_spread,
                                     .55 + droop - (point // 5) * .015]
    return frames.tolist()


class TrainingApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        library = PersonalSigns(root / 'signs.json')
        for offset in (0, .002, -.002):
            library.add('my', clip(offset=offset))
        for offset in (0, .002, -.002):
            library.add('name', clip(wrist_x=.42, offset=offset, droop=.08))
        patcher = patch('server.get_personal_signs', return_value=library)
        patcher.start()
        self.addCleanup(patcher.stop)
        invalidate_trained_model()
        self.addCleanup(invalidate_trained_model)

    def test_status_reports_untrained_until_a_model_exists(self):
        data = self.client.get('/api/personal-signs').get_json()
        self.assertFalse(data['trained'])
        self.assertIsNone(data['trained_accuracy'])

    def wait_for_training(self):
        self.client.post('/api/personal-signs/train')
        state = {'running': True, 'result': None}
        for _ in range(300):
            state = self.client.get('/api/personal-signs/train/status').get_json()
            if not state['running']:
                break
            time.sleep(0.05)
        return state

    def test_training_endpoint_runs_and_then_reports_the_model(self):
        state = self.wait_for_training()
        self.assertFalse(state['running'])
        self.assertTrue(state['result']['ready'])
        data = self.client.get('/api/personal-signs').get_json()
        self.assertTrue(data['trained'])

    def test_trained_model_beats_the_pretrained_model_in_live_recognition(self):
        self.wait_for_training()
        with patch('server.get_recognizer') as pretrained:
            result = self.client.post('/api/live-landmarks',
                                      json={'frames': clip(offset=.001)}).get_json()
            pretrained.assert_not_called()
        self.assertEqual(result['source'], 'trained')
        self.assertIn(result['suggestions'][0]['label'], ('my', 'name'))


if __name__ == '__main__':
    unittest.main()
