"""Saved coordinate examples must be reusable and reject unrelated motion."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from personal_signs import PersonalSigns
from server import app


def clip(wrist_x=.4, finger_spread=.012, offset=0):
    frames = np.zeros((18, 75, 2), dtype=np.float32)
    for t in range(18):
        frames[t, 0] = [.5, .2]
        frames[t, 11] = [.4, .4]
        frames[t, 12] = [.6, .4]
        frames[t, 13:17] = [[.38, .5], [.62, .5], [.38, .6], [.62, .6]]
        x = wrist_x + offset + t * .001
        for point in range(21):
            frames[t, 33 + point] = [x + (point % 5) * finger_spread,
                                     .55 - (point // 5) * .015]
    return frames.tolist()


class PersonalSignsTests(unittest.TestCase):
    def test_three_examples_persist_and_match_a_similar_clip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'signs.json'
            library = PersonalSigns(path)
            for offset in (0, .002, -.002):
                library.add('My', clip(offset=offset))
            restored = PersonalSigns(path)
            self.assertTrue(restored.summary()[0]['ready'])
            self.assertEqual(restored.match(clip(offset=.001))['label'], 'my')
            self.assertIsNone(restored.match(clip(wrist_x=.75, finger_spread=.035)))

    def test_incomplete_training_does_not_make_a_live_guess(self):
        with tempfile.TemporaryDirectory() as directory:
            library = PersonalSigns(Path(directory) / 'signs.json')
            library.add('name', clip())
            self.assertIsNone(library.match(clip()))

    def test_api_saves_examples_and_uses_them_in_live_recognition(self):
        with tempfile.TemporaryDirectory() as directory:
            library = PersonalSigns(Path(directory) / 'signs.json')
            client = app.test_client()
            with patch('server.get_personal_signs', return_value=library):
                for offset in (0, .002, -.002):
                    response = client.post('/api/personal-signs',
                                           json={'label': 'my', 'frames': clip(offset=offset)})
                    self.assertEqual(response.status_code, 200)
                self.assertTrue(client.get('/api/personal-signs').get_json()['signs'][0]['ready'])
                with patch('server.get_recognizer') as pretrained:
                    result = client.post('/api/live-landmarks',
                                         json={'frames': clip(offset=.001)}).get_json()
                    pretrained.assert_not_called()
                self.assertEqual(result['suggestions'][0]['label'], 'my')
                self.assertEqual(result['source'], 'personal')
                self.assertEqual(client.delete('/api/personal-signs/my').status_code, 200)
                self.assertEqual(client.get('/api/personal-signs').get_json()['signs'], [])

    def test_no_hand_recording_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            library = PersonalSigns(Path(directory) / 'signs.json')
            with self.assertRaisesRegex(ValueError, 'signing hands'):
                library.add('my', np.zeros((18, 75, 2)).tolist())


if __name__ == '__main__':
    unittest.main()
