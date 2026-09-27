"""The trainer must learn saved signs and honor the feature contract."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from personal_signs import PersonalSigns, coordinate_features
from train_model import load_model, train


def clip(wrist_x=.4, finger_spread=.012, offset=0, droop=0.0):
    """Two visibly different clips: 'my' near the chest, 'name' higher up."""
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


def build_store(path):
    library = PersonalSigns(path)
    for offset in (0, .002, -.002):
        library.add('my', clip(offset=offset))
    for offset in (0, .002, -.002):
        library.add('name', clip(wrist_x=.42, offset=offset, droop=.08))
    return library


class TrainModelTests(unittest.TestCase):
    def test_trains_and_matches_a_saved_sign(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_store(root / 'signs.json')
            result = train(root / 'signs.json', root / 'models', verbose=False)
            self.assertTrue(result['ready'])
            self.assertEqual(result['classes'], 2)
            model, meta = load_model(root / 'models')
            self.assertIsNotNone(model)
            self.assertEqual(model.classes, ['my', 'name'])
            prediction = model.predict(
                coordinate_features(np.asarray(clip(offset=.001), dtype=np.float32)))
            self.assertIn(prediction['label'], ('my', 'name'))
            self.assertGreaterEqual(prediction['confidence'], 0.0)

    def test_load_returns_none_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            model, meta = load_model(Path(directory))
            self.assertIsNone(model)
            self.assertIsNone(meta)

    def test_stale_model_with_wrong_features_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_store(root / 'signs.json')
            train(root / 'signs.json', root / 'models', verbose=False)
            model_dir = root / 'models'
            for path in model_dir.glob('personal_signs_model.*'):
                path.write_bytes(b'corrupt')
            model, _ = load_model(model_dir)
            self.assertIsNone(model)


if __name__ == '__main__':
    unittest.main()
