"""Focused tests for optional, locally supplied ASL fingerspelling inference."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from alphabet_model import ASLAlphabetRecognizer, LABELS, MODEL_ENV


try:
    import h5py
except ImportError:
    h5py = None


@unittest.skipUnless(h5py is not None, "h5py is needed for optional alphabet recognition")
class AlphabetModelTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.checkpoint = Path(self.directory.name) / "alphabet.h5"
        with h5py.File(self.checkpoint, "w") as file:
            for name, shape in (
                ("dense", (63, 128)),
                ("dense_1", (128, 64)),
                ("dense_2", (64, len(LABELS))),
            ):
                group = file.require_group(f"model_weights/{name}/{name}")
                kernel = np.zeros(shape, dtype=np.float32)
                kernel[0, 0] = 1
                bias = np.zeros(shape[1], dtype=np.float32)
                group.create_dataset("kernel:0", data=kernel)
                group.create_dataset("bias:0", data=bias)

    def test_raw_landmarks_keep_original_coordinates(self):
        recognizer = ASLAlphabetRecognizer(self.checkpoint)
        points = np.zeros((21, 3), dtype=np.float32)
        points[0, 0] = 0.8
        probability = recognizer.predict_proba(points)
        expected = np.exp(0.8) / (np.exp(0.8) + len(LABELS) - 1)
        self.assertAlmostEqual(float(probability[0]), expected, places=6)
        self.assertEqual(recognizer.predict(points, top_k=1)[0]["label"], "A")
        self.assertAlmostEqual(float(probability.sum()), 1, places=6)

        objects = [SimpleNamespace(x=p[0], y=p[1], z=p[2]) for p in points]
        np.testing.assert_allclose(recognizer.predict_proba(objects), probability)

    def test_missing_model_has_actionable_error(self):
        with patch.dict(os.environ, {MODEL_ENV: str(self.checkpoint)}) as environment:
            self.assertEqual(ASLAlphabetRecognizer().model_path, self.checkpoint)
            environment[MODEL_ENV] = str(self.checkpoint.with_name("missing.h5"))
            with self.assertRaisesRegex(FileNotFoundError, MODEL_ENV):
                ASLAlphabetRecognizer()

    def test_invalid_input_and_top_k_are_rejected(self):
        recognizer = ASLAlphabetRecognizer(self.checkpoint)
        with self.assertRaisesRegex(ValueError, "21 x/y/z"):
            recognizer.predict(np.zeros((21, 2)))
        points = np.zeros((21, 3), dtype=np.float32)
        points[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            recognizer.predict(points)
        with self.assertRaisesRegex(ValueError, "top_k"):
            recognizer.predict(np.zeros((21, 3)), top_k=0)

    def test_class_order_matches_source_label_encoder(self):
        self.assertEqual(LABELS, tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ("del", "space"))
        self.assertEqual(len(LABELS), 28)


if __name__ == "__main__":
    unittest.main()
