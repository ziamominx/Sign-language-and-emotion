import ast
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sign_recognition import SignLibrary, features


def clip(motion=0.0, handshape=0.0):
    result = []
    for step in range(24):
        hand = []
        for landmark in range(21):
            x = 0.3 + (landmark % 4) * 0.03 + motion * step / 23
            y = 0.3 + (landmark // 4) * 0.025
            if landmark == 5:
                x += 0.05
            if landmark == 17:
                x += 0.15
            if landmark in (8, 12, 16, 20):
                y += handshape
            hand.append([x, y])
        result.append([hand])
    return result


class RecognitionTests(unittest.TestCase):
    def test_translation_and_scale_invariance(self):
        original, _ = features(clip(motion=0.04))
        shifted = np.asarray(clip(motion=0.04)) * 1.2 + 0.06
        transformed, _ = features(shifted.tolist())
        np.testing.assert_allclose(original, transformed, atol=1e-4)

    def test_requires_examples_and_rejects_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            library = SignLibrary(tmp)
            for _ in range(2):
                library.add("HELLO", clip())
            self.assertIsNone(library.recognize(clip()))
            library.add("HELLO", clip())
            self.assertEqual(library.recognize(clip()), "HELLO")
            self.assertIsNone(library.recognize(clip(handshape=0.3)))
            self.assertEqual(SignLibrary(tmp).recognize(clip()), "HELLO")

    def test_rejects_ambiguous_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            library = SignLibrary(tmp)
            for _ in range(3):
                library.add("ONE", clip())
                library.add("TWO", clip())
            self.assertIsNone(library.recognize(clip()))

    def test_recognizes_while_hand_is_visible(self):
        source = ast.parse(Path("zias_glasses.py").read_text(encoding="utf-8"))
        cls = next(node for node in source.body
                   if isinstance(node, ast.ClassDef) and node.name == "Signs")
        namespace = {}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), "zias_glasses.py", "exec"), namespace)
        with tempfile.TemporaryDirectory() as tmp:
            library = SignLibrary(tmp)
            for _ in range(3):
                library.add("HELLO", clip())
            signs = namespace["Signs"](library)
            matches = []
            for frame in clip():
                points = [{index: np.asarray(point) for index, point in enumerate(frame[0])}]
                matches.append(signs.update(points, 1, 0.04, True, 1, 1))
            self.assertIn("HELLO", matches)
            self.assertEqual(signs.words, ["HELLO"])
            signs.update(points, 1, 0.04, True, 1, 1)
            self.assertEqual(signs.words, ["HELLO"])
            signs.update([], 0, 0.3, True, 1, 1)
            self.assertEqual(signs.state, "IDLE")


if __name__ == "__main__":
    unittest.main()
