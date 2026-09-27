"""Optional inference adapter for the locally supplied ASL alphabet checkpoint.

The source model was trained on raw MediaPipe Hands x/y/z coordinates. Do not
mirror, center, or otherwise normalize landmarks before using this adapter.
Hand presence and temporal stability must be checked by the caller; this
classifier has no reliable no-hand class and cannot recognize ASL words.
"""

import os
from pathlib import Path

import numpy as np


LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ("del", "space")
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / ".models" / "asl_mediapipe_mlp_model.h5"
MODEL_ENV = "ASL_ALPHABET_MODEL_PATH"


def _landmark_array(landmarks):
    """Return one unmodified 21x3 MediaPipe hand as a float32 vector."""
    if len(landmarks) == 21 and hasattr(landmarks[0], "x"):
        values = [[point.x, point.y, point.z] for point in landmarks]
    else:
        values = landmarks
    points = np.asarray(values, dtype=np.float32)
    if points.shape not in ((21, 3), (63,)):
        raise ValueError("Expected one hand with 21 x/y/z landmarks")
    if not np.isfinite(points).all():
        raise ValueError("Hand landmarks must contain only finite coordinates")
    return points.reshape(63)


class ASLAlphabetRecognizer:
    """Run the supplied 63→128→64→28 Keras MLP using NumPy alone."""

    labels = LABELS

    def __init__(self, model_path=None):
        configured = model_path if model_path is not None else os.environ.get(MODEL_ENV)
        self.model_path = Path(configured) if configured else DEFAULT_MODEL_PATH
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"ASL alphabet model not found at {self.model_path}. "
                f"Set {MODEL_ENV} to the existing asl_mediapipe_mlp_model.h5 file."
            )
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("Optional ASL alphabet recognition requires h5py") from exc

        self.layers = []
        with h5py.File(self.model_path, "r") as file:
            for name, input_width, output_width in (
                ("dense", 63, 128),
                ("dense_1", 128, 64),
                ("dense_2", 64, len(LABELS)),
            ):
                prefix = f"model_weights/{name}/{name}"
                try:
                    kernel = np.asarray(file[f"{prefix}/kernel:0"], dtype=np.float32)
                    bias = np.asarray(file[f"{prefix}/bias:0"], dtype=np.float32)
                except KeyError as exc:
                    raise ValueError(f"Unexpected ASL alphabet checkpoint layout: {exc}") from exc
                if kernel.shape != (input_width, output_width) or bias.shape != (output_width,):
                    raise ValueError(f"Unexpected ASL alphabet checkpoint dimensions in {name}")
                if not np.isfinite(kernel).all() or not np.isfinite(bias).all():
                    raise ValueError(f"Non-finite ASL alphabet checkpoint weights in {name}")
                self.layers.append((kernel, bias))

    def predict_proba(self, landmarks):
        """Return probabilities in A-Z, del, space order for one visible hand."""
        hidden = _landmark_array(landmarks)
        for kernel, bias in self.layers[:-1]:
            hidden = np.maximum(hidden @ kernel + bias, 0)
        kernel, bias = self.layers[-1]
        logits = hidden @ kernel + bias
        shifted = logits - np.max(logits)
        exponentials = np.exp(shifted)
        return exponentials / np.sum(exponentials)

    def predict(self, landmarks, top_k=3):
        """Return top suggestions with the same label/score keys as the word model."""
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= len(LABELS):
            raise ValueError(f"top_k must be an integer from 1 to {len(LABELS)}")
        probabilities = self.predict_proba(landmarks)
        indices = np.argsort(probabilities)[::-1][:top_k]
        return [{"label": LABELS[index], "score": round(float(probabilities[index]), 4)}
                for index in indices]
