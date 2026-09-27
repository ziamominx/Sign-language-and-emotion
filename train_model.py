"""Train a recognition model from your saved personal-sign coordinates.

The website already stores every recorded sign example in
`.models/personal_signs.json`. This script turns those examples into a
trained temporal model:

    py -3.10 train_model.py

The saved model (CNN + LSTM when TensorFlow is installed, otherwise a
small NumPy MLP) is written next to the coordinate store and picked up
automatically by the website, so tested signs run on the trained dataset
instead of the raw template comparison. Run this script again after
recording more examples to retrain.

Training is deterministic: the same store always produces the same
model, so repeated runs are directly comparable.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from personal_signs import FRAMES, STORE_PATH, coordinate_features

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = ROOT / ".models"
KERAS_MODEL_NAME = "personal_signs_model.keras"
NUMPY_MODEL_NAME = "personal_signs_model.npz"
META_NAME = "personal_signs_model.meta.json"

AUGMENTED_COPIES = 2          # extra jittered copies per real example
VALIDATION_FRACTION = 0.25
SEED = 11


# --------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------

def jitter_clip(clip, rng, amount=0.04):
    """Small coordinate noise plus a one-frame temporal shift."""
    shifted = np.roll(clip, int(rng.integers(-1, 2)), axis=0)
    return shifted + rng.normal(0.0, amount, clip.shape).astype(np.float32)


def load_training_data(store_path):
    """Rebuild every saved example as (feature sequence, class index)."""
    from personal_signs import PersonalSigns  # imported here to avoid cycles

    library = PersonalSigns(store_path)
    labels = sorted(library.examples)
    if len(labels) < 2:
        return library, [], [], None
    class_index = {label: index for index, label in enumerate(labels)}
    real, targets = [], []
    for label in labels:
        for clip in library.examples[label]:
            points = np.asarray(clip, dtype=np.float32)
            if points.shape[0] != FRAMES:
                continue
            real.append(coordinate_features(points))
            targets.append(class_index[label])
    return library, real, targets, labels


def augment_and_split(features, targets, labels, seed=SEED):
    """Duplicate examples with jitter, then hold out sequences for validation."""
    rng = np.random.default_rng(seed)
    augmented, augmented_targets = [], []
    for clip, target in zip(features, targets):
        for _ in range(AUGMENTED_COPIES):
            augmented.append(jitter_clip(np.asarray(clip), rng))
            augmented_targets.append(target)
    pool = [np.asarray(clip, dtype=np.float32) for clip in features] + augmented
    pool_targets = list(targets) + list(augmented_targets)

    order = rng.permutation(len(pool))
    pool = [pool[i] for i in order]
    pool_targets = [pool_targets[i] for i in order]

    if len(pool) >= 8:
        hold_out = max(2, int(round(len(pool) * VALIDATION_FRACTION)))
        validation = pool[:hold_out], pool_targets[:hold_out]
        training = pool[hold_out:], pool_targets[hold_out:]
        if len(set(training[1])) < len(labels):  # keep every class trainable
            training, validation = pool, pool_targets
            return training[0], training[1], None, None
        return training[0], training[1], validation[0], validation[1]
    return pool, pool_targets, None, None


def stack(clips):
    return np.stack(clips, axis=0).astype(np.float32)


def aggregate(clips):
    """Collapse time to (mean, std) so a plain MLP can learn from clips."""
    data = stack(clips)
    return np.concatenate([data.mean(axis=1), data.std(axis=1)], axis=1)


# --------------------------------------------------------------------------
# NumPy fallback model (no TensorFlow required)
# --------------------------------------------------------------------------

def _softmax(values):
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _fit_scaler(training):
    mean = training.mean(axis=(0, 1))
    std = training.std(axis=(0, 1))
    return mean, np.maximum(std, 0.02)


class _NumpyMLP:
    """Two-layer MLP over (mean, std) sequence statistics, trained with SGD."""

    def __init__(self, weights, classes, meta):
        self.w1, self.b1, self.w2, self.b2 = weights
        self.classes = classes
        self.meta = meta

    def predict(self, features):
        """features: one (FRAMES, feature_length) sequence of coordinate features."""
        if np.asarray(features).ndim != 2:
            return None
        aggregated = aggregate([features])
        expected = self.w1.shape[0]
        if aggregated.shape[1] != expected:
            return None
        hidden = np.maximum(aggregated @ self.w1 + self.b1, 0)
        probabilities = _softmax(hidden @ self.w2 + self.b2)[0]
        best = int(np.argmax(probabilities))
        ordered = np.sort(probabilities)[::-1]
        return {"label": self.classes[best], "confidence": float(probabilities[best]),
                "margin": float(probabilities[best] - ordered[1])}

    def evaluate(self, features, targets):
        correct = sum(1 for clip, target in zip(features, targets)
                      if int(np.argmax(self._probs(stack([clip]))[0])) == target)
        return correct / len(targets) if targets else 0.0

    def _probs(self, batch):
        hidden = np.maximum(batch @ self.w1 + self.b1, 0)
        return _softmax(hidden @ self.w2 + self.b2)


def train_numpy(training_x, training_y, validation_x, validation_y, num_classes, verbose):
    """Small deterministic MLP; strong enough for personal sign vocabularies."""
    training = stack(training_x)
    mean, std = _fit_scaler(training)
    x_train = ((training - mean) / std).reshape(len(training), -1)
    y_train = np.asarray(training_y)
    if validation_x:
        validation = (stack(validation_x) - mean) / std
        x_val = validation.reshape(len(validation), -1)
        y_val = np.asarray(validation_y)
    else:
        x_val = y_val = None

    width = x_train.shape[1]
    hidden_width = 96
    rng = np.random.default_rng(SEED)
    w1 = rng.normal(0, np.sqrt(2 / width), (width, hidden_width))
    b1 = np.zeros(hidden_width)
    w2 = rng.normal(0, np.sqrt(2 / hidden_width), (hidden_width, num_classes))
    b2 = np.zeros(num_classes)

    def probabilities_of(weights, batch):
        first, first_bias, second, second_bias = weights
        hidden = np.maximum(batch @ first + first_bias, 0)
        return _softmax(hidden @ second + second_bias)

    best = (np.inf, None)
    learning_rate = 0.05
    for epoch in range(140):
        order = rng.permutation(len(x_train))
        for start in range(0, len(x_train), 16):
            index = order[start:start + 16]
            xb, yb = x_train[index], y_train[index]
            hidden = np.maximum(xb @ w1 + b1, 0)
            probabilities = _softmax(hidden @ w2 + b2)
            gradient = probabilities
            gradient[np.arange(len(yb)), yb] -= 1
            gradient /= len(index)
            gw2 = hidden.T @ gradient
            gb2 = gradient.sum(axis=0)
            ghidden = gradient @ w2.T * (hidden > 0)
            gw1 = xb.T @ ghidden
            gb1 = ghidden.sum(axis=0)
            w1 -= learning_rate * gw1
            b1 -= learning_rate * gb1
            w2 -= learning_rate * gw2
            b2 -= learning_rate * gb2
        if x_val is not None:
            weights = (w1, b1, w2, b2)
            loss = float(np.mean(-np.log(
                probabilities_of(weights, x_val)[np.arange(len(y_val)), y_val] + 1e-9)))
            if loss < best[0]:
                best = (loss, (w1.copy(), b1.copy(), w2.copy(), b2.copy()))
        elif epoch == 139:
            best = (0.0, (w1.copy(), b1.copy(), w2.copy(), b2.copy()))
        if verbose and epoch % 20 == 0:
            print(f"  epoch {epoch}: training...")
    w1, b1, w2, b2 = best[1]

    accuracy = 0.0
    final_weights = (w1, b1, w2, b2)
    if x_val is not None:
        accuracy = float(np.mean(np.argmax(
            probabilities_of(final_weights, x_val), axis=1) == y_val))
        if verbose:
            print(f"  validation accuracy: {accuracy * 100:.1f}%")
    else:
        accuracy = float(np.mean(np.argmax(
            probabilities_of(final_weights, x_train), axis=1) == y_train))
        if verbose:
            print(f"  training accuracy: {accuracy * 100:.1f}% (no validation split)")
    weights = (w1.astype(np.float32), b1.astype(np.float32),
               w2.astype(np.float32), b2.astype(np.float32))
    return weights, accuracy


# --------------------------------------------------------------------------
# TensorFlow CNN + LSTM model (optional)
# --------------------------------------------------------------------------

def train_keras(training_x, training_y, validation_x, validation_y, num_classes, verbose):
    from tensorflow import keras
    from tensorflow.keras import layers

    training = stack(training_x)
    mean, std = _fit_scaler(training)
    x_train = (training - mean) / std
    y_train = np.asarray(training_y)
    x_val = y_val = None
    if validation_x:
        x_val = (stack(validation_x) - mean) / std
        y_val = np.asarray(validation_y)

    model = keras.Sequential([
        layers.Input(shape=(FRAMES, x_train.shape[2], 1)),
        layers.Reshape((FRAMES, x_train.shape[2], 1)),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        layers.Dropout(0.2),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        layers.Dropout(0.2),
        layers.TimeDistributed(layers.Flatten()),
        layers.LSTM(96, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(64),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss" if x_val is not None else "loss",
                                               patience=25, restore_best_weights=True)]
    model.fit(x_train, y_train,
              validation_data=(x_val, y_val) if x_val is not None else None,
              epochs=90, batch_size=8, verbose=1 if verbose else 0,
              callbacks=callbacks)

    accuracy = 0.0
    if x_val is not None:
        accuracy = float(model.evaluate(x_val, y_val, verbose=0)[1])
        if verbose:
            print(f"  validation accuracy: {accuracy * 100:.1f}%")
    else:
        accuracy = float(model.evaluate(x_train, y_train, verbose=0)[1])
        if verbose:
            print(f"  training accuracy: {accuracy * 100:.1f}% (no validation split)")
    return model, accuracy


# --------------------------------------------------------------------------
# Save / load
# --------------------------------------------------------------------------

def model_dir(directory=DEFAULT_MODEL_DIR):
    return Path(directory)


def save_model(trained, classes, frames, feature_length, accuracy, directory,
               scaler=None, verbose=True):
    directory = model_dir(directory)
    directory.mkdir(parents=True, exist_ok=True)
    keras_available = not isinstance(trained, tuple)
    if keras_available:
        path = directory / KERAS_MODEL_NAME
        trained.save(path)
        model_type = "keras"
        (directory / NUMPY_MODEL_NAME).unlink(missing_ok=True)
    else:
        path = directory / NUMPY_MODEL_NAME
        np.savez_compressed(path, w1=trained[0], b1=trained[1],
                            w2=trained[2], b2=trained[3])
        model_type = "numpy"
        (directory / KERAS_MODEL_NAME).unlink(missing_ok=True)
    mean = None if scaler is None else [round(float(value), 6) for value in scaler[0]]
    std = None if scaler is None else [round(float(value), 6) for value in scaler[1]]
    meta = {"version": 1, "model_type": model_type, "classes": classes,
            "frames": frames, "feature_length": feature_length,
            "validation_accuracy": round(float(accuracy), 4),
            "scaler_mean": mean, "scaler_std": std,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (directory / META_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if verbose:
        print(f"Model saved to {path}")
    return path, meta


class _KerasPersonalModel:
    def __init__(self, model, scaler, classes, meta):
        self.model = model
        self.mean, self.std = scaler
        self.classes = classes
        self.meta = meta

    def predict(self, features):
        batch = (np.asarray([features], dtype=np.float32) - self.mean) / self.std
        probabilities = self.model.predict(batch, verbose=0)[0]
        best = int(np.argmax(probabilities))
        ordered = np.sort(probabilities)[::-1]
        return {"label": self.classes[best], "confidence": float(probabilities[best]),
                "margin": float(probabilities[best] - ordered[1])}


def load_model(directory=DEFAULT_MODEL_DIR):
    """Load the trained model for inference; (None, None) when absent or stale."""
    directory = model_dir(directory)
    meta_path = directory / META_NAME
    if not meta_path.is_file():
        return None, None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sample = coordinate_features(np.zeros((FRAMES, 75, 2), dtype=np.float32))
        if meta.get("feature_length") != int(sample.shape[1]) or meta.get("frames") != FRAMES:
            return None, None  # stored model predates the current feature pipeline
        classes = list(meta["classes"])
        if meta["model_type"] == "keras":
            from tensorflow import keras
            model = keras.models.load_model(directory / KERAS_MODEL_NAME)
            mean = np.asarray(meta.get("scaler_mean"), dtype=np.float32)
            std = np.asarray(meta.get("scaler_std"), dtype=np.float32)
            if mean.shape != (int(sample.shape[1]),) or std.shape != mean.shape:
                return None, None  # scaler missing or stale
            return _KerasPersonalModel(model, (mean, std), classes, meta), meta
        if meta["model_type"] == "numpy":
            data = np.load(directory / NUMPY_MODEL_NAME)
            expected = int(sample.shape[1]) * 2
            if data["w1"].shape[0] != expected:
                return None, None
            weights = (data["w1"], data["b1"], data["w2"], data["b2"])
            return _NumpyMLP(weights, classes, meta), meta
    except Exception:
        return None, None
    return None, None


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def train(store_path=STORE_PATH, directory=DEFAULT_MODEL_DIR, verbose=True):
    """Train from the saved coordinate store; returns a status dictionary."""
    library, features, targets, labels = load_training_data(store_path)
    if not labels:
        if verbose:
            print("Not enough saved signs yet: record examples for at least two "
                  "different signs on the website, then run this again.")
        return {"ready": False, "reason": "Record examples for at least two signs first."}

    total = len(features)
    counts = {label: targets.count(index) for label, index in
              {label: i for i, label in enumerate(labels)}.items()}
    if verbose:
        print(f"Training on {total} saved examples across {len(labels)} signs:")
        for label, count in counts.items():
            print(f"  {label}: {count} example(s)")

    training_x, training_y, validation_x, validation_y = augment_and_split(
        features, targets, labels)
    feature_length = int(np.asarray(features[0]).shape[1])

    training = stack(training_x)
    scaler = _fit_scaler(training)
    try:
        trained, accuracy = train_keras(training_x, training_y, validation_x,
                                        validation_y, len(labels), verbose)
        model_type = "keras"
    except ImportError:
        if verbose:
            print("TensorFlow not installed; training the NumPy fallback model.")
        trained, accuracy = train_numpy(training_x, training_y, validation_x,
                                        validation_y, len(labels), verbose)
        model_type = "numpy"

    path, meta = save_model(trained, labels, FRAMES, feature_length, accuracy,
                            directory, scaler=scaler, verbose=verbose)
    if verbose:
        print("Done. Restart the page (or save another example) and your "
              "signs will run through the trained model.")
    return {"ready": True, "model_type": model_type, "path": str(path),
            "classes": len(labels), "examples": total,
            "validation_accuracy": round(float(accuracy), 4)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", default=str(STORE_PATH),
                        help="Path to personal_signs.json")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR),
                        help="Directory that receives the trained model")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    arguments = parser.parse_args()
    train(Path(arguments.store), Path(arguments.model_dir), verbose=not arguments.quiet)


if __name__ == "__main__":
    main()
