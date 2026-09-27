"""Local website for pretrained, isolated ASL word recognition."""

from pathlib import Path
from contextlib import ExitStack
from importlib.util import find_spec
import os
import threading
from threading import Lock, RLock
from time import perf_counter

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, render_template, request

from asl_model import ASLRecognizer
from alphabet_model import ASLAlphabetRecognizer
from personal_signs import PersonalSigns
from train_model import load_model as load_trained_model
from train_model import train as run_training
from arena.api import arena_bp

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(ROOT / "web" / "templates"),
            static_folder=str(ROOT / "web" / "static"))
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024
app.register_blueprint(arena_bp)
recognizer = None
recognizer_lock = Lock()
alphabet_recognizer = None
alphabet_recognizer_lock = Lock()
tracker = None
tracker_lock = RLock()
face_tracker = None
face_tracker_lock = RLock()
pose_tracker = None
pose_tracker_lock = RLock()
personal_signs = None
personal_signs_lock = Lock()
trained_model = None
trained_model_lock = Lock()
training_state = {"running": False, "result": None}
training_lock = Lock()

# The trained model must beat these gates before it may auto-add a word;
# otherwise the request falls through to the template matcher and the
# pretrained 2,000-word model.
TRAINED_MIN_CONFIDENCE = 0.72
TRAINED_MIN_MARGIN = 0.12


def get_trained_model():
    """Load the trained personal-sign model once; invalidate when retrained."""
    global trained_model
    if trained_model is None:
        with trained_model_lock:
            if trained_model is None:
                trained_model = load_trained_model(get_personal_signs().path.parent)[0]
    return trained_model


def get_tracker():
    """Keep a lightweight hand tracker warm for responsive camera feedback."""
    global tracker
    if tracker is None:
        with tracker_lock:
            if tracker is None:
                tracker = mp.solutions.hands.Hands(
                    static_image_mode=False, max_num_hands=2, model_complexity=1,
                    min_detection_confidence=0.55, min_tracking_confidence=0.5)
    return tracker


def get_face_tracker():
    global face_tracker
    if face_tracker is None:
        with face_tracker_lock:
            if face_tracker is None:
                face_tracker = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False, max_num_faces=1,
                    refine_landmarks=False, min_detection_confidence=0.35,
                    min_tracking_confidence=0.45)
    return face_tracker


def get_pose_tracker():
    global pose_tracker
    if pose_tracker is None:
        with pose_tracker_lock:
            if pose_tracker is None:
                pose_tracker = mp.solutions.pose.Pose(
                    static_image_mode=False, model_complexity=0,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
    return pose_tracker


def get_personal_signs():
    global personal_signs
    if personal_signs is None:
        with personal_signs_lock:
            if personal_signs is None:
                personal_signs = PersonalSigns()
    return personal_signs


def invalidate_trained_model():
    """Pick up a freshly trained model on the next recognition request."""
    global trained_model
    with trained_model_lock:
        trained_model = None


def _open_palm(landmarks):
    """Require four extended fingers, regardless of hand rotation or mirror."""
    wrist = np.array([landmarks[0].x, landmarks[0].y])
    extended = 0
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        tip_distance = np.linalg.norm(np.array([landmarks[tip].x, landmarks[tip].y]) - wrist)
        pip_distance = np.linalg.norm(np.array([landmarks[pip].x, landmarks[pip].y]) - wrist)
        extended += tip_distance > pip_distance * 1.18
    return extended == 4


def classify_expression(face_image):
    """Classify a detected face crop; never infer emotion from a blank frame."""
    # DeepFace logs emoji during import, which crashes under Windows cp1252.
    os.environ.setdefault("DEEPFACE_LOG_LEVEL", "60")
    from deepface import DeepFace

    analysis = DeepFace.analyze(face_image, actions=["emotion"],
                                detector_backend="skip", enforce_detection=False,
                                silent=True)
    if isinstance(analysis, list):
        analysis = analysis[0]
    return choose_expression(analysis["dominant_emotion"], analysis["emotion"])


EXPRESSION_SMOOTHING_WINDOW = 3
_expression_history = []


def _record_expression(expression):
    """Keep recent expression readings so mild changes survive a jittery model."""
    _expression_history.append((str(expression.get("label", "")),
                                float(expression.get("score", 0)),
                                bool(expression.get("tentative"))))
    del _expression_history[:-EXPRESSION_SMOOTHING_WINDOW]


def smoothed_expression():
    """Fuse the last readings; a tentative majority becomes a confident guess."""
    if not _expression_history:
        return None
    if len(_expression_history) < EXPRESSION_SMOOTHING_WINDOW:
        label, score, tentative = _expression_history[-1]
        return {"label": label, "score": round(score, 1), "tentative": tentative}
    tally = {}
    for label, score, tentative in _expression_history:
        entry = tally.setdefault(label, {"count": 0, "score": 0.0, "tentative": 0})
        entry["count"] += 1
        entry["score"] += score
        entry["tentative"] += int(tentative)
    best_label, best = max(tally.items(), key=lambda item: item[1]["count"])
    if best["count"] < 2:
        label, score, tentative = _expression_history[-1]
        return {"label": label, "score": round(score, 1), "tentative": tentative}
    return {"label": best_label, "score": round(best["score"] / best["count"], 1),
            "tentative": best["tentative"] < best["count"]}


def choose_expression(dominant, raw_scores):
    """Promote happy/sad when DeepFace's neutral bias nearly ties them.

    DeepFace's training set makes neutral absorb mild expressions, so the
    dominant label alone almost always reads Neutral. Compare the gap to
    neutral against the spread between the other expressions: when an
    expression is meaningfully ahead of neutral but not of its peers, it
    is shown as tentative; when it is genuinely competitive, it wins.
    """
    scores = {key: float(value) for key, value in raw_scores.items()}
    label = str(dominant).lower()
    if label not in {"angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"}:
        raise ValueError("Unknown expression label")
    neutral = scores.get("neutral", 0)
    expressive = {name: value for name, value in scores.items() if name != "neutral"}
    best_expressive = max(expressive, key=expressive.get, default=None)
    runner_up = max((value for name, value in expressive.items() if name != best_expressive),
                    default=0)
    tentative = False
    if best_expressive:
        gap_to_runner = expressive[best_expressive] - runner_up
        gap_to_neutral = neutral - expressive[best_expressive]
        if expressive[best_expressive] >= 18:
            if gap_to_neutral <= 0 and gap_to_runner >= 8:
                label = best_expressive
            elif 0 < gap_to_neutral <= 12 and gap_to_runner >= 6:
                label = best_expressive
                tentative = True
    return {"label": label.capitalize(), "score": round(scores.get(label, 0), 1),
            "tentative": tentative}


def get_recognizer():
    global recognizer
    if recognizer is None:
        with recognizer_lock:
            if recognizer is None:
                recognizer = ASLRecognizer()
    return recognizer


def get_alphabet_recognizer():
    global alphabet_recognizer
    if alphabet_recognizer is None:
        with alphabet_recognizer_lock:
            if alphabet_recognizer is None:
                alphabet_recognizer = ASLAlphabetRecognizer()
    return alphabet_recognizer


def _hands_by_side(result):
    """Map Hands' selfie-image labels to anatomical sides in our raw frames."""
    sides = {}
    landmarks = result.multi_hand_landmarks or []
    handedness = result.multi_handedness or []
    for hand, classification in zip(landmarks, handedness):
        if not classification.classification:
            continue
        label = classification.classification[0].label
        # Browser frames are not flipped for inference. MediaPipe Hands assumes
        # mirrored input, while Holistic's left/right slots are anatomical.
        if label == "Left":
            sides["right"] = hand.landmark
        elif label == "Right":
            sides["left"] = hand.landmark
    return sides


def _visibility_guidance(hand_ratio, pose_ratio):
    if hand_ratio < 0.4 and pose_ratio < 0.4:
        return "Step back and center your head, shoulders, and signing hands in the camera."
    if hand_ratio < 0.4:
        return "Show your signing hands clearly inside the camera frame."
    if pose_ratio < 0.4:
        return "Step back until your head and shoulders are visible with your hands."
    return "Your upper body and signing hands are visible."


def extract_keypoints(files, diagnostics=None):
    frames = []
    hand_frames = 0
    pose_frames = 0
    fallback_frames = 0
    with ExitStack() as stack:
        holistic = stack.enter_context(mp.solutions.holistic.Holistic(
            static_image_mode=False, model_complexity=0,
            min_detection_confidence=0.5, min_tracking_confidence=0.5))
        hands = None
        for upload in files:
            data = np.frombuffer(upload.read(), np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image is None or image.shape[0] < 120 or image.shape[1] < 120:
                raise ValueError("The camera frames could not be read")
            image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = holistic.process(rgb)
            points = np.zeros((75, 2), dtype=np.float32)
            if result.pose_landmarks:
                pose_frames += 1
                points[:33] = [(p.x, p.y) for p in result.pose_landmarks.landmark]
            left = result.left_hand_landmarks.landmark if result.left_hand_landmarks else None
            right = result.right_hand_landmarks.landmark if result.right_hand_landmarks else None
            if left is None or right is None:
                if hands is None:
                    hands = stack.enter_context(mp.solutions.hands.Hands(
                        static_image_mode=False, max_num_hands=2, model_complexity=0,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5))
                recovered = _hands_by_side(hands.process(rgb))
                recovered_side = False
                if left is None and recovered.get("left") is not None:
                    left = recovered["left"]
                    recovered_side = True
                if right is None and recovered.get("right") is not None:
                    right = recovered["right"]
                    recovered_side = True
                if recovered_side:
                    fallback_frames += 1
            if left is not None:
                hand_frames += 1
                points[33:54] = [(p.x, p.y) for p in left]
            if right is not None:
                hand_frames += 1 if left is None else 0
                points[54:75] = [(p.x, p.y) for p in right]
            frames.append(points)
    if diagnostics is not None:
        diagnostics.update({"hands": round(hand_frames / len(frames), 2),
                            "pose": round(pose_frames / len(frames), 2),
                            "fallback_frames": fallback_frames})
    if hand_frames < len(frames) * 0.4 or pose_frames < len(frames) * 0.4:
        raise ValueError("Keep your upper body and signing hands visible in the camera")
    return np.stack(frames)


@app.get("/")
def index():
    return render_template("home.html")


@app.get("/communicator")
def communicator_page():
    return render_template("communicator.html")


@app.get("/charades")
def charades_page():
    return render_template("charades.html")


@app.get("/memes")
def memes_page():
    return render_template("memes.html")


@app.get("/history")
def history_page():
    return render_template("history.html")


@app.get("/settings")
def settings_page():
    return render_template("settings.html")


@app.get("/lab")
def lab_page():
    return render_template("lab.html")


@app.get("/api/status")
def status():
    try:
        model = get_recognizer()
        return jsonify({"ready": True, "labels": len(model.labels)})
    except Exception as exc:
        app.logger.exception("Model unavailable")
        return jsonify({"ready": False, "error": str(exc)}), 503


@app.get("/api/alphabet/status")
def alphabet_status():
    """The optional local checkpoint may be absent without breaking word mode."""
    try:
        model = get_alphabet_recognizer()
        return jsonify({"ready": True, "labels": len(model.labels), "experimental": True})
    except FileNotFoundError as exc:
        return jsonify({"ready": False, "experimental": True, "error": str(exc)})
    except Exception:
        app.logger.exception("Alphabet model unavailable")
        return jsonify({"ready": False, "experimental": True,
                        "error": "Alphabet model could not be loaded"}), 503


@app.post("/api/track")
def track_hands():
    """Return hand landmarks for the live overlay and the finish gesture."""
    upload = request.files.get("frame")
    if upload is None:
        return jsonify({"error": "Send one JPEG camera frame in the frame field"}), 400
    data = upload.read()
    if not data.startswith(b"\xff\xd8"):
        return jsonify({"error": "The camera frame must be a JPEG image"}), 422
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None or min(image.shape[:2]) < 120:
        return jsonify({"error": "The camera frame could not be read"}), 422
    try:
        with tracker_lock:
            # Match the original desktop preview: detect on the mirrored frame.
            result = get_tracker().process(cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB))
        landmarks = result.multi_hand_landmarks or []
        model_hands = {"left": None, "right": None}
        for hand, handedness in zip(landmarks, getattr(result, "multi_handedness", None) or []):
            if handedness.classification:
                side = handedness.classification[0].label.lower()
                if side in model_hands:
                    # Overlay tracking uses a mirrored image. Restore the raw
                    # camera coordinates expected by the word model.
                    model_hands[side] = [[round(1 - point.x, 4), round(point.y, 4)]
                                         for point in hand.landmark]
        return jsonify({
            "hands": [[[round(point.x, 4), round(point.y, 4)] for point in hand.landmark]
                      for hand in landmarks],
            "model_hands": model_hands,
            "finish_gesture": len(landmarks) == 2 and all(
                _open_palm(hand.landmark) for hand in landmarks),
        })
    except Exception:
        app.logger.exception("Hand tracking failed")
        return jsonify({"error": "Hand tracking failed"}), 500


@app.post("/api/pose-track")
def track_pose():
    """Supply body landmarks for fast word inference without clip reprocessing."""
    upload = request.files.get("frame")
    if upload is None:
        return jsonify({"error": "Send one JPEG camera frame in the frame field"}), 400
    data = upload.read()
    if not data.startswith(b"\xff\xd8"):
        return jsonify({"error": "The camera frame must be a JPEG image"}), 422
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None or min(image.shape[:2]) < 120:
        return jsonify({"error": "The camera frame could not be read"}), 422
    try:
        with pose_tracker_lock:
            result = get_pose_tracker().process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pose = result.pose_landmarks.landmark if result.pose_landmarks else None
        return jsonify({"pose": [[round(point.x, 4), round(point.y, 4)] for point in pose]
                        if pose else None})
    except Exception:
        app.logger.exception("Body tracking failed")
        return jsonify({"error": "Body tracking failed"}), 500


@app.post("/api/face-track")
def track_face():
    """Provide the face outline and feature landmarks in preview coordinates."""
    upload = request.files.get("frame")
    if upload is None:
        return jsonify({"error": "Send one JPEG camera frame in the frame field"}), 400
    data = upload.read()
    if not data.startswith(b"\xff\xd8"):
        return jsonify({"error": "The camera frame must be a JPEG image"}), 422
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None or min(image.shape[:2]) < 120:
        return jsonify({"error": "The camera frame could not be read"}), 422
    try:
        with face_tracker_lock:
            result = get_face_tracker().process(
                cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB))
        if not result.multi_face_landmarks:
            return jsonify({"face": None, "box": None})
        face = result.multi_face_landmarks[0].landmark
        points = [[round(point.x, 4), round(point.y, 4)] for point in face]
        return jsonify({"face": points,
                        "box": [min(point[0] for point in points),
                                min(point[1] for point in points),
                                max(point[0] for point in points),
                                max(point[1] for point in points)]})
    except Exception:
        app.logger.exception("Face tracking failed")
        return jsonify({"error": "Face tracking failed"}), 500


@app.get("/api/emotion/status")
def emotion_status():
    available = find_spec("deepface") is not None
    return jsonify({"ready": available,
                    "error": None if available else "Install optional DeepFace support to show facial expressions"})


@app.post("/api/emotion")
def emotion_predict():
    """Estimate a facial expression only when a face is actually detected."""
    if find_spec("deepface") is None:
        return jsonify({"error": "Facial expression model is not installed"}), 503
    upload = request.files.get("frame")
    if upload is None:
        return jsonify({"error": "Send one JPEG camera frame in the frame field"}), 400
    data = upload.read()
    if not data.startswith(b"\xff\xd8"):
        return jsonify({"error": "The camera frame must be a JPEG image"}), 422
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None or min(image.shape[:2]) < 120:
        return jsonify({"error": "The camera frame could not be read"}), 422
    height, width = image.shape[:2]
    try:
        with mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.6) as detector:
            found = detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not found.detections:
            return jsonify({"visible": False, "label": None})
        box = found.detections[0].location_data.relative_bounding_box
        pad_x, pad_y = box.width * 0.22, box.height * 0.30
        x1 = max(0, int((box.xmin - pad_x) * width))
        y1 = max(0, int((box.ymin - pad_y) * height))
        x2 = min(width, int((box.xmin + box.width + pad_x) * width))
        y2 = min(height, int((box.ymin + box.height + pad_y) * height))
        if x2 - x1 < 48 or y2 - y1 < 48:
            return jsonify({"visible": False, "label": None})
        expression = classify_expression(image[y1:y2, x1:x2])
        _record_expression(expression)
        return jsonify({"visible": True, **smoothed_expression()})
    except Exception:
        app.logger.exception("Facial expression analysis failed")
        return jsonify({"error": "Facial expression analysis unavailable"}), 503


@app.post("/api/alphabet/predict")
def alphabet_predict():
    """Recognize one visible hand as an experimental fingerspelling symbol."""
    upload = request.files.get("frame")
    if upload is None:
        return jsonify({"error": "Send one JPEG camera frame in the frame field"}), 400
    data = upload.read()
    if not data.startswith(b"\xff\xd8"):
        return jsonify({"error": "The camera frame must be a JPEG image"}), 422
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None or min(image.shape[:2]) < 120:
        return jsonify({"error": "The camera frame could not be read"}), 422
    try:
        model = get_alphabet_recognizer()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception:
        app.logger.exception("Alphabet model unavailable")
        return jsonify({"error": "Alphabet model could not be loaded"}), 503

    started = perf_counter()
    try:
        # Each request is independent, so a static-image detector must locate
        # the hand afresh. Preserve the raw x/y/z coordinates used in training.
        with mp.solutions.hands.Hands(
                static_image_mode=True, max_num_hands=1, model_complexity=0,
                min_detection_confidence=0.6) as hands:
            result = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not result.multi_hand_landmarks:
            return jsonify({"visible": False, "suggestions": [], "uncertain": True,
                            "guidance": "Show one hand clearly near the center of the camera.",
                            "processing_ms": round((perf_counter() - started) * 1000)})
        predictions = model.predict(result.multi_hand_landmarks[0].landmark)
    except Exception:
        app.logger.exception("Alphabet recognition failed")
        return jsonify({"error": "Alphabet recognition failed; please try again"}), 500

    uncertain = predictions[0]["score"] < 0.65 or (
        predictions[0]["score"] - predictions[1]["score"] < 0.15)
    return jsonify({"visible": True, "suggestions": predictions, "uncertain": uncertain,
                    "guidance": "Hold one letter clearly; review the suggestion before adding it.",
                    "processing_ms": round((perf_counter() - started) * 1000)})


@app.post("/api/recognize")
def recognize():
    files = request.files.getlist("frames")
    if not 12 <= len(files) <= 48:
        return jsonify({"error": "Send between 12 and 48 camera frames"}), 400
    try:
        points = extract_keypoints(files)
        predictions = get_recognizer().predict(points)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422
    except Exception:
        app.logger.exception("Recognition failed")
        return jsonify({"error": "Recognition failed; please try again"}), 500
    uncertain = predictions[0]["score"] < 0.45 or (
        predictions[0]["score"] - predictions[1]["score"] < 0.10)
    return jsonify({"suggestions": predictions, "uncertain": uncertain,
                    "frames": len(points)})


@app.post("/api/live")
def live():
    """Classify a short rolling clip; absence of a signer is a normal live state."""
    files = request.files.getlist("frames")
    if not 12 <= len(files) <= 32:
        return jsonify({"error": "Send between 12 and 32 camera frames"}), 400
    started = perf_counter()
    visibility = {}
    try:
        points = extract_keypoints(files, diagnostics=visibility)
        predictions = get_recognizer().predict(points)
    except ValueError as exc:
        if str(exc) == "Keep your upper body and signing hands visible in the camera":
            return jsonify({"visible": False, "suggestions": [], "uncertain": True,
                            "visibility": visibility,
                            "guidance": _visibility_guidance(visibility["hands"], visibility["pose"]),
                            "processing_ms": round((perf_counter() - started) * 1000)})
        return jsonify({"error": str(exc)}), 422
    except Exception:
        app.logger.exception("Live recognition failed")
        return jsonify({"error": "Recognition failed; please try again"}), 500
    uncertain = predictions[0]["score"] < 0.45 or (
        predictions[0]["score"] - predictions[1]["score"] < 0.10)
    return jsonify({"visible": True, "suggestions": predictions, "uncertain": uncertain,
                    "frames": len(points),
                    "visibility": visibility,
                    "guidance": _visibility_guidance(visibility["hands"], visibility["pose"]),
                    "processing_ms": round((perf_counter() - started) * 1000)})


@app.post("/api/live-landmarks")
def live_landmarks():
    """Classify a rolling clip built from already-tracked camera landmarks."""
    started = perf_counter()
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or "frames" not in payload:
        return jsonify({"error": "Send landmark frames as JSON"}), 400
    try:
        points = np.asarray(payload["frames"], dtype=np.float32)
    except (TypeError, ValueError):
        return jsonify({"error": "Expected numeric landmark frames"}), 422
    if (points.ndim != 3 or not 12 <= len(points) <= 32 or
            points.shape[1:] != (75, 2) or not np.isfinite(points).all()):
        return jsonify({"error": "Expected 12–32 finite frames of 75 x/y landmarks"}), 422
    pose_ratio = float(np.count_nonzero(np.any(points[:, 11:17] != 0, axis=(1, 2)))) / len(points)
    hand_ratio = float(np.count_nonzero(np.any(points[:, 33:] != 0, axis=(1, 2)))) / len(points)
    visibility = {"hands": round(hand_ratio, 2), "pose": round(pose_ratio, 2),
                  "fallback_frames": 0}
    guidance = _visibility_guidance(hand_ratio, pose_ratio)
    if min(hand_ratio, pose_ratio) < 0.4:
        return jsonify({"visible": False, "suggestions": [], "uncertain": True,
                        "visibility": visibility, "guidance": guidance,
                        "processing_ms": round((perf_counter() - started) * 1000)})
    try:
        trained_label, trained_confidence = _personal_from_trained_model(points)
        if trained_label:
            return jsonify({"visible": True, "suggestions": [{"label": trained_label,
                            "score": round(trained_confidence, 4), "source": "trained"}],
                            "uncertain": False,
                            "source": "trained", "frames": len(points),
                            "visibility": visibility, "guidance": guidance,
                            "processing_ms": round((perf_counter() - started) * 1000)})
        personal = get_personal_signs().match(points)
        if personal:
            score = round(0.82 + 0.16 * (1 - personal["distance"] /
                                           personal["threshold"]), 4)
            return jsonify({"visible": True, "suggestions": [{"label": personal["label"],
                            "score": score, "source": "personal"}], "uncertain": False,
                            "source": "personal", "frames": len(points),
                            "visibility": visibility, "guidance": guidance,
                            "processing_ms": round((perf_counter() - started) * 1000)})
        predictions = get_recognizer().predict(points)
    except Exception:
        app.logger.exception("Landmark recognition failed")
        return jsonify({"error": "Recognition failed; please try again"}), 500
    uncertain = predictions[0]["score"] < 0.45 or (
        predictions[0]["score"] - predictions[1]["score"] < 0.10)
    return jsonify({"visible": True, "suggestions": predictions,
                    "uncertain": uncertain, "frames": len(points),
                    "visibility": visibility, "guidance": guidance,
                    "processing_ms": round((perf_counter() - started) * 1000)})


def _training_paths():
    """Reuse the active personal-sign store and .models directory for training."""
    store = get_personal_signs().path
    return store, store.parent


def _training_worker():
    try:
        store_path, directory = _training_paths()
        result = run_training(store_path, directory, verbose=False)
        with training_lock:
            training_state["result"] = result
    except Exception as exc:
        app.logger.exception("Training failed")
        with training_lock:
            training_state["result"] = {"ready": False, "error": str(exc)}
    finally:
        with training_lock:
            training_state["running"] = False
        if training_state["result"].get("ready"):
            invalidate_trained_model()


@app.post("/api/personal-signs/train")
def train_personal_signs():
    """Train on the saved coordinate examples; runs in the background."""
    with training_lock:
        if training_state["running"]:
            return jsonify({"error": "Training is already running", "running": True}), 409
        training_state["running"] = True
        training_state["result"] = None
    worker = threading.Thread(target=_training_worker, daemon=True)
    worker.start()
    return jsonify({"running": True})


@app.get("/api/personal-signs/train/status")
def training_status():
    with training_lock:
        state = {"running": training_state["running"],
                 "result": training_state["result"]}
    return jsonify(state)


def _personal_from_trained_model(points):
    """Query the trained model; returns (label, confidence) or (None, None)."""
    trained = get_trained_model()
    if trained is None:
        return None, None
    from personal_signs import coordinate_features, validate_clip
    try:
        prediction = trained.predict(coordinate_features(validate_clip(points)))
    except ValueError:
        return None, None
    if prediction is None or prediction["confidence"] < TRAINED_MIN_CONFIDENCE \
            or prediction["margin"] < TRAINED_MIN_MARGIN:
        return None, None
    return prediction["label"], prediction["confidence"]


@app.route("/api/personal-signs", methods=["GET", "POST"])
def personal_signs_api():
    library = get_personal_signs()
    if request.method == "GET":
        try:
            trained = get_trained_model()
        except Exception:
            app.logger.exception("Trained model failed to load")
            trained = None
        return jsonify({"signs": library.summary(), "required_examples": 3,
                        "trained": bool(trained),
                        "trained_accuracy": trained.meta.get("validation_accuracy") if trained else None})
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send a sign label and coordinate frames as JSON"}), 400
    try:
        saved = library.add(payload.get("label"), payload.get("frames"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422
    return jsonify({"saved": saved, "signs": library.summary()})


@app.delete("/api/personal-signs/<path:label>")
def delete_personal_sign(label):
    try:
        get_personal_signs().remove(label)
    except KeyError:
        return jsonify({"error": "That saved sign was not found"}), 404
    return jsonify({"signs": get_personal_signs().summary()})


if __name__ == "__main__":
    print("Open http://127.0.0.1:8000 in your browser")
    app.run(host="127.0.0.1", port=8000, debug=False)
