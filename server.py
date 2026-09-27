"""Local website for pretrained, isolated ASL word recognition."""

from pathlib import Path
from contextlib import ExitStack
from threading import Lock, RLock
from time import perf_counter

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, render_template, request

from asl_model import ASLRecognizer
from alphabet_model import ASLAlphabetRecognizer


ROOT = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(ROOT / "web" / "templates"),
            static_folder=str(ROOT / "web" / "static"))
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024
recognizer = None
recognizer_lock = Lock()
alphabet_recognizer = None
alphabet_recognizer_lock = Lock()
tracker = None
tracker_lock = RLock()


def get_tracker():
    """Keep a lightweight hand tracker warm for responsive camera feedback."""
    global tracker
    if tracker is None:
        with tracker_lock:
            if tracker is None:
                tracker = mp.solutions.hands.Hands(
                    static_image_mode=False, max_num_hands=2, model_complexity=0,
                    min_detection_confidence=0.55, min_tracking_confidence=0.5)
    return tracker


def _open_palm(landmarks):
    """Require four extended fingers, regardless of hand rotation or mirror."""
    wrist = np.array([landmarks[0].x, landmarks[0].y])
    extended = 0
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        tip_distance = np.linalg.norm(np.array([landmarks[tip].x, landmarks[tip].y]) - wrist)
        pip_distance = np.linalg.norm(np.array([landmarks[pip].x, landmarks[pip].y]) - wrist)
        extended += tip_distance > pip_distance * 1.18
    return extended == 4


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
    return render_template("index.html")


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
            result = get_tracker().process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        landmarks = result.multi_hand_landmarks or []
        return jsonify({
            "hands": [[[round(point.x, 4), round(point.y, 4)] for point in hand.landmark]
                      for hand in landmarks],
            "finish_gesture": len(landmarks) == 2 and all(
                _open_palm(hand.landmark) for hand in landmarks),
        })
    except Exception:
        app.logger.exception("Hand tracking failed")
        return jsonify({"error": "Hand tracking failed"}), 500


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


if __name__ == "__main__":
    print("Open http://127.0.0.1:8000 in your browser")
    app.run(host="127.0.0.1", port=8000, debug=False)
