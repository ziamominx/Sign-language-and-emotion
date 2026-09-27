"""Local website for pretrained, isolated ASL word recognition."""

from pathlib import Path
from threading import Lock
from time import perf_counter

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, render_template, request

from asl_model import ASLRecognizer


ROOT = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(ROOT / "web" / "templates"),
            static_folder=str(ROOT / "web" / "static"))
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024
recognizer = None
recognizer_lock = Lock()


def get_recognizer():
    global recognizer
    if recognizer is None:
        with recognizer_lock:
            if recognizer is None:
                recognizer = ASLRecognizer()
    return recognizer


def extract_keypoints(files):
    frames = []
    hand_frames = 0
    pose_frames = 0
    with mp.solutions.holistic.Holistic(
            static_image_mode=False, model_complexity=0,
            min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for upload in files:
            data = np.frombuffer(upload.read(), np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image is None or image.shape[0] < 120 or image.shape[1] < 120:
                raise ValueError("The camera frames could not be read")
            image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
            result = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            points = np.zeros((75, 2), dtype=np.float32)
            if result.pose_landmarks:
                pose_frames += 1
                points[:33] = [(p.x, p.y) for p in result.pose_landmarks.landmark]
            if result.left_hand_landmarks:
                hand_frames += 1
                points[33:54] = [(p.x, p.y) for p in result.left_hand_landmarks.landmark]
            if result.right_hand_landmarks:
                hand_frames += 1 if not result.left_hand_landmarks else 0
                points[54:75] = [(p.x, p.y) for p in result.right_hand_landmarks.landmark]
            frames.append(points)
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
    try:
        points = extract_keypoints(files)
        predictions = get_recognizer().predict(points)
    except ValueError as exc:
        if str(exc) == "Keep your upper body and signing hands visible in the camera":
            return jsonify({"visible": False, "suggestions": [], "uncertain": True,
                            "processing_ms": round((perf_counter() - started) * 1000)})
        return jsonify({"error": str(exc)}), 422
    except Exception:
        app.logger.exception("Live recognition failed")
        return jsonify({"error": "Recognition failed; please try again"}), 500
    uncertain = predictions[0]["score"] < 0.45 or (
        predictions[0]["score"] - predictions[1]["score"] < 0.10)
    return jsonify({"visible": True, "suggestions": predictions, "uncertain": uncertain,
                    "frames": len(points),
                    "processing_ms": round((perf_counter() - started) * 1000)})


if __name__ == "__main__":
    print("Open http://127.0.0.1:8000 in your browser")
    app.run(host="127.0.0.1", port=8000, debug=False)
