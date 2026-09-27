"""Arena HTTP API: perception, agents, game state, and sessions.

The browser runs MediaPipe natively (smooth overlays) and posts landmark
windows + events here. All AI decisions happen server-side so the agent
state stays authoritative, and every response is plain JSON.
"""

import random
import threading

from flask import Blueprint, jsonify, request

from arena.charades_agent import AIActorScript, CharadesAgent, load_movies
from arena.game_engine import CharadesGame
from arena.meme_agent import MemeAgent
from arena.perception import (classify_action, classify_expression,
                              classify_hand_gesture, face_features,
                              hand_features, pose_features)
from arena.store import ArenaStore

arena_bp = Blueprint("arena", __name__)
store = ArenaStore()
charades = CharadesAgent(store=store)
memes = MemeAgent()
game = CharadesGame()
game_lock = threading.Lock()
_movies, _clues = load_movies()
actor_script = AIActorScript(_movies, _clues)

# ---------------------------------------------------------------------------
# Perception intake: the browser posts a rolling window of landmark frames.
# ---------------------------------------------------------------------------
_LAST_ANALYSIS = {"gesture": None, "action": None, "expression": None}


def _frames_from_payload(payload):
    """Extract parallel landmark series from the posted window."""
    frames = payload.get("frames") or []
    hands, faces, poses = [], [], []
    for frame in frames:
        hands.append(frame.get("hands") or [])
        faces.append(frame.get("face") or [])
        poses.append(frame.get("pose") or [])
    return hands, faces, poses


def analyze_window(payload):
    """Run perception over one landmark window; returns all classifier outputs."""
    hands, faces, poses = _frames_from_payload(payload)
    latest_hands = hands[-1] if hands else []
    hand = hand_features(latest_hands[0]) if latest_hands else None
    face = face_features(faces[-1]) if faces and faces[-1] else None
    pose = pose_features(poses[-1]) if poses and poses[-1] else None
    result = {
        "hand": hand,
        "face": face,
        "pose": pose,
        "gesture": classify_hand_gesture(
            [h[0] for h in hands if h], faces, poses),
        "action": classify_action(
            [h[0] for h in hands if h], poses, faces),
        "expression": classify_expression(faces,
                                          [h[0] for h in hands if h], poses),
    }
    _LAST_ANALYSIS.update({key: result[key] for key in _LAST_ANALYSIS})
    return result


# ---------------------------------------------------------------------------
# Perception endpoints
# ---------------------------------------------------------------------------

@arena_bp.post("/api/arena/perceive")
def perceive():
    payload = request.get_json(silent=True) or {}
    result = analyze_window(payload)
    return jsonify({"ok": True, "analysis": result})


# ---------------------------------------------------------------------------
# Communicator endpoints
# ---------------------------------------------------------------------------

@arena_bp.post("/api/arena/communicator/observe")
def communicator_observe():
    payload = request.get_json(silent=True) or {}
    result = analyze_window(payload)
    gesture = result.get("gesture")
    if gesture:
        store.record_player_example(payload.get("player", "player_1"),
                                    gesture["label"],
                                    [round(gesture["confidence"], 2)])
    return jsonify({"ok": True, "gesture": gesture,
                    "hand": result.get("hand")})


@arena_bp.post("/api/arena/communicator/correct")
def communicator_correct():
    payload = request.get_json(silent=True) or {}
    actual = str(payload.get("actual", "")).strip()
    if not actual:
        return jsonify({"error": "Send the actual label"}), 422
    store.record_correction(payload.get("player", "player_1"),
                            str(payload.get("guessed", "")), actual)
    return jsonify({"ok": True, "learned": actual})


# ---------------------------------------------------------------------------
# Charades endpoints
# ---------------------------------------------------------------------------

@arena_bp.get("/api/arena/charades/state")
def charades_state():
    return jsonify({"ok": True, "game": game.state(masked=False),
                    "agent": charades.state()})


@arena_bp.post("/api/arena/charades/start")
def charades_start():
    payload = request.get_json(silent=True) or {}
    actor_team = payload.get("actor_team") or random.choice(game.teams)
    movie = payload.get("movie")
    if not movie:
        pool = [title for title in _movies if title not in charades.rejected]
        movie = random.choice(pool or list(_movies))
    try:
        with game_lock:
            state = game.start_round(actor_team, movie)
            charades.reset_round()
    except ValueError as error:
        return jsonify({"error": str(error)}), 409
    store.add_session("charades", "round_started", {"movie": movie})
    return jsonify({"ok": True, "game": state,
                    "secret_movie": movie,
                    "ai_script": actor_script.script_for(movie)})


@arena_bp.post("/api/arena/charades/observe")
def charades_observe():
    payload = request.get_json(silent=True) or {}
    result = analyze_window(payload)
    detections = []
    if result.get("gesture"):
        detections.append({"label": result["gesture"]["label"],
                           "confidence": result["gesture"]["confidence"],
                           "kind": "gesture"})
    if result.get("action"):
        detections.append({"label": result["action"]["label"],
                           "confidence": result["action"]["confidence"],
                           "kind": "action"})
    if result.get("face") and result.get("expression"):
        detections.append({"label": result["expression"]["label"],
                           "confidence": result["expression"]["confidence"],
                           "kind": "expression"})
    charades.observe(detections)
    if game.expired():
        with game_lock:
            game.resolve_timeout()
    state = game.state(masked=False)
    guessing_by_ai = state["round_active"]
    response = {"ok": True, "game": state, "agent": charades.state(),
                "detections": detections, "analysis": result,
                "ai_should_guess": guessing_by_ai and charades.should_guess()}
    if guessing_by_ai and charades.should_guess() and not charades.last_guess:
        response["ai_guess"] = charades.guess()
    return jsonify(response)


@arena_bp.post("/api/arena/charades/feedback")
def charades_feedback():
    payload = request.get_json(silent=True) or {}
    correct = bool(payload.get("correct"))
    if not game.round_active:
        return jsonify({"error": "No active round"}), 409
    learning = charades.feedback(correct)
    elapsed = game.rules["round_seconds"] - game.time_remaining()
    with game_lock:
        if correct:
            game.resolve_correct(elapsed)
        else:
            game.resolve_wrong()
    outcome = "correct" if correct else "wrong"
    store.add_session("charades", outcome,
                      {"movie": game.last_result.get("movie") if game.last_result else None,
                       "elapsed": round(elapsed, 1)})
    return jsonify({"ok": True, "game": game.state(masked=False),
                    "learning": learning})


@arena_bp.post("/api/arena/charades/timeout")
def charades_timeout():
    if not game.round_active:
        return jsonify({"error": "No active round"}), 409
    with game_lock:
        game.resolve_timeout()
    store.add_session("charades", "timeout", {})
    return jsonify({"ok": True, "game": game.state(masked=False)})


@arena_bp.post("/api/arena/charades/reset")
def charades_reset():
    with game_lock:
        game.round_active = False
        game.round_number = 0
        game.rounds_played = 0
        for team in game.teams:
            game.scores[team] = 0
            game.streak[team] = 0
        game.last_result = None
        charades.reset_round()
    store.reset_scores()
    return jsonify({"ok": True, "game": game.state(masked=False)})


# ---------------------------------------------------------------------------
# Meme challenge endpoints
# ---------------------------------------------------------------------------

@arena_bp.get("/api/arena/memes/challenge")
def meme_challenge():
    challenge = memes.current_challenge()
    return jsonify({"ok": True, "challenge": challenge,
                    "index": memes.challenge_index})


@arena_bp.post("/api/arena/memes/next")
def meme_next():
    challenge = memes.next_challenge()
    return jsonify({"ok": True, "challenge": challenge})


@arena_bp.post("/api/arena/memes/analyze")
def meme_analyze():
    payload = request.get_json(silent=True) or {}
    result = analyze_window(payload)
    report = memes.analyze(result.get("expression"), result.get("face"),
                           result.get("action"))
    store.add_session("meme", "attempt",
                      {"challenge": report["challenge"]["id"],
                       "score": report["score"]["final"]})
    return jsonify({"ok": True, **report, "analysis": result})


# ---------------------------------------------------------------------------
# Home, history, learning status
# ---------------------------------------------------------------------------

@arena_bp.get("/api/arena/status")
def arena_status():
    sessions = store.recent_sessions(limit=8)
    return jsonify({"ok": True, "sessions": sessions,
                    "accuracy": store.accuracy(),
                    "movies": len(_movies),
                    "memes": len(memes.memes),
                    "challenges": len(memes.challenges),
                    "movie_weights": store.movie_weights()})


@arena_bp.get("/api/arena/history")
def arena_history():
    return jsonify({"ok": True, "sessions": store.recent_sessions(limit=40),
                    "accuracy": store.accuracy(),
                    "teams": {team: store.team_score(team) for team in game.teams}})


@arena_bp.get("/api/arena/learning")
def arena_learning():
    return jsonify({"ok": True,
                    "movie_weights": store.movie_weights(),
                    "players": store.player_profile("player_1")})


@arena_bp.post("/api/arena/learning/reset")
def arena_learning_reset():
    store.reset_learning()
    return jsonify({"ok": True})
