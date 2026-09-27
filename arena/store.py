"""SQLite persistence for the arena: weights, players, sessions, scores.

Everything is local (arena.sqlite3 next to this module). Camera frames are
never stored — only derived features, decisions, and game outcomes.
"""

import sqlite3
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "arena.sqlite3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS movie_weights (
    title TEXT PRIMARY KEY,
    weight REAL NOT NULL DEFAULT 1.0,
    correct INTEGER NOT NULL DEFAULT 0,
    wrong INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS player_profiles (
    player TEXT NOT NULL,
    gesture TEXT NOT NULL,
    examples INTEGER NOT NULL DEFAULT 0,
    features TEXT NOT NULL DEFAULT '[]',
    corrections INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (player, gesture)
);
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mode TEXT NOT NULL,
    outcome TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS team_scores (
    team TEXT PRIMARY KEY,
    score INTEGER NOT NULL DEFAULT 0,
    rounds INTEGER NOT NULL DEFAULT 0
);
"""


class ArenaStore:
    def __init__(self, path=DB_PATH):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.executescript(SCHEMA)
        self.connection.commit()

    def close(self):
        """Release the SQLite file handle (matters on Windows)."""
        with self.lock:
            self.connection.close()

    # ------------------------------------------------------------------
    # Movie weight learning
    # ------------------------------------------------------------------
    def movie_weights(self):
        rows = self.connection.execute(
            "SELECT title, weight FROM movie_weights WHERE weight != 1.0").fetchall()
        return {title: weight for title, weight in rows}

    def apply_weight_deltas(self, deltas):
        with self.lock:
            now = time.time()
            for title, delta in deltas.items():
                self.connection.execute(
                    """INSERT INTO movie_weights (title, weight, correct, wrong, updated_at)
                       VALUES (?, MAX(0.2, MIN(2.5, 1.0 + ?)), ?, ?, ?)
                       ON CONFLICT(title) DO UPDATE SET
                         weight = MAX(0.2, MIN(2.5, weight + ?)),
                         correct = correct + ?,
                         wrong = wrong + ?,
                         updated_at = ?""",
                    (title, delta, 1 if delta > 0 else 0,
                     1 if delta < 0 else 0, now, delta,
                     1 if delta > 0 else 0, 1 if delta < 0 else 0, now))
            self.connection.commit()

    def reset_learning(self):
        with self.lock:
            self.connection.execute("DELETE FROM movie_weights")
            self.connection.execute("DELETE FROM player_profiles")
            self.connection.commit()

    # ------------------------------------------------------------------
    # Player style learning ("Learn My Style")
    # ------------------------------------------------------------------
    def record_player_example(self, player, gesture, features):
        with self.lock:
            now = time.time()
            row = self.connection.execute(
                "SELECT examples, features FROM player_profiles WHERE player=? AND gesture=?",
                (player, gesture)).fetchone()
            if row:
                examples, stored = row[0] + 1, json_loads_merge(row[1], features)
            else:
                examples, stored = 1, list(features)
            self.connection.execute(
                """INSERT INTO player_profiles (player, gesture, examples, features, corrections, updated_at)
                   VALUES (?, ?, ?, ?, 0, ?)
                   ON CONFLICT(player, gesture) DO UPDATE SET
                     examples=?, features=?, updated_at=?""",
                (player, gesture, examples, dump(stored), now, examples, dump(stored), now))
            self.connection.commit()
            return examples

    def record_correction(self, player, gesture, actual_label):
        """The player corrected the AI: store the true label association."""
        with self.lock:
            self.connection.execute(
                """INSERT INTO player_profiles (player, gesture, examples, features, corrections, updated_at)
                   VALUES (?, ?, 1, '[]', 1, ?)
                   ON CONFLICT(player, gesture) DO UPDATE SET
                     corrections = corrections + 1, updated_at = ?""",
                (player, actual_label or gesture, time.time(), time.time()))
            self.connection.commit()

    def player_profile(self, player):
        rows = self.connection.execute(
            "SELECT gesture, examples, corrections FROM player_profiles WHERE player=?",
            (player,)).fetchall()
        return [{"gesture": gesture, "examples": examples, "corrections": corrections}
                for gesture, examples, corrections in rows]

    # ------------------------------------------------------------------
    # Sessions and scores
    # ------------------------------------------------------------------
    def add_session(self, mode, outcome, detail=None):
        with self.lock:
            cursor = self.connection.execute(
                "INSERT INTO sessions (mode, outcome, detail, created_at) VALUES (?, ?, ?, ?)",
                (mode, outcome, dump(detail or {}), time.time()))
            self.connection.commit()
            return cursor.lastrowid

    def recent_sessions(self, limit=20):
        rows = self.connection.execute(
            "SELECT mode, outcome, detail, created_at FROM sessions ORDER BY id DESC LIMIT ?",
            (limit,)).fetchall()
        return [{"mode": mode, "outcome": outcome, "detail": load(detail),
                 "created_at": created} for mode, outcome, detail, created in rows]

    def accuracy(self):
        row = self.connection.execute(
            "SELECT SUM(CASE WHEN outcome='correct' THEN 1 ELSE 0 END), COUNT(*) "
            "FROM sessions WHERE mode='charades'").fetchone()
        if not row or not row[1]:
            return None
        return round(row[0] / row[1], 3)

    def team_score(self, team):
        row = self.connection.execute(
            "SELECT score, rounds FROM team_scores WHERE team=?", (team,)).fetchone()
        if row:
            return {"score": row[0], "rounds": row[1]}
        return {"score": 0, "rounds": 0}

    def award_points(self, team, points):
        with self.lock:
            self.connection.execute(
                """INSERT INTO team_scores (team, score, rounds) VALUES (?, ?, 1)
                   ON CONFLICT(team) DO UPDATE SET score = score + ?, rounds = rounds + 1""",
                (team, points, points))
            self.connection.commit()

    def reset_scores(self):
        with self.lock:
            self.connection.execute("DELETE FROM team_scores")
            self.connection.commit()


def dump(value):
    import json
    return json.dumps(value)


def load(value):
    import json
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return {}


def json_loads_merge(existing_json, features):
    existing = load(existing_json)
    if not isinstance(existing, list):
        existing = []
    return existing[-19:] + list(features)
