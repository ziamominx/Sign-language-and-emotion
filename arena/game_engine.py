"""Charades game state: teams, rounds, timers, and scoring rules.

The engine is pure state + validation — the HTTP layer feeds it events
(start round, guess feedback, timeout) and renders `state()` for the UI.
"""

import time

DEFAULT_RULES = {
    "round_seconds": 60,
    "fast_bonus_seconds": 10,
    "fast_bonus_points": 2,
    "normal_points": 1,
    "wrong_gives_opponent": 1,
    "total_rounds": 8,
}


class CharadesGame:
    def __init__(self, teams=("TEAM A", "TEAM B"), rules=None):
        self.teams = list(teams)
        self.rules = dict(DEFAULT_RULES)
        if rules:
            self.rules.update(rules)
        self.scores = {team: 0 for team in self.teams}
        self.rounds_played = 0
        self.round_active = False
        self.actor_team = None
        self.guessing_team = None
        self.secret_movie = None
        self.round_started_at = None
        self.round_number = 0
        self.streak = {team: 0 for team in self.teams}
        self.last_result = None

    # ------------------------------------------------------------------
    def start_round(self, actor_team, movie):
        if actor_team not in self.scores:
            raise ValueError("Unknown team")
        if self.round_active:
            raise ValueError("A round is already running")
        self.round_active = True
        self.round_number += 1
        self.actor_team = actor_team
        self.guessing_team = self._other(actor_team)
        self.secret_movie = movie
        self.round_started_at = time.time()
        self.last_result = None
        return self.state(masked=True)

    def time_remaining(self):
        if not self.round_active or self.round_started_at is None:
            return self.rules["round_seconds"]
        elapsed = time.time() - self.round_started_at
        return max(0, round(self.rules["round_seconds"] - elapsed))

    def expired(self):
        return self.round_active and self.time_remaining() <= 0

    # ------------------------------------------------------------------
    def resolve_correct(self, elapsed_seconds=None):
        if not self.round_active:
            raise ValueError("No active round")
        points = self.rules["normal_points"]
        elapsed = elapsed_seconds if elapsed_seconds is not None else \
            self.rules["round_seconds"] - self.time_remaining()
        if elapsed <= self.rules["fast_bonus_seconds"]:
            points = self.rules["fast_bonus_points"]
        winner = self.guessing_team
        self.scores[winner] += points
        self.streak[winner] += 1
        self._close(f"correct", {"winner": winner, "points": points,
                                 "movie": self.secret_movie, "elapsed": round(elapsed, 1)})

    def resolve_wrong(self):
        if not self.round_active:
            raise ValueError("No active round")
        winner = self.actor_team
        self.scores[winner] += self.rules["wrong_gives_opponent"]
        self.streak[winner] += 1
        self.streak[self.guessing_team] = 0
        self._close("wrong", {"winner": winner, "movie": self.secret_movie})

    def resolve_timeout(self):
        if not self.round_active:
            raise ValueError("No active round")
        winner = self.actor_team
        self.scores[winner] += self.rules["wrong_gives_opponent"]
        self.streak[winner] += 1
        self._close("timeout", {"winner": winner, "movie": self.secret_movie})

    def _close(self, outcome, detail):
        self.rounds_played += 1
        self.round_active = False
        self.round_started_at = None
        self.last_result = {"outcome": outcome, **detail}

    def _other(self, team):
        return next(t for t in self.teams if t != team)

    def finished(self):
        return self.rounds_played >= self.rules["total_rounds"]

    def masked_state(self):
        return self.state(masked=True)

    def state(self, masked=False):
        return {
            "teams": [{"name": team, "score": self.scores[team],
                       "streak": self.streak[team]} for team in self.teams],
            "round": self.round_number,
            "round_active": self.round_active,
            "actor_team": self.actor_team,
            "guessing_team": self.guessing_team,
            "time_remaining": self.time_remaining(),
            "total_rounds": self.rules["total_rounds"],
            "finished": self.finished(),
            "last_result": self.last_result,
            "secret_movie": None if masked else self.secret_movie,
        }
