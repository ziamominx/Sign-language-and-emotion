# AI Sign & Expression Arena

**See it. Understand it. Play it.**

A real-time AI platform that watches your webcam, understands hands / face / body
over time, and powers three experiences on one shared perception engine:

| Mode | What it does |
|---|---|
| 🗣️ **AI Communicator** | Signs/gestures → text → voice (controlled vocabulary) |
| 🎭 **AI Dumb Charades** | You act out a movie; the AI observes concepts, ranks candidates, guesses — and **learns from your right/wrong feedback**. Reverse mode: the AI "acts" via gesture cards while your team guesses. |
| 😂 **Meme Challenge** | Perform a challenge; the AI reads the observable expression/pose, scores the act, and picks the matching meme. |

The engine never claims to read minds: it classifies **observable** expressions
("smile-like", "surprise-like") and says so in the UI.

---

## QUICK START

```bash
# 1. Install (Python 3.10 on Windows)
py -3.10 -m pip install -r requirements-web.txt

# 2. Run
py -3.10 server.py

# 3. Open
#    http://127.0.0.1:8000            → Arena home
#    /communicator /charades /memes   → the three modes
#    /lab /history /settings          → AI lab, history, settings
```

Click **Start camera** on any mode and allow camera access. Everything runs
locally: MediaPipe WASM runs in the browser for smooth 30–60 fps overlays, and
all AI decisions run in the local Flask server. No cloud calls, no recording.

---

## ARCHITECTURE

```
                 AI SIGN & EXPRESSION ARENA
                            |
                     CAMERA (browser)
                            |
              MediaPipe Tasks (WASM/GPU, in-page)
             hands 21×2 | face 478 | pose 33
                            |
              rolling landmark window (≤24 frames)
                            |
              POST /api/arena/*  (JSON, local)
                            |
                    PERCEPTION ENGINE  arena/perception.py
              hand_features · face_features · pose_features
              motion_energy · wave/nod/wag/arms_up/facepalm…
                            |
        +-------------------+-------------------+
        |                   |                   |
  classify_hand_gesture  classify_action  classify_expression
        |                   |                   |
        +-------------------+-------------------+
                            |
                  INTELLIGENT AGENTS (arena/)
        CharadesAgent          MemeAgent          (communicator =
        concept accumulation   tag profile        stability + TTS)
        candidate scoring      meme matching
        hypothesis ranking     challenge scoring
        feedback → weight update (SQLite)
                            |
        COMMUNICATOR      CHARADES GAME        MEME GAME
        text + voice      teams/rounds/timer   score + meme card
```

### Key design rules honored

- **No fake AI.** "The AI learns" is a real mechanism: every charades round ends
  with your correct/wrong verdict, which updates per-movie weights in SQLite
  (`arena/movie_weights`, ±8–12% per verdict, clamped 0.2–2.5×). Guesses are
  scored from *evidence over concepts*, never `if college: return "3 Idiots"`.
- **Confidence everywhere.> 0.85 strong · 0.6–0.85 possible · <0.6 ignored.
  The AI says "I'm not sure" and shows alternatives instead of guessing blindly.
- **Real measured FPS/latency** in every HUD — never "zero latency".

---

## DATA (all sample data included, all editable)

| File | Contents |
|---|---|
| `arena/data/gestures.json` | 20 communicator signs with hints + phrases, 14 action gestures, 12 concept gestures |
| `arena/data/movies.json` | 12 recognizable Indian movies with genres, concepts, actions + a `concept_clues` map the agent uses for candidate generation |
| `arena/data/memes.json` | 16 memes with tags/intensity + 8 challenges |

**Add a movie:** append an object to `movies` and (optionally) list it under
`concept_clues` for each concept it shares. The agent picks it up on restart.

**Add a sign / meme:** same idea — append to `gestures.json` or `memes.json`.

---

## THE AGENTS

### CharadesAgent (`arena/charades_agent.py`)

1. **Observe** — perception detections arrive each tick; concepts accumulate
   (confidence-gated at 0.5, decay 0.96/tick, capped at 12).
2. **Generate candidates** — every movie sharing an observed concept is a candidate.
3. **Score** — `weight × (0.5 + evidence)`; evidence sums concept mass plus a
   *concentration bonus* (a concept matching few movies is more informative).
4. **Guess** — only when ≥2 distinct concepts are observed AND the top candidate
   leads by ≥18 points. Wrong guesses are penalized (×0.35 evidence) and the
   agent immediately considers alternatives.
5. **Learn** — your ✓/✗ updates SQLite movie weights that persist across matches.

### MemeAgent (`arena/meme_agent.py`)

1. Face features (eye openness, mouth openness, smile lift, brow height, head tilt)
   → **tag profile** (`eyes_wide 0.9`, `smile 0.8`, …).
2. Pose/hands add tags (`hands_raised`, `facepalm`).
3. Every meme is scored on tag coverage + performed intensity; best match wins.
4. Challenge score = 35% challenge-match + 25% expression + 20% pose + 20% intensity.

---

## THE GAME (`arena/game_engine.py`)

- Two teams, configurable rounds (default 8) and timer (default 60s).
- Correct within 10s = **+2**, otherwise **+1**; wrong/timeout = **+1 to the actor's team**.
- The secret movie is only ever sent to the actor's view (`/charades/start`), and
  the state endpoint masks it during active rounds.
- Mode B (**AI ACTS**) shows gesture cards generated from the movie's concepts —
  clearly labeled as *game communication*, not real sign language.

---

## PAGES

| Route | Page |
|---|---|
| `/` | Home: mode cards, live stats, recent sessions, **Demo Mode** (guided 6-step walkthrough) |
| `/communicator` | Sign → text → voice with stability gate, corrections, voice speed/volume |
| `/charades` | Scoreboard, timer, live camera, **AI Brain panel** (observed concepts + candidate bars), AI guess with ✓/✗, AI actor mode |
| `/memes` | Challenge prompt, live feature chips, capture → score breakdown → meme card |
| `/history` | All sessions, team scores, the agent's actual learned weights, reset button |
| `/settings` | Camera mirror, AI threshold/sensitivity, voice, round length, reduced motion |
| `/lab` | Developer HUD: raw landmark counts, per-classifier confidences, live log |

---

## TESTING

```bash
py -3.10 -m unittest test_arena -v          # 26 arena tests
# + the full site suite (recognition, training, emotion, tracking…):
py -3.10 -m unittest test_sign_recognition.py test_asl_model.py test_alphabet_model.py \
  test_alphabet_api.py test_hand_tracking.py test_pose_tracking.py test_live_landmarks.py \
  test_personal_signs.py test_face_tracking.py test_emotion_api.py test_train_model.py \
  test_expression_choice.py test_training_api.py test_arena.py
```

Covered: hand/face/pose feature extraction, temporal scores, gesture/action/
expression classification, agent evidence ranking, wrong-feedback demotion,
meme matching + challenge scoring, game scoring/timer/team switching,
SQLite weight persistence, and the HTTP endpoints including full page renders.

---

## PRIVACY

- All perception runs locally (browser WASM + local Flask). No cloud APIs.
- No camera frames are stored — only derived landmarks/confidences and game outcomes.
- The camera is active only while you enabled it; **Stop** kills the track instantly.

## KNOWN LIMITATIONS

- Gesture recognition is a controlled vocabulary with rule/geometry-based
  classifiers — not full ASL. It is designed so a trained sequence model
  (LSTM/GRU) can replace `classify_*` later without touching the agents.
- Expression labels describe observable facial geometry, not inner feelings.
- The charades agent learns via weight tables, not neural retraining.

## FUTURE IMPROVEMENTS

- Train the temporal classifier on recorded player data ("Learn My Style" already
  stores per-player examples/corrections in SQLite).
- WebSocket streaming instead of window POSTs.
- Avatar-based AI actor animation instead of gesture cards.
