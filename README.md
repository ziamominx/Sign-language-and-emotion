# Zia's Glasses

A Windows webcam prototype for recording and recognizing **isolated ASL signs** from hand landmarks, speaking recognized glosses, and optionally showing facial emotion analysis. It is not a complete ASL interpreter. There is no bundled trained vocabulary, and a gloss is not the same as an ASL sentence.

## What changed

The earlier version mapped finger-up patterns to words such as `HELP` and `WAR`. Those patterns were not validated ASL signs, and nearest-pattern guessing could produce a word for an unknown gesture. The app now compares the **full movement of one or two hands** with examples that you record. It checks a short sequence while you sign and returns a label only when at least three examples exist and repeated matches pass distance and ambiguity checks. Lowering your hands lets you start the next sign. Unknown signs remain unknown. Examples are stored locally under `sign_examples/`; that directory is excluded from Git because recordings may be personal.

The matcher uses MediaPipe's 21 landmarks per hand and normalizes for hand position and size. It does **not** yet model facial grammar, upper-body posture, continuous signing, or signer-independent variations. Its thresholds are conservative starting values, not measured accuracy guarantees. To evaluate reliability, collect examples from multiple signers and hold out test recordings for each label before using it for communication decisions.

## Setup (Windows)

1. Install Python 3.10 and connect a webcam.
2. Install dependencies: `py -3.10 -m pip install -r requirements.txt`.
3. Optional emotion overlay: `py -3.10 -m pip install deepface`. If DeepFace is unavailable, the overlay says so rather than inventing scores.
4. Optional English sentence rewriting: set `ANTHROPIC_API_KEY`. Without it, speech uses the recognized glosses in order. This option sends gloss text to the API; webcam frames and recorded examples stay local.

The program can also install missing core packages on first run. Explicit installation is recommended so errors are visible.

## Build a vocabulary

In the webcam app, press **R**, type a verified ASL sign label such as `HELLO`, and press **Enter**. Make the isolated sign, then lower both hands to save that example. Repeat five times. Recognition becomes available immediately after training; you do not need to restart the app. Press **Esc** during teaching to cancel.

You can also record examples with the separate tool. Record an ASL sign verified with a fluent signer or reliable ASL reference:

```powershell
py -3.10 record_sign.py HELLO --count 5
py -3.10 record_sign.py HELP --count 5
```

In the recording window, make one isolated sign, then lower both hands. Repeat until five examples are saved. Use consistent camera framing and lighting, but vary speed and position naturally. Record at least three examples for every label; five or more are recommended. Collect examples from the intended users. Use a different sign language only with examples from that language and a separate vocabulary directory.

Run `py -3.10 zias_glasses.py` (or `run.bat`). Make a sign and watch for recognition while your hands are visible. Lower your hands between signs to reset, then repeat. Press **R** to teach a sign, **Enter** to speak the current gloss sequence, **C** or the Clear button to reset, and **Q/Esc** to quit. The app also speaks each recognized word. If it prints `[UNKNOWN]`, record better examples or adjust framing; it will not guess a word.

## Scope and next steps

ASL uses movements of the hands and face and has grammar distinct from English. No finite finger-pattern table can cover all ASL. Expanding toward broad communication needs a verified ASL corpus, a model trained on sign videos, signer-independent evaluation, nonmanual features, and continuous-sign segmentation. The [WLASL project](https://github.com/dxli94/WLASL) provides a research dataset with 2,000 word classes and pretrained-model resources, subject to its data agreement. A future model should be evaluated on people absent from training before it replaces the local-example recognizer.

Run core tests with `py -3.10 -m unittest -v test_sign_recognition.py`. Webcam capture, speech, and model accuracy require checks on the target computer with real signers.
