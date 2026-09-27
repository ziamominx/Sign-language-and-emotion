# Zia's Glasses — ASL to voice

A local website that uses a pretrained model to recognize words from a **2,000 word isolated ASL vocabulary**. No sign recording or training is required. The camera analyzes short rolling clips continuously, adds confident words to a message, and can speak them aloud. The earlier Windows webcam prototype remains in `zias_glasses.py` for reference; the website is the primary interface.

## Run the website

On Windows, install Python 3.10, then:

```powershell
py -3.10 -m pip install -r requirements-web.txt
py -3.10 server.py
```

Open **http://127.0.0.1:8000** in a browser. You can also run `run_web.bat`. On first launch, the server downloads the 15 MB [SignBart WLASL-2000 checkpoint](https://huggingface.co/tinh2312/SignBart-WLASL-2000) and its label list to `.models/`. The model is checked against pinned SHA-256 hashes and cached for later launches. PyTorch is a separate, larger install. Internet is only needed for installation and the initial model download; camera frames go to the local server at `127.0.0.1`.

Click **Start live translation**, allow camera access, and frame your upper body and hands. Sign one isolated ASL word at a time, pausing between words. The live guess appears over the camera view; confident results enter the message automatically and are spoken when **Speak recognized words aloud** is checked. Tap a candidate to correct or manually add a word. **Pause translation**, **Stop camera**, **Undo last**, **Clear**, and **Speak message** are available. The server needs to stay running while the page is open.

The camera now draws tracked hand joints and finger lines over the live view. To finish, hold **both hands open for 1.5 seconds** after at least one word is in the message. A progress indicator appears, then the website shows and speaks the whole collected message. The finish gesture does not add a sign and does not paraphrase or invent words. Lower either hand before using the gesture again. The **Speak message** button remains available as a fallback.

### Teach your own signs from coordinates

Start the camera in **ASL words** mode, enter a label in **Teach a sign**, and click **Record example**. After a short get-ready pause, the website saves the next 18 frames of hand and body coordinates. Record the same sign at least **three times**, with slightly different speed and position. Once the saved list says **ready**, test it by signing again in the normal live camera view. A confident match from saved examples takes priority over the pretrained vocabulary and enters the message with the same stability check and optional speech. Use **Remove** to delete a sign and its examples.

This is a fast personal coordinate-template learner, not a new universal ASL model. It uses normalized hand shapes, hand location relative to the shoulders, and movement over time, then compares live sequences with the examples you recorded. It can recognize only the personal signs you teach; record clear, distinct examples and test for false matches. The coordinate dataset is stored locally in `.models/personal_signs.json`, excluded from Git, and loaded automatically on restart. The website sends camera frames and coordinates only to its local server.

The hand overlay follows the earlier desktop app's mirrored camera and 480-pixel MediaPipe tracking frame. The fast hand loop also supplies landmarks for word recognition, while a separate body-pose loop supplies the remaining landmarks. The recognizer classifies a rolling landmark sequence instead of reprocessing a JPEG batch for every guess. The hand lines keep their independent update loop, and the face lines run separately. A Face Mesh loop draws the face outline, eyes, brows, lips, nose, and feature dots, matching the older app's marking style. **Show facial expression estimate** uses the optional DeepFace model from the older desktop project. It checks for a real face before analysis and displays a small estimated-expression label above the tracked face, refreshing as results arrive. The higher-resolution face sample and full expression-score distribution help reveal a tentative happy or sad expression when it nearly ties neutral; a question mark marks such a tentative result. On this computer the model is already installed and cached. To enable it elsewhere, run `py -3.10 -m pip install -r requirements-emotion.txt`; the first run may download DeepFace's expression weights. If unavailable, the website disables the expression control. The label remains an uncertain visual estimate of facial expression, not a measure of someone's actual emotion. Camera frames and expression results remain on the local server.

The site checks consecutive predictions before automatically adding a word. A different stable word can follow immediately; repeating the same word requires lowering the hands briefly. If hands or the upper body are lost, it shows specific framing guidance. The former image-batch route remains available at `/api/live` for compatibility, but the website uses the faster landmark route.

### Optional fingerspelling

The **Fingerspell · experimental** mode uses a separate 28-label ASL alphabet checkpoint from the user's local reference project. It does not require training. On this computer the checkpoint is already cached at `.models/asl_mediapipe_mlp_model.h5`. On another computer, place an existing `asl_mediapipe_mlp_model.h5` there or set the `ASL_ALPHABET_MODEL_PATH` environment variable to its location, then restart the server. If the file is absent, the website keeps the mode disabled while ASL words remains available. The checkpoint is deliberately excluded from Git. The reference model was trained on images from the [Kaggle ASL Alphabet dataset](https://www.kaggle.com/grassknoted/asl-alphabet/data), which lists a GPL 2 license; this repository does not redistribute those images or its checkpoint.

Select **Fingerspell · experimental**, show one hand, and hold each letter until it appears in the message. Lower the hand briefly before repeating a letter. Use **Finish spelled word** to separate and speak the completed word, or **Delete letter** to correct it. The J and Z signs involve motion and are not added automatically. Letter suggestions are experimental: the supplied reference model was trained on static images and has no reliable idle/no-hand class. The website requires a visible hand and repeated high-scoring predictions before adding a letter. Review every spelled word.

## Scope and accuracy

This is a research prototype, not a complete or certified ASL interpreter. The [SignBart project](https://github.com/TinhNguyen2312/SignBart) reports **68% accuracy on its WLASL-2000 evaluation split**; that result is not a guarantee for a live webcam or an individual signer. The model only classifies short isolated clips among its 2,000 labels. It cannot understand continuous ASL, facial grammar, signs outside the vocabulary, or translate full ASL sentences into English. The displayed percentages are model scores, not verified probabilities of correctness. Automatic additions may be wrong, so review the message before important communication. Initial analysis needs a rolling sequence of about one second of tracked landmarks before the first guess; network, tracking, and stability checks add delay.

The [National Institute on Deafness and Other Communication Disorders](https://www.nidcd.nih.gov/health/american-sign-language) describes ASL as a complete language expressed through movements of the hands and face, with grammar distinct from English. The message builder preserves selected words in order; it does not claim to translate ASL grammar.

## Development checks

Run `py -3.10 -m unittest -v test_sign_recognition.py test_asl_model.py test_alphabet_model.py test_alphabet_api.py test_hand_tracking.py test_pose_tracking.py test_live_landmarks.py test_personal_signs.py test_face_tracking.py test_emotion_api.py` and `node --test test_sign_stability.js test_letter_stability.js test_finish_gesture.js test_hand_geometry.js` for the recognizer, personal-sign storage, model adapters, tracking, expression API, overlay geometry, and temporal gates. For a live check, open the site, grant camera access, record a personal sign three times, sign it again, then hold both hands open to speak the message. Camera and speech access depend on the browser and target device.
