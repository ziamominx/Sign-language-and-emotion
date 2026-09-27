# Zia's Glasses — ASL to voice

A local website that uses a pretrained model to suggest words from a **2,000 word isolated ASL vocabulary**. No sign recording or training is required. Review a suggestion, add it to a message, and use browser speech output. The earlier Windows webcam prototype remains in `zias_glasses.py` for reference; the website is the primary interface.

## Run the website

On Windows, install Python 3.10, then:

```powershell
py -3.10 -m pip install -r requirements-web.txt
py -3.10 server.py
```

Open **http://127.0.0.1:8000** in a browser. You can also run `run_web.bat`. On first launch, the server downloads the 15 MB [SignBart WLASL-2000 checkpoint](https://huggingface.co/tinh2312/SignBart-WLASL-2000) and its label list to `.models/`. The model is checked against pinned SHA-256 hashes and cached for later launches. PyTorch is a separate, larger install. Internet is only needed for installation and the initial model download; camera frames go to the local server at `127.0.0.1`.

Click **Enable camera**, frame your head, shoulders, and hands, then **Capture one sign**. Perform one isolated ASL sign during the 2.5 second capture. Choose the intended word from the three suggestions, capture more signs, and click **Speak message**. Use **Undo last** or **Clear** to correct the message. The server needs to stay running while the page is open.

## Scope and accuracy

This is a research prototype, not a complete or certified ASL interpreter. The [SignBart project](https://github.com/TinhNguyen2312/SignBart) reports **68% accuracy on its WLASL-2000 evaluation split**; that result is not a guarantee for a live webcam or an individual signer. The model only classifies short isolated clips among its 2,000 labels. It cannot understand continuous ASL, facial grammar, signs outside the vocabulary, or translate full ASL sentences into English. The displayed percentages are model scores, not verified probabilities of correctness. The site asks you to choose the word before speaking it to reduce harmful miscommunication.

The [National Institute on Deafness and Other Communication Disorders](https://www.nidcd.nih.gov/health/american-sign-language) describes ASL as a complete language expressed through movements of the hands and face, with grammar distinct from English. The message builder preserves selected words in order; it does not claim to translate ASL grammar.

## Development checks

Run `py -3.10 -m unittest -v test_sign_recognition.py test_asl_model.py` for the recognizer and model adapter checks. For a live check, open the site, grant camera access, capture a known sign, and review the suggestions. Camera and speech access depend on the browser and target device.
