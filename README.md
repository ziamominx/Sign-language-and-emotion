# Zia's Glasses

## dev/creator: Zia (ziamo)

## What is this project?
Zia's Glasses is a real-time sign language to voice system. It uses your webcam to recognize specific hand gestures (signs) and translates them into spoken English sentences. The project is designed to help bridge communication between sign language users and others by providing instant voice output for a set of defined signs.

### Key Features
- Recognizes 6 main hand signs (USE, TECHNOLOGY, FOR, HELP, NOT, WAR) and several extras
- Hold each sign for 1.5 seconds to lock and speak the word
- Raise both hands open for 2 seconds to speak the full interpreted sentence
- Real-time webcam-based hand and face tracking
- Emotion detection overlay
- Voice output using text-to-speech

## How to fork and run
1. **Fork the repository**
   - Click the "Fork" button on the top right of the GitHub page to create your own copy.
2. **Clone your fork**
   - Open a terminal and run:
     ```
     git clone https://github.com/YOUR-USERNAME/zias-glasses.git
     cd zias-glasses
     ```
3. **Install Python 3.8+**
   - Make sure you have Python 3.8 or newer installed.
4. **Install dependencies**
   - The script will auto-install required packages on first run, but you can also run:
     ```
     pip install opencv-python==4.9.0.80 mediapipe==0.10.14 numpy pyttsx3 requests deepface
     ```
5. **Run the program**
   - Start the app with:
     ```
     python zias_glasses.py
     ```
   - Make sure your webcam is connected and accessible.

## Relevant Links
- [Mediapipe documentation](https://google.github.io/mediapipe/solutions/hands.html)
- [OpenCV documentation](https://docs.opencv.org/)
- [pyttsx3 documentation](https://pyttsx3.readthedocs.io/)
- [DeepFace documentation](https://github.com/serengil/deepface)

---

This project is for educational and assistive technology purposes. For questions or contributions, fork and submit a pull request.
