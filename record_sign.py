"""Record examples of an isolated sign using the same webcam convention as the app."""

import argparse
import time

import cv2
import mediapipe as mp

from sign_recognition import SignLibrary


def main():
    parser = argparse.ArgumentParser(description="Record ASL sign examples locally")
    parser.add_argument("label", help="Gloss or word to associate with the sign")
    parser.add_argument("--count", type=int, default=5, help="Examples to record (default: 5)")
    args = parser.parse_args()
    if args.count < 3:
        parser.error("Record at least three examples")
    library = SignLibrary()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam")
    clip = []
    last_hand = 0.0
    saved = 0
    try:
        with mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                      min_detection_confidence=0.6,
                                      min_tracking_confidence=0.55) as hands:
            while saved < args.count:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                landmarks = result.multi_hand_landmarks or []
                if landmarks:
                    clip.append([[[point.x, point.y] for point in hand.landmark]
                                 for hand in landmarks[:2]])
                    clip = clip[-90:]
                    last_hand = time.monotonic()
                elif clip and time.monotonic() - last_hand >= 0.3:
                    try:
                        path = library.add(args.label, clip)
                        saved += 1
                        print(f"Saved {saved}/{args.count}: {path}")
                    except ValueError as exc:
                        print(f"Try again: {exc}")
                    clip = []
                status = f"{args.label.upper()}: {saved}/{args.count} saved | sign, then lower hands | Q quit"
                cv2.putText(frame, status, (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("Record sign", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"Recorded {saved} example(s) for {args.label.upper()}")


if __name__ == "__main__":
    main()
