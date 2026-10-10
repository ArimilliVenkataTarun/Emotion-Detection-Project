"""Real-time emotion detection from your webcam. Press 'q' to quit."""

import cv2

from emotion_detector import EmotionDetector, annotate


def main(camera_index: int = 0) -> None:
    detector = EmotionDetector()
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise SystemExit(f"Could not open camera {camera_index}.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            cv2.imshow("Emotion Detection (press q to quit)", annotate(frame, detector.predict(frame)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
