"""Facial emotion detection with OpenCV.

Faces are located with OpenCV's Haar cascade and each face is classified by the
pretrained FER+ network (ONNX) running on OpenCV's DNN module.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODEL_URL = (
    "https://github.com/onnx/models/raw/main/validated/vision/"
    "body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
)
MODEL_PATH = Path(__file__).parent / "models" / "emotion-ferplus-8.onnx"
MIN_MODEL_BYTES = 30_000_000

EMOTIONS = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
]

EMOTION_COLORS = {
    "neutral": (200, 200, 200),
    "happiness": (0, 215, 255),
    "surprise": (255, 191, 0),
    "sadness": (255, 120, 30),
    "anger": (40, 40, 230),
    "disgust": (60, 160, 60),
    "fear": (200, 80, 160),
    "contempt": (120, 120, 180),
}


@dataclass
class FaceEmotion:
    box: tuple[int, int, int, int]
    emotion: str
    confidence: float
    scores: dict[str, float]


def ensure_model(path: Path = MODEL_PATH) -> Path:
    if path.exists() and path.stat().st_size >= MIN_MODEL_BYTES:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".part")
    print(f"Downloading emotion model to {path} ...")
    urllib.request.urlretrieve(MODEL_URL, tmp_path)
    if tmp_path.stat().st_size < MIN_MODEL_BYTES:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError("Emotion model download is incomplete. Please try again.")
    tmp_path.replace(path)
    return path


class EmotionDetector:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.net = cv2.dnn.readNetFromONNX(str(ensure_model(model_path)))
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(str(cascade_path))
        if self.face_cascade.empty():
            raise RuntimeError(f"Could not load face cascade from {cascade_path}")

    def detect_faces(self, gray: np.ndarray) -> list[tuple[int, int, int, int]]:
        min_side = max(30, min(gray.shape[:2]) // 12)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_side, min_side)
        )
        return [tuple(int(v) for v in face) for face in faces]

    def classify_face(self, gray_face: np.ndarray) -> dict[str, float]:
        face = cv2.resize(gray_face, (64, 64)).astype(np.float32)
        blob = cv2.dnn.blobFromImage(face)
        self.net.setInput(blob)
        logits = self.net.forward().flatten()
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        return {label: float(p) for label, p in zip(EMOTIONS, probs)}

    def predict(self, image_bgr: np.ndarray) -> list[FaceEmotion]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        results = []
        for x, y, w, h in self.detect_faces(gray):
            pad = int(0.1 * w)
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
            scores = self.classify_face(gray[y0:y1, x0:x1])
            emotion = max(scores, key=scores.get)
            results.append(FaceEmotion((x, y, w, h), emotion, scores[emotion], scores))
        return results


def annotate(image_bgr: np.ndarray, results: list[FaceEmotion]) -> np.ndarray:
    output = image_bgr.copy()
    thickness = max(2, min(output.shape[:2]) // 300)
    font_scale = max(0.5, min(output.shape[:2]) / 900)
    for result in results:
        x, y, w, h = result.box
        color = EMOTION_COLORS.get(result.emotion, (0, 255, 0))
        label = f"{result.emotion} {result.confidence * 100:.0f}%"
        cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        label_y = y - 8 if y - th - 12 > 0 else y + h + th + 8
        cv2.rectangle(
            output,
            (x, label_y - th - 6),
            (x + tw + 8, label_y + baseline - 2),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            output,
            label,
            (x + 4, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (20, 20, 20),
            thickness,
            cv2.LINE_AA,
        )
    return output


if __name__ == "__main__":
    ensure_model()
    print("Emotion model is ready.")
