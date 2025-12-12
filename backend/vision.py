"""Vision pipeline: detection, tracking, and emotion recognition."""
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .tracker import CentroidTracker

# Try to load FER; fallback to stub if unavailable
try:
    from fer import FER  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    FER = None


class EmotionRecognizer:
    """Emotion recognizer with graceful fallback."""

    def __init__(self):
        if FER is not None:
            try:
                self.detector = FER(mtcnn=False)
            except Exception:
                self.detector = None
        else:
            self.detector = None

    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """Return (emotion, confidence)."""
        if self.detector is None:
            return "neutral", 0.0
        try:
            result = self.detector.detect_emotions(image)
            if result:
                emotions = result[0]["emotions"]
                emotion = max(emotions, key=emotions.get)
                return emotion, float(emotions[emotion])
        except Exception:
            pass
        return "neutral", 0.0


class VisionSystem:
    """Coordinate camera capture, detection, tracking, and emotion."""

    def __init__(self, camera_index: int = 0, emotion_every: int = 5):
        self.camera_index = camera_index
        self.emotion_every = emotion_every
        self.cap = cv2.VideoCapture(camera_index)
        self.detector = self._init_mediapipe_detector()
        self.tracker = CentroidTracker()
        self.emotion = EmotionRecognizer()
        self.latest_data: List[Dict] = []
        self.frame_count = 0
        self.logs_path = Path("logs/emotions.jsonl")
        self.logs_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _init_mediapipe_detector():
        try:
            import mediapipe as mp

            return mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5
            )
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(f"Cannot import MediaPipe: {exc}") from exc

    def is_open(self) -> bool:
        return self.cap.isOpened()

    def _detect_faces(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)
        detections = results.detections or []
        return detections

    def _bbox_from_detection(self, detection, frame_shape: Tuple[int, int, int]):
        image_h, image_w, _ = frame_shape
        bbox = detection.location_data.relative_bounding_box
        x = max(int(bbox.xmin * image_w), 0)
        y = max(int(bbox.ymin * image_h), 0)
        w = int(bbox.width * image_w)
        h = int(bbox.height * image_h)
        return (x, y, w, h)

    def _log_emotion(self, person_id: int, emotion: str, confidence: float):
        entry = {
            "ts": time.time(),
            "person_id": int(person_id),
            "emotion": emotion,
            "confidence": float(confidence),
        }
        with self.logs_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def process_frame(self) -> Tuple[bytes, List[Dict]]:
        """Capture frame, run detection/tracking/emotion, return JPEG bytes and data."""
        if not self.is_open():
            raise RuntimeError(
                f"Cannot open camera at index {self.camera_index}. Update CAMERA_INDEX in README."
            )
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise RuntimeError("Failed to read frame from camera")

        detections = self._detect_faces(frame)
        boxes = [self._bbox_from_detection(det, frame.shape) for det in detections]

        tracked = self.tracker.update(boxes)
        self.frame_count += 1

        results: List[Dict] = []
        for person_id, (x, y, w, h) in tracked.items():
            # Safety to avoid negative sizes
            x, y = max(x, 0), max(y, 0)
            w, h = max(w, 1), max(h, 1)
            emotion, conf = "neutral", 0.0
            if self.frame_count % self.emotion_every == 0:
                face_roi = frame[y : y + h, x : x + w]
                if face_roi.size > 0:
                    emotion, conf = self.emotion.predict(face_roi)
                    self._log_emotion(person_id, emotion, conf)
            results.append(
                {
                    "person_id": int(person_id),
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "emotion": emotion,
                    "confidence": float(conf),
                }
            )
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"ID {person_id}: {emotion} ({conf:.2f})"
            cv2.putText(
                frame,
                label,
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        self.latest_data = results
        _, buffer = cv2.imencode(".jpg", frame)
        return buffer.tobytes(), results

    def get_latest_data(self) -> List[Dict]:
        if not self.latest_data:
            # Warm up one frame if nothing processed yet
            try:
                self.process_frame()
            except RuntimeError:
                pass
        return self.latest_data

    def shutdown(self):
        if self.cap:
            self.cap.release()
