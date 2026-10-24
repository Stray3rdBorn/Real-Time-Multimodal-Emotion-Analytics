"""Video / face utilities – MediaPipe detection + 48×48 preprocessing."""

from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from src.config import IMG_SIZE, FACE_CONFIDENCE, MAX_FACES


# MediaPipe Face Detection (lightweight & robust)
_mp_face = mp.solutions.face_detection
_face_detector = _mp_face.FaceDetection(
    model_selection=0,          # 0 = short-range (webcam)
    min_detection_confidence=FACE_CONFIDENCE
)


def detect_faces_mediapipe(
    frame_bgr: np.ndarray,
) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces and return bounding boxes (x, y, w, h) in original frame coordinates.
    Returns empty list if no face found.
    """
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = _face_detector.process(rgb)

    boxes = []
    if results.detections:
        for det in results.detections[:MAX_FACES]:
            bbox = det.location_data.relative_bounding_box
            x = max(0, int(bbox.xmin * w))
            y = max(0, int(bbox.ymin * h))
            bw = min(w - x, int(bbox.width * w))
            bh = min(h - y, int(bbox.height * h))
            if bw > 20 and bh > 20:          # ignore tiny detections
                boxes.append((x, y, bw, bh))
    return boxes


def preprocess_face(
    frame_bgr: np.ndarray,
    box: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """
    Crop face (if box given), convert to grayscale, resize to 48×48,
    replicate to 3 channels for MobileNetV2, normalize to [0, 1].
    Returns shape (48, 48, 3)
    """
    if box is not None:
        x, y, w, h = box
        # Expand box a little for context
        margin = int(0.15 * max(w, h))
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame_bgr.shape[1], x + w + margin)
        y2 = min(frame_bgr.shape[0], y + h + margin)
        face = frame_bgr[y1:y2, x1:x2]
    else:
        face = frame_bgr

    if face.size == 0:
        # Fallback black image
        face = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    # Replicate grayscale → 3 channels (MobileNetV2 expects RGB)
    rgb = np.stack([resized] * 3, axis=-1).astype(np.float32) / 255.0
    return rgb


def draw_emotion_overlay(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    label: str,
    confidence: float,
    color: Tuple[int, int, int] = (0, 255, 120),
) -> np.ndarray:
    """Draw bounding box + label + confidence on the frame."""
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    text = f"{label} ({confidence:.0%})"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x, y - th - 10), (x + tw + 6, y), color, -1)
    cv2.putText(frame, text, (x + 3, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return frame
