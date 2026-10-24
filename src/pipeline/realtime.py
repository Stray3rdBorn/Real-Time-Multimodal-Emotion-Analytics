"""
Real-time multimodal emotion detection demo.
Webcam (MediaPipe face) + microphone → late fusion with temporal smoothing.
"""

import sys
import time
from collections import deque
from pathlib import Path
from threading import Thread, Lock

import cv2
import numpy as np
import pyaudio
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (
    MODEL_DIR, VIDEO_MODEL_NAME, AUDIO_MODEL_NAME,
    CAMERA_INDEX, SAMPLE_RATE, AUDIO_CHUNK_SECONDS,
    N_MFCC, MAX_PAD_LEN, EMOTIONS
)
from src.utils.video_utils import detect_faces_mediapipe, preprocess_face, draw_emotion_overlay
from src.utils.audio_utils import extract_mfcc, normalize_audio
from src.pipeline.fusion import EmotionFusion


class AudioRecorder:
    """Non-blocking audio buffer that keeps the latest N seconds."""

    def __init__(self, rate=SAMPLE_RATE, chunk_seconds=AUDIO_CHUNK_SECONDS):
        self.rate = rate
        self.chunk_samples = int(rate * chunk_seconds)
        self.buffer = deque(maxlen=self.chunk_samples)
        self.lock = Lock()
        self.running = False
        self.pa = pyaudio.PyAudio()
        self.stream = None

    def start(self):
        self.running = True
        self.stream = self.pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=1024,
            stream_callback=self._callback,
        )
        self.stream.start_stream()

    def _callback(self, in_data, frame_count, time_info, status):
        data = np.frombuffer(in_data, dtype=np.float32)
        with self.lock:
            self.buffer.extend(data)
        return (None, pyaudio.paContinue)

    def get_audio(self) -> np.ndarray:
        with self.lock:
            if len(self.buffer) < self.rate * 0.5:  # need at least 0.5 s
                return None
            return np.array(self.buffer, dtype=np.float32)

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pa.terminate()


def load_models():
    video_path = MODEL_DIR / VIDEO_MODEL_NAME
    audio_path = MODEL_DIR / AUDIO_MODEL_NAME

    if not video_path.exists() or not audio_path.exists():
        print("Models not found. Train them first:")
        print("  python scripts/train_video.py")
        print("  python scripts/train_audio.py")
        sys.exit(1)

    print("Loading models ...")
    video_model = tf.keras.models.load_model(video_path)
    audio_model = tf.keras.models.load_model(audio_path)
    return video_model, audio_model


def run_demo():
    video_model, audio_model = load_models()
    fusion = EmotionFusion()
    recorder = AudioRecorder()
    recorder.start()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Cannot open camera")
        sys.exit(1)

    print("\n=== Real-time Emotion Analytics ===")
    print("Press 'q' to quit, 'r' to reset fusion history\n")

    last_audio_pred = None
    last_video_pred = None
    fps_time = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # mirror
            boxes = detect_faces_mediapipe(frame)

            video_probs = None
            if boxes:
                face_img = preprocess_face(frame, boxes[0])
                face_batch = np.expand_dims(face_img, axis=0)
                video_probs = video_model.predict(face_batch, verbose=0)[0]
                last_video_pred = video_probs

            # Audio prediction every few frames
            if frame_count % 8 == 0:
                audio_data = recorder.get_audio()
                if audio_data is not None:
                    audio_data = normalize_audio(audio_data)
                    mfcc = extract_mfcc(audio_data)
                    mfcc_batch = np.expand_dims(mfcc, axis=0)
                    audio_probs = audio_model.predict(mfcc_batch, verbose=0)[0]
                    last_audio_pred = audio_probs

            # Fuse
            label, conf, _ = fusion.fuse(last_video_pred, last_audio_pred)

            # Draw
            if boxes:
                draw_emotion_overlay(frame, boxes[0], label, conf)
            else:
                cv2.putText(frame, f"{label} ({conf:.0%}) [no face]",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

            # FPS
            frame_count += 1
            if frame_count % 15 == 0:
                now = time.time()
                fps = 15 / (now - fps_time)
                fps_time = now
                cv2.putText(frame, f"FPS: {fps:.1f}", (20, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("Emotion Analytics (q=quit, r=reset)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                fusion.reset()
                print("Fusion history reset")

    finally:
        recorder.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("Demo stopped.")


if __name__ == "__main__":
    run_demo()
