"""
Central configuration for the Real-Time Multimodal Emotion Analytics system.
"""

from pathlib import Path

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FER_DIR = DATA_DIR / "fer2013"
RAVDESS_DIR = DATA_DIR / "ravdess"
MODEL_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "assets"

# ---------------------------------------------------------------------
# Emotions
# ---------------------------------------------------------------------
EMOTIONS = ["Angry", "Fear", "Happy", "Sad", "Neutral"]
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}
IDX_TO_EMOTION = {i: e for i, e in enumerate(EMOTIONS)}

# FER-2013 original labels → our 5-class mapping
# 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
FER_LABEL_MAP = {
    0: 0,  # Angry
    2: 1,  # Fear
    3: 2,  # Happy
    4: 3,  # Sad
    6: 4,  # Neutral
    # Disgust (1) and Surprise (5) are dropped
}

# RAVDESS emotion codes → our labels
# 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
RAVDESS_LABEL_MAP = {
    "01": 4,  # Neutral
    "03": 2,  # Happy
    "04": 3,  # Sad
    "05": 0,  # Angry
    "06": 1,  # Fear
    # calm, disgust, surprised dropped
}

# ---------------------------------------------------------------------
# Video / Face settings
# ---------------------------------------------------------------------
IMG_SIZE = 48
IMG_CHANNELS = 3               # MobileNetV2 expects 3 channels (grayscale is replicated)
FACE_CONFIDENCE = 0.6          # MediaPipe min detection confidence
MAX_FACES = 1

# ---------------------------------------------------------------------
# Audio settings
# ---------------------------------------------------------------------
SAMPLE_RATE = 22050
N_MFCC = 40
MAX_PAD_LEN = 130
N_FFT = 2048
HOP_LENGTH = 512
AUDIO_DURATION = 3.0           # seconds used for training clips

# ---------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS_VIDEO = 40
EPOCHS_AUDIO = 50
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42

# ---------------------------------------------------------------------
# Fusion & Real-time
# ---------------------------------------------------------------------
# Weighted late fusion (video weight, audio weight)
FUSION_WEIGHTS = (0.45, 0.55)  # Slight preference for audio (speech is more reliable in many cases)
TEMPORAL_WINDOW = 7            # Moving average over last N predictions
CONFIDENCE_THRESHOLD = 0.35    # Below this → "Uncertain"

# Real-time loop
CAMERA_INDEX = 0
AUDIO_CHUNK_SECONDS = 1.5
TARGET_FPS = 15

# ---------------------------------------------------------------------
# Model filenames
# ---------------------------------------------------------------------
VIDEO_MODEL_NAME = "video_mobilenetv2_48.keras"
AUDIO_MODEL_NAME = "audio_lstm_mfcc.keras"
HISTORY_VIDEO = "history_video.json"
HISTORY_AUDIO = "history_audio.json"
