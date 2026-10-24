"""Audio feature extraction – improved MFCC pipeline."""

from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

from src.config import SAMPLE_RATE, N_MFCC, MAX_PAD_LEN, N_FFT, HOP_LENGTH


def load_audio(path: str, sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """Load audio file and resample to target sample rate."""
    y, orig_sr = librosa.load(path, sr=sr, mono=True)
    return y, sr


def extract_mfcc(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    max_pad_len: int = MAX_PAD_LEN,
) -> np.ndarray:
    """
    Extract MFCC + delta + delta-delta features and pad/truncate to fixed length.
    Returns shape (max_pad_len, n_mfcc * 3)
    """
    # Base MFCCs
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    # First and second derivatives (prosody dynamics)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Stack → (n_mfcc*3, time)
    features = np.vstack([mfcc, delta, delta2])

    # Transpose to (time, features)
    features = features.T

    # Pad or truncate
    if features.shape[0] < max_pad_len:
        pad_width = max_pad_len - features.shape[0]
        features = np.pad(features, ((0, pad_width), (0, 0)), mode="constant")
    else:
        features = features[:max_pad_len, :]

    return features.astype(np.float32)


def extract_mfcc_from_file(path: str) -> np.ndarray:
    """Convenience wrapper: load file → extract features."""
    y, sr = load_audio(path)
    return extract_mfcc(y, sr)


def normalize_audio(y: np.ndarray) -> np.ndarray:
    """Simple peak normalization."""
    peak = np.max(np.abs(y)) + 1e-8
    return y / peak
