"""
Prepare RAVDESS audio for the 5-emotion task.
Expects the RAVDESS speech archive already extracted under data/raw/ravdess/
or downloads instructions will be printed.
"""

import os
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    RAVDESS_DIR, RAW_DIR, PROCESSED_DIR, RAVDESS_LABEL_MAP,
    SEED, TEST_SPLIT, VALIDATION_SPLIT, MAX_PAD_LEN, N_MFCC
)
from src.utils.common import ensure_dirs, set_seed
from src.utils.audio_utils import extract_mfcc_from_file


def find_ravdess_root() -> Path:
    """Locate the folder that contains Actor_01 ... Actor_24."""
    candidates = [
        RAW_DIR / "ravdess",
        RAW_DIR / "Audio_Speech_Actors_01-24",
        RAVDESS_DIR,
    ]
    for c in candidates:
        if c.exists() and any(c.glob("Actor_*")):
            return c
        # sometimes nested one level deeper
        for sub in c.glob("*"):
            if sub.is_dir() and any(sub.glob("Actor_*")):
                return sub
    return None


def prepare():
    set_seed()
    ensure_dirs()

    root = find_ravdess_root()
    if root is None:
        print("RAVDESS not found.")
        print("\nDownload steps:")
        print("1. Go to https://zenodo.org/record/1188976")
        print("2. Download 'Audio_Speech_Actors_01-24.zip'")
        print("3. Unzip into data/raw/ravdess/  (so you have Actor_01, Actor_02, ...)")
        print("4. Re-run this script.")
        sys.exit(1)

    print(f"Using RAVDESS root: {root}")

    features = []
    labels = []
    paths = []

    wav_files = list(root.rglob("*.wav"))
    print(f"Found {len(wav_files)} wav files")

    for wav in tqdm(wav_files, desc="Extracting MFCC"):
        # Filename pattern: 03-01-05-01-02-01-12.wav
        # indices: modality-vocalchannel-emotion-intensity-statement-repetition-actor
        parts = wav.stem.split("-")
        if len(parts) < 3:
            continue
        emotion_code = parts[2]
        if emotion_code not in RAVDESS_LABEL_MAP:
            continue

        try:
            feat = extract_mfcc_from_file(str(wav))
            features.append(feat)
            labels.append(RAVDESS_LABEL_MAP[emotion_code])
            paths.append(str(wav))
        except Exception as e:
            print(f"Skipping {wav.name}: {e}")

    X = np.stack(features)
    y = np.array(labels, dtype=np.int32)

    print(f"\nUsable samples (5 emotions): {len(X)}")
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  label {u}: {c}")

    # Stratified split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, stratify=y, random_state=SEED
    )
    val_ratio = VALIDATION_SPLIT / (1 - TEST_SPLIT)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=SEED
    )

    out = PROCESSED_DIR / "ravdess_5emo.npz"
    np.savez_compressed(
        out,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
    )
    print(f"\nSaved → {out}")
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")


if __name__ == "__main__":
    prepare()
