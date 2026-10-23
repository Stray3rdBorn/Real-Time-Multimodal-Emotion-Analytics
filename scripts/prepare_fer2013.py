"""
Download & prepare FER-2013 for the 5-emotion task.
Requires either:
  - Kaggle API credentials (kaggle.json), or
  - Manual download of fer2013.csv into data/raw/
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    FER_DIR, RAW_DIR, PROCESSED_DIR, FER_LABEL_MAP,
    IMG_SIZE, SEED, TEST_SPLIT, VALIDATION_SPLIT
)
from src.utils.common import ensure_dirs, set_seed


def download_fer2013():
    """Try Kaggle download; fall back to instruction if credentials missing."""
    ensure_dirs()
    csv_path = RAW_DIR / "fer2013.csv"
    if csv_path.exists():
        print(f"[OK] Found existing {csv_path}")
        return csv_path

    print("Attempting Kaggle download of FER-2013 ...")
    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "msambare/fer2013", path=str(RAW_DIR), unzip=True
        )
        # Kaggle usually places fer2013.csv in the folder
        if (RAW_DIR / "fer2013.csv").exists():
            return RAW_DIR / "fer2013.csv"
        # sometimes nested
        for p in RAW_DIR.rglob("fer2013.csv"):
            return p
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        print("\nManual steps:")
        print("1. Go to https://www.kaggle.com/datasets/msambare/fer2013")
        print("2. Download fer2013.csv")
        print(f"3. Place it at: {csv_path}")
        sys.exit(1)

    raise FileNotFoundError("Could not locate fer2013.csv after download")


def pixels_to_image(pixel_str: str) -> np.ndarray:
    """Convert the space-separated pixel string into a (48,48) array."""
    pixels = np.array(pixel_str.split(), dtype=np.uint8)
    return pixels.reshape(IMG_SIZE, IMG_SIZE)


def prepare():
    set_seed()
    ensure_dirs()
    csv_path = download_fer2013()

    print("Loading CSV ...")
    df = pd.read_csv(csv_path)

    # Keep only the 5 emotions we care about
    df = df[df["emotion"].isin(FER_LABEL_MAP.keys())].copy()
    df["label"] = df["emotion"].map(FER_LABEL_MAP)

    print(f"Samples after filtering to 5 emotions: {len(df)}")
    print(df["label"].value_counts().sort_index())

    images = []
    labels = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Converting pixels"):
        img = pixels_to_image(row["pixels"])
        # Replicate to 3 channels
        img = np.stack([img] * 3, axis=-1).astype(np.float32) / 255.0
        images.append(img)
        labels.append(row["label"])

    X = np.stack(images)
    y = np.array(labels, dtype=np.int32)

    # Stratified split → train / val / test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, stratify=y, random_state=SEED
    )
    val_ratio = VALIDATION_SPLIT / (1 - TEST_SPLIT)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=SEED
    )

    out = PROCESSED_DIR / "fer2013_5emo.npz"
    np.savez_compressed(
        out,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
    )
    print(f"\nSaved processed data → {out}")
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")


if __name__ == "__main__":
    prepare()
