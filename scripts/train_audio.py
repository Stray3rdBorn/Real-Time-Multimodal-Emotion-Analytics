"""
Train the BiLSTM audio model on prepared RAVDESS (5 emotions).
"""

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    PROCESSED_DIR, MODEL_DIR, AUDIO_MODEL_NAME, HISTORY_AUDIO,
    BATCH_SIZE, EPOCHS_AUDIO, LEARNING_RATE, SEED
)
from src.models.audio_model import build_audio_model
from src.utils.common import ensure_dirs, set_seed


def main():
    set_seed()
    ensure_dirs()

    data_path = PROCESSED_DIR / "ravdess_5emo.npz"
    if not data_path.exists():
        print(f"Processed data not found: {data_path}")
        print("Run: python scripts/prepare_ravdess.py first")
        sys.exit(1)

    print("Loading data ...")
    data = np.load(data_path)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    model = build_audio_model(learning_rate=LEARNING_RATE)
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
        ModelCheckpoint(
            str(MODEL_DIR / AUDIO_MODEL_NAME),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        TensorBoard(log_dir=str(MODEL_DIR / "logs_audio")),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS_AUDIO,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}")

    hist = {k: [float(x) for x in v] for k, v in history.history.items()}
    hist["test_accuracy"] = float(test_acc)
    with open(MODEL_DIR / HISTORY_AUDIO, "w") as f:
        json.dump(hist, f, indent=2)

    print(f"Model saved → {MODEL_DIR / AUDIO_MODEL_NAME}")


if __name__ == "__main__":
    main()
