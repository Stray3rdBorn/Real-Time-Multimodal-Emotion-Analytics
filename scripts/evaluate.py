"""
Quick evaluation of trained models on the held-out test sets.
Prints accuracy, classification report and saves confusion matrices.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    PROCESSED_DIR, MODEL_DIR, VIDEO_MODEL_NAME, AUDIO_MODEL_NAME, EMOTIONS
)


def evaluate_model(model_path, data_path, title):
    if not model_path.exists() or not data_path.exists():
        print(f"Skipping {title}: missing model or data")
        return

    model = tf.keras.models.load_model(model_path)
    data = np.load(data_path)
    X_test, y_test = data["X_test"], data["y_test"]

    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    print(f"\n===== {title} =====")
    print(classification_report(y_test, y_pred, target_names=EMOTIONS, digits=3))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title(f"Confusion Matrix – {title}")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    out = MODEL_DIR / f"cm_{title.lower().replace(' ', '_')}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {out}")


def main():
    evaluate_model(
        MODEL_DIR / VIDEO_MODEL_NAME,
        PROCESSED_DIR / "fer2013_5emo.npz",
        "Video (MobileNetV2)"
    )
    evaluate_model(
        MODEL_DIR / AUDIO_MODEL_NAME,
        PROCESSED_DIR / "ravdess_5emo.npz",
        "Audio (BiLSTM)"
    )


if __name__ == "__main__":
    main()
