"""
Audio / speech prosody model.
MFCC + delta + delta-delta features → Bidirectional LSTM → Dense classifier.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Input

from src.config import MAX_PAD_LEN, N_MFCC, EMOTIONS


def build_audio_model(
    input_shape=(MAX_PAD_LEN, N_MFCC * 3),   # 40 MFCC + delta + delta2
    num_classes: int = len(EMOTIONS),
    learning_rate: float = 1e-4,
) -> Model:
    """
    Bidirectional LSTM model that captures temporal dynamics of speech prosody.
    """
    inputs = Input(shape=input_shape, name="mfcc_input")

    x = layers.Masking(mask_value=0.0)(inputs)          # ignore padding
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.2)
    )(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="emotion")(x)

    model = Model(inputs, outputs, name="audio_bilstm")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
