"""
Video / facial emotion model.
MobileNetV2 backbone (ImageNet pretrained) + custom classification head.
Input shape: 48×48×3 (grayscale channel is replicated).
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import MobileNetV2

from src.config import IMG_SIZE, IMG_CHANNELS, EMOTIONS


def build_video_model(
    input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS),
    num_classes: int = len(EMOTIONS),
    trainable_backbone: bool = False,
    learning_rate: float = 1e-4,
) -> Model:
    """
    Build a MobileNetV2-based classifier.

    - Backbone is frozen by default (good for small datasets / fast training).
    - Set trainable_backbone=True later for fine-tuning if you have enough data.
    """
    inputs = Input(shape=input_shape, name="face_input")

    # MobileNetV2 expects values in [-1, 1] when using preprocess_input,
    # but we already normalize to [0,1]. We apply the official preprocess.
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)

    backbone = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    backbone.trainable = trainable_backbone

    x = backbone(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="emotion")(x)

    model = Model(inputs, outputs, name="video_mobilenetv2")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def unfreeze_backbone(model: Model, from_layer: int = 100) -> None:
    """Unfreeze the last part of the backbone for fine-tuning."""
    backbone = model.get_layer("mobilenetv2_1.00_224")  # default name
    backbone.trainable = True
    for layer in backbone.layers[:from_layer]:
        layer.trainable = False
