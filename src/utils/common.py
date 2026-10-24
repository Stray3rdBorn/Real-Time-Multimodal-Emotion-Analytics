"""Common helpers used across the project."""

import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.config import SEED, ROOT, DATA_DIR, MODEL_DIR, PROCESSED_DIR


def set_seed(seed: int = SEED) -> None:
    """Make runs as reproducible as possible."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dirs() -> None:
    """Create all required directories if they do not exist."""
    for d in [DATA_DIR, MODEL_DIR, PROCESSED_DIR,
              DATA_DIR / "fer2013", DATA_DIR / "ravdess",
              DATA_DIR / "raw", DATA_DIR / "processed"]:
        Path(d).mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    return ROOT
