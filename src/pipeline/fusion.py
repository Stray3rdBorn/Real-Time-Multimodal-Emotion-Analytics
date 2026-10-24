"""
Late fusion with weighted probability average + temporal smoothing.
"""

from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np

from src.config import (
    EMOTIONS,
    FUSION_WEIGHTS,
    TEMPORAL_WINDOW,
    CONFIDENCE_THRESHOLD,
    IDX_TO_EMOTION,
)


class EmotionFusion:
    """
    Maintains a short history of predictions and produces a smoothed final label.
    """

    def __init__(
        self,
        video_weight: float = FUSION_WEIGHTS[0],
        audio_weight: float = FUSION_WEIGHTS[1],
        window: int = TEMPORAL_WINDOW,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ):
        assert abs(video_weight + audio_weight - 1.0) < 1e-5, "Weights must sum to 1"
        self.w_v = video_weight
        self.w_a = audio_weight
        self.window = window
        self.conf_thresh = confidence_threshold
        self.history: Deque[np.ndarray] = deque(maxlen=window)

    def fuse(
        self,
        video_probs: Optional[np.ndarray],
        audio_probs: Optional[np.ndarray],
    ) -> Tuple[str, float, np.ndarray]:
        """
        Parameters
        ----------
        video_probs : (5,) softmax or None if face missing
        audio_probs : (5,) softmax or None if audio silent / failed

        Returns
        -------
        label : str
        confidence : float
        fused_probs : (5,) array
        """
        if video_probs is None and audio_probs is None:
            return "Uncertain", 0.0, np.zeros(len(EMOTIONS))

        if video_probs is None:
            fused = audio_probs
        elif audio_probs is None:
            fused = video_probs
        else:
            # Weighted average of probability vectors
            fused = self.w_v * video_probs + self.w_a * audio_probs

        # Temporal smoothing
        self.history.append(fused)
        if len(self.history) > 1:
            smoothed = np.mean(list(self.history), axis=0)
        else:
            smoothed = fused

        idx = int(np.argmax(smoothed))
        conf = float(smoothed[idx])

        if conf < self.conf_thresh:
            label = "Uncertain"
        else:
            label = IDX_TO_EMOTION[idx]

        return label, conf, smoothed

    def reset(self) -> None:
        self.history.clear()
