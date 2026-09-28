"""
Inference engine for sub-THz blockage prediction.

Maintains an internal sliding window of KPM frames and returns a prediction
for each new frame once the window is sufficiently filled.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np

from src.config import (
    WINDOW_SIZE,
    BLOCKAGE_THRESHOLD,
    MODEL_PATH,
)
from src.data.feature_engineer import FeatureEngineer


@dataclass
class PredictionResult:
    """Immutable result of a single prediction invocation."""
    timestamp_ms: float
    blockage_probability: float
    blockage_predicted: bool
    inference_latency_ms: float
    trigger_action: Optional[str] = None


class BlockagePredictor:
    """
    Lightweight, CPU-only blockage predictor.

    Usage
    -----
    >>> predictor = BlockagePredictor()
    >>> predictor.load_model()
    >>> result = predictor.update(kpm_frame)
    >>> if result.blockage_predicted:
    ...     trigger_antenna_switch(result.blockage_probability)
    """

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self._model_path = model_path or MODEL_PATH
        self._booster: Optional[lgb.Booster] = None
        self._feature_engineer = FeatureEngineer(window_size=WINDOW_SIZE)
        self._buffer: list[dict] = []

    def load_model(self) -> None:
        """Load the trained LightGBM model from disk."""
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {self._model_path}. "
                "Run `python -m src.model.train` to generate it."
            )
        with open(self._model_path, "r", encoding="utf-8") as f:
            model_str = f.read()
        self._booster = lgb.Booster(model_str=model_str)

    @property
    def is_ready(self) -> bool:
        return self._booster is not None and len(self._buffer) >= WINDOW_SIZE

    @property
    def buffer_fill(self) -> int:
        return len(self._buffer)

    def update(self, kpm_frame: dict) -> PredictionResult:
        """
        Ingest one KPM frame and return a prediction.

        Parameters
        ----------
        kpm_frame : dict
            Must contain at minimum the feature keys expected by the
            feature engineer, plus ``timestamp_ms``.

        Returns
        -------
        PredictionResult
        """
        t_start = time.perf_counter_ns()

        self._buffer.append(kpm_frame)
        if len(self._buffer) > WINDOW_SIZE:
            self._buffer.pop(0)

        if not self.is_ready:
            return PredictionResult(
                timestamp_ms=kpm_frame.get("timestamp_ms", 0.0),
                blockage_probability=0.0,
                blockage_predicted=False,
                inference_latency_ms=0.0,
            )

        feature_vector = self._feature_engineer.transform(self._buffer)
        prob = float(self._booster.predict(feature_vector.reshape(1, -1))[0])
        predicted = prob >= BLOCKAGE_THRESHOLD

        elapsed_ms = (time.perf_counter_ns() - t_start) / 1e6

        return PredictionResult(
            timestamp_ms=kpm_frame.get("timestamp_ms", 0.0),
            blockage_probability=prob,
            blockage_predicted=predicted,
            inference_latency_ms=elapsed_ms,
            trigger_action="switch_antenna" if predicted else None,
        )