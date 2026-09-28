"""Unit tests for the feature engineering module."""

import numpy as np
import pytest

from src.data.feature_engineer import FeatureEngineer, RAW_FEATURES


def _make_frames(n: int) -> list[dict]:
    return [
        {name: float(i) for name in RAW_FEATURES}
        for i in range(n)
    ]


class TestFeatureEngineer:
    def test_output_shape(self):
        fe = FeatureEngineer(window_size=10)
        frames = _make_frames(10)
        vec = fe.transform(frames)
        assert vec.shape == (fe.output_dim,)
        assert vec.shape == (len(RAW_FEATURES) * 10 + 5,)

    def test_insufficient_frames_raises(self):
        fe = FeatureEngineer(window_size=10)
        with pytest.raises(ValueError, match="Need at least"):
            fe.transform(_make_frames(5))

    def test_uses_most_recent_frames(self):
        fe = FeatureEngineer(window_size=3)
        frames = _make_frames(10)
        vec = fe.transform(frames)
        # The first raw feature of the window should be from frame 7
        # (frames 7, 8, 9 are used; frame 7 has value 7.0)
        assert vec[0] == pytest.approx(7.0)

    def test_zero_variance_returns_zero_correlation(self):
        fe = FeatureEngineer(window_size=5)
        frames = [{name: 1.0 for name in RAW_FEATURES} for _ in range(5)]
        vec = fe.transform(frames)
        assert vec[-1] == 0.0  # cross-correlation is the last derived feature