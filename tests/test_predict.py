"""Unit tests for the prediction engine."""

from src.model.predict import BlockagePredictor
from src.data.feature_engineer import RAW_FEATURES


def _make_frame(i: int) -> dict:
    frame = {name: float(i) for name in RAW_FEATURES}
    frame["timestamp_ms"] = float(i * 5)
    return frame


class TestBlockagePredictor:
    def test_not_ready_before_window_filled(self):
        predictor = BlockagePredictor()
        # Do not load a model — just test buffer logic
        for i in range(5):
            result = predictor.update(_make_frame(i))
            assert result.blockage_predicted is False
            assert result.blockage_probability == 0.0

    def test_buffer_fill_tracks_frames(self):
        predictor = BlockagePredictor()
        for i in range(3):
            predictor.update(_make_frame(i))
        assert predictor.buffer_fill == 3

    def test_buffer_caps_at_window_size(self):
        predictor = BlockagePredictor()
        for i in range(50):
            predictor.update(_make_frame(i))
        assert predictor.buffer_fill == 10  # WINDOW_SIZE