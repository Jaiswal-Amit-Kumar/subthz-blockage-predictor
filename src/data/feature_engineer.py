"""
Feature extraction from sliding windows of KPM telemetry frames.

Transforms a sequence of frames into a fixed-length numeric vector suitable
for LightGBM inference. Computes both raw per-frame features and derived
time-series statistics capturing the dynamic signature of approaching blockage.
"""

from __future__ import annotations

import numpy as np

# Feature keys expected in each KPM frame dict
RAW_FEATURES = [
    "rlc_delay_dl",
    "rlc_drop_rate",
    "harq_retx_ratio",
    "prb_utilization",
    "ue_buffer_occupancy",
    "rssi_dbm",
    "sinr_db",
    "doppler_hz",
]

# Number of derived time-series features appended to the flat raw vector
N_DERIVED_FEATURES = 5


class FeatureEngineer:
    """
    Transforms a sequence of KPM frames into a flat feature vector.

    Parameters
    ----------
    window_size : int
        Number of most recent frames to use. Frames older than this are dropped.
    """

    def __init__(self, window_size: int = 10) -> None:
        self.window_size = window_size

    @property
    def output_dim(self) -> int:
        return len(RAW_FEATURES) * self.window_size + N_DERIVED_FEATURES

    def transform(self, frames: list[dict]) -> np.ndarray:
        """
        Parameters
        ----------
        frames : list[dict]
            KPM frames ordered oldest to newest. Must contain at least
            ``window_size`` frames.

        Returns
        -------
        np.ndarray
            1-D float64 feature vector of length ``output_dim``.
        """
        if len(frames) < self.window_size:
            raise ValueError(
                f"Need at least {self.window_size} frames, got {len(frames)}."
            )

        window = frames[-self.window_size :]

        # Raw feature matrix: (window_size, n_raw_features)
        raw = np.array(
            [[float(f.get(k, 0.0)) for k in RAW_FEATURES] for f in window],
            dtype=np.float64,
        )

        # Derived time-series features
        rssi_idx = RAW_FEATURES.index("rssi_dbm")
        sinr_idx = RAW_FEATURES.index("sinr_db")
        delay_idx = RAW_FEATURES.index("rlc_delay_dl")
        retx_idx = RAW_FEATURES.index("harq_retx_ratio")

        rssi_seq = raw[:, rssi_idx]
        sinr_seq = raw[:, sinr_idx]

        derived = np.array(
            [
                # Mean first difference (rate of degradation)
                float(np.mean(np.diff(rssi_seq))) if len(rssi_seq) > 1 else 0.0,
                float(np.mean(np.diff(sinr_seq))) if len(sinr_seq) > 1 else 0.0,
                # Volatility
                float(np.std(rssi_seq)),
                float(np.std(sinr_seq)),
                # Cross-correlation between delay and retransmission
                self._safe_correlation(raw[:, delay_idx], raw[:, retx_idx]),
            ],
            dtype=np.float64,
        )

        return np.concatenate([raw.flatten(), derived])

    @staticmethod
    def _safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
        """Pearson correlation with guard against zero-variance inputs."""
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])