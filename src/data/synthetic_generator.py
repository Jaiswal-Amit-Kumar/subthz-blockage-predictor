"""
Synthetic KPM telemetry generator for sub-THz blockage prediction.

Generates time-series sequences that mimic the signature of an approaching
line-of-sight blockage: gradual RSSI degradation, rising HARQ retransmissions,
increasing RLC delay, and anomalous Doppler signature in the frames preceding
link failure.

Each generated sequence is labeled 1 if a blockage occurs within the prediction
horizon at the end of the window, 0 otherwise.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    SYNTHETIC_SAMPLES,
    SEQUENCE_LENGTH,
    PREDICTION_HORIZON_MS,
    RANDOM_SEED,
)

# Feature names expected by the feature engineer
FEATURE_NAMES = [
    "rlc_delay_dl",
    "rlc_drop_rate",
    "harq_retx_ratio",
    "prb_utilization",
    "ue_buffer_occupancy",
    "rssi_dbm",
    "sinr_db",
    "doppler_hz",
]


def _generate_clear_sequence(rng: np.random.Generator) -> np.ndarray:
    """Generate a sequence with no approaching blockage."""
    n = SEQUENCE_LENGTH
    t = np.arange(n)
    base = np.column_stack([
        rng.normal(1.5, 0.2, n),           # rlc_delay_dl (ms)
        rng.uniform(0.001, 0.01, n),        # rlc_drop_rate
        rng.uniform(0.01, 0.05, n),         # harq_retx_ratio
        rng.uniform(0.6, 0.9, n),           # prb_utilization
        rng.normal(1500, 200, n),           # ue_buffer_occupancy (bytes)
        rng.normal(-80, 1.5, n),            # rssi_dbm
        rng.normal(22, 1.5, n),             # sinr_db
        rng.normal(0, 5, n),                # doppler_hz
    ])
    return base


def _generate_blockage_sequence(rng: np.random.Generator) -> np.ndarray:
    """
    Generate a sequence where a blockage develops in the final frames.

    The last 3 frames exhibit the characteristic degradation signature.
    """
    n = SEQUENCE_LENGTH
    seq = _generate_clear_sequence(rng)

    # Inject degradation in the last 3 frames
    onset = n - 3
    ramp = np.linspace(0, 1, n - onset)

    seq[onset:, 0] += 8.0 * ramp        # rlc_delay_dl surges
    seq[onset:, 1] += 0.25 * ramp       # rlc_drop_rate rises
    seq[onset:, 2] += 0.35 * ramp       # harq_retx_ratio rises
    seq[onset:, 3] -= 0.3 * ramp        # prb_utilization drops
    seq[onset:, 4] += 800 * ramp        # buffer occupancy rises
    seq[onset:, 5] -= 18 * ramp         # rssi_dbm drops sharply
    seq[onset:, 6] -= 15 * ramp         # sinr_db drops
    seq[onset:, 7] += rng.normal(0, 20, n - onset) * ramp  # doppler anomaly

    return seq


def generate_dataset(
    n_samples: int = SYNTHETIC_SAMPLES,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate a balanced synthetic dataset of KPM sequences.

    Returns
    -------
    pd.DataFrame
        Columns: ``sequence_id``, ``frame_idx``, all feature names, and
        ``label`` (1 if blockage within horizon, 0 otherwise).
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for seq_id in range(n_samples):
        # 50/50 class balance
        if rng.random() < 0.5:
            seq = _generate_blockage_sequence(rng)
            label = 1
        else:
            seq = _generate_clear_sequence(rng)
            label = 0

        for frame_idx in range(SEQUENCE_LENGTH):
            row = {
                "sequence_id": seq_id,
                "frame_idx": frame_idx,
                "label": label if frame_idx == SEQUENCE_LENGTH - 1 else 0,
            }
            for j, name in enumerate(FEATURE_NAMES):
                row[name] = float(seq[frame_idx, j])
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic sub-THz blockage KPM dataset."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/synthetic_train.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--samples", type=int, default=SYNTHETIC_SAMPLES,
        help="Number of sequences to generate.",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset(n_samples=args.samples, seed=args.seed)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df):,} rows to {args.output}")
    print(f"Sequences: {df['sequence_id'].nunique():,}")
    print(f"Positive sequences: {df[df['label'] == 1]['sequence_id'].nunique():,}")


if __name__ == "__main__":
    main()