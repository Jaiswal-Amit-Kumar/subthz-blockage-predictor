"""Central configuration for the sub-THz blockage predictor."""

from pathlib import Path

# ── Prediction parameters ─────────────────────────────────────────
WINDOW_SIZE = 10                   # Number of KPM frames in sliding window
PREDICTION_HORIZON_MS = 5.0        # Predict blockage within this many ms
BLOCKAGE_THRESHOLD = 0.75          # Probability threshold for triggering action

# ── Model artifact ────────────────────────────────────────────────
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "blockage_lgbm.txt"

# ── Synthetic data generation ─────────────────────────────────────
SYNTHETIC_SAMPLES = 20_000         # Number of sequences to generate
SEQUENCE_LENGTH = WINDOW_SIZE + 1  # Window + one label frame
RANDOM_SEED = 42

# ── API ───────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000