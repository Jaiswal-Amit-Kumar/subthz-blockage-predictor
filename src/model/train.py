"""
Train a LightGBM classifier for sub-THz blockage prediction.

Reads a synthetic or real KPM dataset (long-format CSV with sequence_id,
frame_idx, features, and label columns), constructs sliding-window feature
vectors, and trains a binary classifier.

Usage
-----
    python -m src.model.train --data data/synthetic_train.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

from src.config import MODEL_DIR, MODEL_PATH, WINDOW_SIZE
from src.data.feature_engineer import FeatureEngineer

FEATURE_COLUMNS = [
    "rlc_delay_dl",
    "rlc_drop_rate",
    "harq_retx_ratio",
    "prb_utilization",
    "ue_buffer_occupancy",
    "rssi_dbm",
    "sinr_db",
    "doppler_hz",
]


def build_features(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a long-format KPM DataFrame into sliding-window feature vectors.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``sequence_id``, ``frame_idx``, ``label``,
        and all feature columns.
    window_size : int
        Number of frames per window.

    Returns
    -------
    X : np.ndarray of shape (n_samples, output_dim)
    y : np.ndarray of shape (n_samples,)
    groups : np.ndarray of shape (n_samples,) — sequence IDs for group split
    """
    fe = FeatureEngineer(window_size=window_size)
    X_list, y_list, groups = [], [], []

    for seq_id, group in df.groupby("sequence_id"):
        group = group.sort_values("frame_idx").reset_index(drop=True)
        frames = group[FEATURE_COLUMNS].to_dict("records")
        label = int(group["label"].iloc[-1])

        if len(frames) < window_size:
            continue

        # Use every position as an end-of-window sample
        for end in range(window_size, len(frames) + 1):
            window = frames[end - window_size : end]
            X_list.append(fe.transform(window))
            y_list.append(label)
            groups.append(seq_id)

    X = np.vstack(X_list)
    y = np.asarray(y_list, dtype=np.int64)
    groups = np.asarray(groups)
    return X, y, groups


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train LightGBM blockage classifier."
    )
    parser.add_argument("--data", type=Path, required=True, help="Training CSV.")
    parser.add_argument(
        "--output", type=Path, default=MODEL_PATH, help="Model output path."
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading {args.data} ...")
    df = pd.read_csv(args.data)
    print(f"  Rows: {len(df):,}  Sequences: {df['sequence_id'].nunique():,}")

    X, y, groups = build_features(df)
    print(f"Feature matrix: {X.shape}  Positive rate: {y.mean():.3f}")

    # Group split to prevent sequence leakage
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    train_idx, val_idx = next(gss.split(X, y, groups))

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Handle class imbalance
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)

    print(f"Training LightGBM (scale_pos_weight={scale_pos_weight:.2f}) ...")
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 200,
        "scale_pos_weight": scale_pos_weight,
        "verbosity": -1,
        "seed": args.seed,
    }

    callbacks = [
        lgb.early_stopping(stopping_rounds=20, verbose=True),
        lgb.log_evaluation(period=25),
    ]

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=500,
        valid_sets=[val_set],
        callbacks=callbacks,
    )

    # Evaluation on validation set
    y_prob = booster.predict(X_val, num_iteration=booster.best_iteration)
    y_pred = (y_prob >= 0.5).astype(int)

    print("\nValidation results:")
    print(f"  AUC-ROC: {roc_auc_score(y_val, y_prob):.4f}")
    print(f"  Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(classification_report(y_val, y_pred, target_names=["clear", "blockage"]))

    # Save model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(args.output))
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()