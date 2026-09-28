"""
Evaluate a trained blockage predictor on a held-out dataset.

Computes AUC-ROC, precision, recall, F1, and the confusion matrix.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

from src.model.train import build_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate blockage predictor.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    X, y, _ = build_features(df)

    booster = lgb.Booster(model_file=str(args.model))
    y_prob = booster.predict(X)
    y_pred = (y_prob >= args.threshold).astype(int)

    print("=" * 60)
    print("Evaluation Report")
    print("=" * 60)
    print(f"Model:     {args.model}")
    print(f"Data:      {args.data}  ({len(y):,} samples)")
    print(f"Threshold: {args.threshold}")
    print("-" * 60)

    if len(np.unique(y)) > 1:
        print(f"AUC-ROC:   {roc_auc_score(y, y_prob):.4f}")

        precision, recall, _ = precision_recall_curve(y, y_prob)
        pr_auc = auc(recall, precision)
        print(f"PR-AUC:    {pr_auc:.4f}")

    print(f"Positive rate (ground truth): {y.mean():.4f}")
    print(f"Positive rate (predicted):    {y_pred.mean():.4f}")

    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=["clear", "blockage"]))

    print("Confusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    print(f"  TN={cm[0,0]:6d}  FP={cm[0,1]:6d}")
    print(f"  FN={cm[1,0]:6d}  TP={cm[1,1]:6d}")

    # Feature importance
    importance = booster.feature_importance(importance_type="gain")
    print(f"\nTop 10 features by gain:")
    indices = np.argsort(importance)[::-1][:10]
    for rank, idx in enumerate(indices, 1):
        print(f"  {rank:2d}. feature[{idx:3d}]  gain={importance[idx]:.2f}")


if __name__ == "__main__":
    main()