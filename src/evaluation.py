"""Evaluation utilities for anomaly detection models.

Owner: Diego. See TASKS_DIEGO.md for the checklist.

Conventions:
- sklearn output: -1 = anomaly, 1 = inlier
- ground truth Machine failure: 1 = failure, 0 = normal

Always convert sklearn labels with `to_binary` before comparing to the target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Default business costs (in euros) — taken from the assessment statement.
FP_COST: float = 500.0
FN_COST: float = 15_000.0


def to_binary(y_pred_sklearn: np.ndarray) -> np.ndarray:
    """Convert sklearn anomaly labels (-1/1) to binary (1=anomaly, 0=normal)."""
    raise NotImplementedError


def confusion_counts(y_true: np.ndarray, y_pred_binary: np.ndarray) -> dict[str, int]:
    """Compute TP, FP, FN, TN for binary anomaly predictions."""
    raise NotImplementedError


def cost_score(
    y_true: np.ndarray,
    y_pred_binary: np.ndarray,
    fp_cost: float = FP_COST,
    fn_cost: float = FN_COST,
) -> dict[str, float]:
    """Compute total business cost + recall + precision."""
    raise NotImplementedError


def compare_models(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    fp_cost: float = FP_COST,
    fn_cost: float = FN_COST,
) -> pd.DataFrame:
    """Build a comparison table across models, sorted by ascending cost."""
    raise NotImplementedError


def disagreement_matrix(predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    """Build a per-row prediction matrix and flag disagreements."""
    raise NotImplementedError


def sensitivity_curve(
    y_true: np.ndarray,
    train_predict_fn,
    contamination_grid: list[float],
    fp_cost: float = FP_COST,
    fn_cost: float = FN_COST,
) -> pd.DataFrame:
    """Re-train a model across a contamination grid and report cost per setting."""
    raise NotImplementedError
