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
    """Convert sklearn anomaly labels (-1/1) to binary (1=anomaly, 0=normal).

    Args:
        y_pred_sklearn: Array of -1 (anomaly) and 1 (inlier).

    Returns:
        Array of 0 (normal) and 1 (anomaly), matching the convention of
        the `Machine failure` column.
    """
    arr = np.asarray(y_pred_sklearn)
    return (arr == -1).astype(int)


def confusion_counts(y_true: np.ndarray, y_pred_binary: np.ndarray) -> dict[str, int]:
    """Compute TP, FP, FN, TN for binary anomaly predictions.

    Args:
        y_true: Ground-truth binary labels (1 = failure / anomaly).
        y_pred_binary: Predicted binary labels (1 = anomaly).

    Returns:
        Dict with keys tp, fp, fn, tn (all ints).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred_binary = np.asarray(y_pred_binary).astype(int)
    tp = int(((y_pred_binary == 1) & (y_true == 1)).sum())
    fp = int(((y_pred_binary == 1) & (y_true == 0)).sum())
    fn = int(((y_pred_binary == 0) & (y_true == 1)).sum())
    tn = int(((y_pred_binary == 0) & (y_true == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def cost_score(
    y_true: np.ndarray,
    y_pred_binary: np.ndarray,
    fp_cost: float = FP_COST,
    fn_cost: float = FN_COST,
) -> dict[str, float]:
    """Compute total business cost + recall + precision.

    Cost = FP * fp_cost + FN * fn_cost.

    Args:
        y_true: Ground-truth binary labels.
        y_pred_binary: Predicted binary labels.
        fp_cost: Cost per false positive (default €500).
        fn_cost: Cost per false negative (default €15 000).

    Returns:
        Dict containing tp, fp, fn, tn, cost_eur, recall, precision.
    """
    counts = confusion_counts(y_true, y_pred_binary)
    total_cost = counts["fp"] * fp_cost + counts["fn"] * fn_cost
    recall_denom = max(counts["tp"] + counts["fn"], 1)
    precision_denom = max(counts["tp"] + counts["fp"], 1)
    recall = counts["tp"] / recall_denom
    precision = counts["tp"] / precision_denom
    return {
        **counts,
        "cost_eur": float(total_cost),
        "recall": float(recall),
        "precision": float(precision),
    }


def compare_models(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    fp_cost: float = FP_COST,
    fn_cost: float = FN_COST,
) -> pd.DataFrame:
    """Build a comparison table across models, sorted by ascending cost.

    Predictions can be either sklearn-style (-1/1) or already binary (0/1):
    they are normalised internally with ``to_binary`` if any -1 is detected.

    Args:
        y_true: Ground-truth binary labels.
        predictions: Mapping model name -> prediction array.
        fp_cost: Cost per false positive.
        fn_cost: Cost per false negative.

    Returns:
        DataFrame with one row per model, columns:
        model, tp, fp, fn, tn, cost_eur, recall, precision.
    """
    rows = []
    for name, pred in predictions.items():
        pred = np.asarray(pred)
        if (pred == -1).any():
            pred = to_binary(pred)
        scores = cost_score(y_true, pred, fp_cost=fp_cost, fn_cost=fn_cost)
        rows.append({"model": name, **scores})
    return (
        pd.DataFrame(rows)
        .sort_values("cost_eur", kind="stable")
        .reset_index(drop=True)
    )


def disagreement_matrix(predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    """Build a per-row prediction matrix and flag disagreements.

    Args:
        predictions: Mapping model name -> sklearn prediction array.

    Returns:
        DataFrame with one column per model + a boolean column
        ``disagreement`` (True if at least two models disagree on this row).
    """
    df = pd.DataFrame({k: np.asarray(v) for k, v in predictions.items()})
    df["disagreement"] = df.nunique(axis=1) > 1
    return df


def sensitivity_curve(
    y_true: np.ndarray,
    train_predict_fn,
    contamination_grid: list[float],
    fp_cost: float = FP_COST,
    fn_cost: float = FN_COST,
) -> pd.DataFrame:
    """Re-train a model across a contamination grid and report cost per setting.

    Args:
        y_true: Ground-truth binary labels (held out, only used to score).
        train_predict_fn: Callable ``contamination -> y_pred_sklearn``.
            Typically a closure around the chosen model + scaled X.
        contamination_grid: Contamination values to test.
        fp_cost: Cost per false positive.
        fn_cost: Cost per false negative.

    Returns:
        DataFrame with columns: contamination, tp, fp, fn, tn,
        cost_eur, recall, precision (one row per contamination).
    """
    rows = []
    for c in contamination_grid:
        y_pred = train_predict_fn(c)
        y_pred_binary = to_binary(y_pred)
        scores = cost_score(y_true, y_pred_binary, fp_cost=fp_cost, fn_cost=fn_cost)
        rows.append({"contamination": float(c), **scores})
    return pd.DataFrame(rows)
