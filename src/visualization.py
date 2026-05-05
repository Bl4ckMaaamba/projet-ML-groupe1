"""Visualizations for anomaly detection models.

Owner: Diego. See TASKS_DIEGO.md for the checklist.

Conventions:
- sklearn anomaly labels: -1 = anomaly, 1 = inlier
- Color convention in scatter plots: red = anomaly, blue = inlier
"""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def pca_2d(X: np.ndarray, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Project X onto 2 principal components. Returns (X_2d, explained_variance_ratio)."""
    raise NotImplementedError


def tsne_2d(
    X: np.ndarray,
    perplexity: float = 30.0,
    random_state: int = 42,
    n_iter: int = 1000,
) -> np.ndarray:
    """Project X to 2D using t-SNE."""
    raise NotImplementedError


def plot_model_decisions(
    X_2d: np.ndarray,
    y_pred: np.ndarray,
    ax: Axes,
    title: str,
    alpha: float = 0.4,
    s: float = 12,
) -> Axes:
    """Scatter X_2d colored by anomaly prediction (-1 red / 1 blue)."""
    raise NotImplementedError


def plot_models_grid(
    X_2d: np.ndarray,
    predictions: dict[str, np.ndarray],
    save_path: str | None = None,
    figsize: tuple[float, float] = (12, 10),
    dpi: int = 150,
) -> plt.Figure:
    """2x2 grid of model decisions overlaid on a 2D projection."""
    raise NotImplementedError


def plot_cost_curve(
    contaminations: Iterable[float],
    costs: Iterable[float],
    save_path: str | None = None,
    figsize: tuple[float, float] = (8, 5),
    dpi: int = 150,
) -> plt.Figure:
    """Plot total business cost as a function of contamination."""
    raise NotImplementedError
