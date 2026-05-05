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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def pca_2d(X: np.ndarray, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Project X onto 2 principal components.

    Args:
        X: Feature matrix (n_samples, n_features), already scaled.
        random_state: For reproducibility.

    Returns:
        (X_2d, explained_variance_ratio) where X_2d has shape (n_samples, 2)
        and explained_variance_ratio has shape (2,).
    """
    pca = PCA(n_components=2, random_state=random_state)
    X_2d = pca.fit_transform(X)
    return X_2d, pca.explained_variance_ratio_


def tsne_2d(
    X: np.ndarray,
    perplexity: float = 30.0,
    random_state: int = 42,
    max_iter: int = 1000,
) -> np.ndarray:
    """Project X to 2D using t-SNE.

    t-SNE preserves local neighborhoods (non-linear) whereas PCA preserves
    global variance (linear). Useful for revealing clusters that PCA may miss.

    Args:
        X: Feature matrix (n_samples, n_features), already scaled.
        perplexity: Roughly the number of effective nearest neighbors.
            Try [5, 30, 50] and pick the one with the clearest separation.
        random_state: For reproducibility.
        max_iter: Number of optimization iterations.

    Returns:
        2D embedding of shape (n_samples, 2).
    """
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        max_iter=max_iter,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(X)


def plot_model_decisions(
    X_2d: np.ndarray,
    y_pred: np.ndarray,
    ax: Axes,
    title: str,
    alpha: float = 0.4,
    s: float = 12,
) -> Axes:
    """Scatter X_2d colored by anomaly prediction.

    Args:
        X_2d: 2D projection of the feature matrix (n_samples, 2).
        y_pred: sklearn-style anomaly labels (-1 = anomaly, 1 = inlier).
        ax: Matplotlib axes to draw on.
        title: Subplot title (e.g., "Isolation Forest — 3.4% anomalies").
        alpha: Point transparency.
        s: Point size.

    Returns:
        The same axes (for chaining).
    """
    y_pred = np.asarray(y_pred)
    is_anomaly = y_pred == -1
    ax.scatter(
        X_2d[~is_anomaly, 0],
        X_2d[~is_anomaly, 1],
        c="tab:blue",
        alpha=alpha,
        s=s,
        label="Inlier",
    )
    ax.scatter(
        X_2d[is_anomaly, 0],
        X_2d[is_anomaly, 1],
        c="tab:red",
        alpha=alpha,
        s=s,
        label="Anomaly",
    )
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(loc="best", fontsize=8)
    return ax


def plot_models_grid(
    X_2d: np.ndarray,
    predictions: dict[str, np.ndarray],
    save_path: str | None = None,
    figsize: tuple[float, float] = (12, 10),
    dpi: int = 150,
) -> plt.Figure:
    """Plot a 2x2 grid of model decisions overlaid on a 2D projection.

    Args:
        X_2d: 2D projection (typically PCA) shared across all subplots.
        predictions: Mapping model name -> sklearn-style prediction array.
            Expected keys (in order): isolation_forest, ocsvm, lof, elliptic.
        save_path: If given, save the figure to this path.
        figsize: Figure size in inches.
        dpi: Resolution for save.

    Returns:
        The matplotlib figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    expected_order = ["isolation_forest", "ocsvm", "lof", "elliptic"]
    pretty = {
        "isolation_forest": "Isolation Forest",
        "ocsvm": "One-Class SVM",
        "lof": "Local Outlier Factor",
        "elliptic": "Elliptic Envelope",
    }
    flat_axes = axes.flatten()
    for ax, key in zip(flat_axes, expected_order):
        if key not in predictions:
            ax.set_axis_off()
            ax.set_title(f"{pretty.get(key, key)} (missing)")
            continue
        y_pred = np.asarray(predictions[key])
        anomaly_pct = float((y_pred == -1).mean()) * 100
        title = f"{pretty[key]} — {anomaly_pct:.1f}% anomalies"
        plot_model_decisions(X_2d, y_pred, ax, title)
    fig.suptitle("Model decisions overlaid on PCA projection", fontsize=14)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_cost_curve(
    contaminations: Iterable[float],
    costs: Iterable[float],
    save_path: str | None = None,
    figsize: tuple[float, float] = (8, 5),
    dpi: int = 150,
) -> plt.Figure:
    """Plot total business cost as a function of contamination.

    A vertical line marks the cost-minimizing contamination.

    Args:
        contaminations: Tested contamination values.
        costs: Total cost for each contamination value (same order).
        save_path: If given, save the figure to this path.
        figsize: Figure size in inches.
        dpi: Resolution for save.

    Returns:
        The matplotlib figure.
    """
    contaminations = np.asarray(list(contaminations))
    costs = np.asarray(list(costs))
    best_idx = int(np.argmin(costs))
    best_c = float(contaminations[best_idx])
    best_cost = float(costs[best_idx])

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(contaminations, costs, marker="o", color="tab:purple")
    ax.axvline(best_c, color="tab:green", linestyle="--", label=f"min @ c={best_c:.3f}")
    ax.scatter([best_c], [best_cost], color="tab:green", zorder=5)
    ax.set_xlabel("Contamination")
    ax.set_ylabel("Total cost (€)")
    ax.set_title("Sensitivity of business cost to contamination")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig
