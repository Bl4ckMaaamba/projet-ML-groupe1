"""Anomaly detection models for the AI4I 2020 dataset.

Owner: Isaac. See TASKS_ISAAC.md for the checklist.

All trainers accept a preprocessed feature matrix ``X`` (scaled + encoded
via :func:`src.preprocessing.build_preprocessor`) and return a fitted
estimator. Predictions follow the sklearn convention:

- ``-1`` indicates an anomaly,
- ``1`` indicates an inlier (normal observation).
"""

from __future__ import annotations

import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


def train_isolation_forest(
    X: np.ndarray,
    contamination: float = 0.034,
    n_estimators: int = 200,
    max_samples: str | float = "auto",
    random_state: int = 42,
) -> IsolationForest:
    """Train an Isolation Forest on the preprocessed feature matrix.

    Isolation Forest isolates anomalies via random axis-aligned splits.
    It is invariant to monotonic feature rescaling and does not require
    standardization, although we still feed scaled features to keep a
    single shared ``X_processed`` matrix across all four models.

    Args:
        X: Preprocessed feature matrix of shape ``(n_samples, n_features)``.
        contamination: Expected proportion of anomalies in the dataset.
            Calibrated from the EDA-observed failure rate (3.39%).
        n_estimators: Number of isolation trees in the forest.
        max_samples: Number of samples drawn to fit each tree
            (``"auto"`` falls back to ``min(256, n_samples)``).
        random_state: Seed for reproducibility.

    Returns:
        Fitted :class:`sklearn.ensemble.IsolationForest`.
    """
    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        max_samples=max_samples,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X)
    return model


def train_one_class_svm(
    X: np.ndarray,
    nu: float = 0.034,
    kernel: str = "rbf",
    gamma: str | float = "scale",
) -> OneClassSVM:
    """Train a One-Class SVM on the preprocessed feature matrix.

    OC-SVM learns a non-linear boundary that encloses the bulk of the
    training data. The ``nu`` parameter is an upper bound on the fraction
    of training errors and a lower bound on the fraction of support
    vectors — it plays a role similar to ``contamination`` for the other
    models.

    Args:
        X: Preprocessed feature matrix.
        nu: Fraction of expected anomalies / SVs.
        kernel: Kernel function (``"rbf"`` is the default and most expressive
            choice for a non-linear boundary).
        gamma: Kernel coefficient. ``"scale"`` adapts to the variance of the
            standardized features (recommended).

    Returns:
        Fitted :class:`sklearn.svm.OneClassSVM`.
    """
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    model.fit(X)
    return model


def train_lof(
    X: np.ndarray,
    n_neighbors: int = 35,
    contamination: float = 0.034,
) -> LocalOutlierFactor:
    """Train a Local Outlier Factor estimator (in ``fit_predict`` mode).

    LOF compares the local density of a point to the density of its
    ``n_neighbors`` nearest neighbours. A point in a region whose density
    is much lower than its neighbours' has a high LOF score and is
    flagged as an outlier.

    Args:
        X: Preprocessed feature matrix.
        n_neighbors: Size of the neighbourhood used for density estimation.
            A small ``k`` is noise-sensitive, a large ``k`` smooths subtle
            anomalies. We retain ``35`` after a sweep over ``[10, 20, 35, 50]``.
        contamination: Expected proportion of outliers, used to set the
            decision threshold on the LOF score.

    Returns:
        Unfitted :class:`sklearn.neighbors.LocalOutlierFactor` (call
        ``fit_predict(X)`` to obtain labels — LOF does not expose ``predict``
        unless ``novelty=True``).
    """
    return LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        n_jobs=-1,
    )


def train_elliptic_envelope(
    X: np.ndarray,
    contamination: float = 0.034,
    support_fraction: float | None = None,
    random_state: int = 42,
) -> EllipticEnvelope:
    """Train an Elliptic Envelope (robust covariance) on the feature matrix.

    Fits a robust multivariate Gaussian using the Minimum Covariance
    Determinant (MCD) estimator, then flags points whose Mahalanobis
    distance exceeds a quantile derived from ``contamination``.

    The Gaussian assumption is partially violated on AI4I 2020
    (``Rotational speed [rpm]`` has a skewness of 1.99). We keep the model
    as a baseline but expect degraded recall on non-elliptic anomalies.

    Args:
        X: Preprocessed feature matrix.
        contamination: Expected proportion of anomalies.
        support_fraction: Proportion of points used to compute the MCD.
            ``None`` triggers the automatic rule
            ``(n_samples + n_features + 1) / (2 * n_samples)``.
        random_state: Seed for reproducibility.

    Returns:
        Fitted :class:`sklearn.covariance.EllipticEnvelope`.
    """
    model = EllipticEnvelope(
        contamination=contamination,
        support_fraction=support_fraction,
        random_state=random_state,
    )
    model.fit(X)
    return model
