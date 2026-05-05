from __future__ import annotations

import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope


def train_isolation_forest(
    X: np.ndarray,
    contamination: float = 0.034,
    n_estimators: int = 200,
    max_samples: str | float = "auto",
    random_state: int = 42,
) -> IsolationForest:

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

    model = OneClassSVM(
        nu=nu,
        kernel=kernel,
        gamma=gamma,
    )

    model.fit(X)

    return model


def train_lof(
    X: np.ndarray,
    n_neighbors: int = 35,
    contamination: float = 0.034,
) -> LocalOutlierFactor:

    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        n_jobs=-1,
    )

    return model


def train_elliptic_envelope(
    X: np.ndarray,
    contamination: float = 0.034,
    support_fraction: float | None = None,
    random_state: int = 42,
) -> EllipticEnvelope:

    model = EllipticEnvelope(
        contamination=contamination,
        support_fraction=support_fraction,
        random_state=random_state,
    )

    model.fit(X)

    return model