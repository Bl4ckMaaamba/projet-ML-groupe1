"""Data loading and preprocessing for AI4I 2020 anomaly detection.

Owner: Noah. See TASKS_NOAH.md for the checklist.

The pipeline standardizes the 5 numeric sensors and one-hot encodes the
``Type`` categorical feature. Identifier columns and label columns are
dropped from the training matrix; ``Machine failure`` is held out
separately for the Part 4 evaluation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES: list[str] = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
CATEGORICAL_FEATURES: list[str] = ["Type"]
ID_COLUMNS: list[str] = ["UDI", "Product ID"]
LABEL_COLUMNS: list[str] = ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the AI4I 2020 dataset from a CSV file.

    Args:
        path: Path to ``ai4i2020.csv``.

    Returns:
        DataFrame with all 14 original columns.
    """
    return pd.read_csv(path)


def split_features_and_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the feature matrix from the held-out target.

    The ``Machine failure`` column and its 5 subtypes must not be used as
    a training signal in an unsupervised pipeline. We hold ``Machine failure``
    aside for the Part 4 reveal and drop the subtypes (which are leaks of
    the global label).

    Args:
        df: Raw dataset returned by :func:`load_data`.

    Returns:
        ``(X, y_true)`` where ``X`` contains only training features
        (``Type`` + 5 numeric sensors) and ``y_true`` is the held-out
        ``Machine failure`` series.
    """
    y_true = df["Machine failure"].copy()
    X = df.drop(columns=ID_COLUMNS + LABEL_COLUMNS)
    return X, y_true


def build_preprocessor() -> ColumnTransformer:
    """Build the sklearn preprocessing pipeline.

    Numeric features are standardized via ``StandardScaler`` (necessary for
    LOF, One-Class SVM, and Elliptic Envelope; harmless for Isolation
    Forest). The ``Type`` categorical feature is one-hot encoded with
    ``drop="first"`` to avoid perfect collinearity.

    Returns:
        Unfitted ``ColumnTransformer`` ready to be passed to ``fit_transform``.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
