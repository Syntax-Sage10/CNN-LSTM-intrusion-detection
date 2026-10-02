"""
Data loading: raw CSV -> cleaned numeric matrix -> processed .npz.

Only label-independent, statistics-free cleaning happens here (renaming,
dropping identifier columns, one-hot encoding, removing NaN/inf rows).
Anything that learns from the data (scaling, feature selection) lives in
preprocessing.py and is fit on the training split only.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification

from .config import Config

log = logging.getLogger(__name__)


@dataclass
class Dataset:
    X: np.ndarray                 # (n_samples, n_features) float32
    y: np.ndarray                 # (n_samples,) int64 class indices
    class_names: list[str]
    feature_names: list[str]
    groups: np.ndarray | None     # (n_samples,) str, e.g. source file / capture day


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
def generate_dummy_data(n_samples: int = 20_000, n_features: int = 41,
                        seed: int = 42) -> Dataset:
    """
    NSL-KDD-shaped synthetic data: 41 features, 5 classes, long-tail imbalance.
    For smoke-testing the pipeline only. Results on it mean nothing.
    """
    class_names = ["normal", "dos", "probe", "r2l", "u2r"]
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features, n_informative=12,
        n_redundant=8, n_classes=len(class_names), n_clusters_per_class=2,
        weights=[0.53, 0.36, 0.09, 0.015, 0.005], class_sep=1.0,
        flip_y=0.01, random_state=seed,
    )
    # Skewed, non-negative like real byte/count features. exp() is monotonic,
    # so class information is preserved.
    X = np.exp(X / 2.0) * np.random.default_rng(seed).uniform(1, 1000, n_features)
    return Dataset(X.astype(np.float32), y.astype(np.int64), class_names,
                   [f"f{i}" for i in range(n_features)], None)