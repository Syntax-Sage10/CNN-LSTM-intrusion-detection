"""
Split -> scale -> select -> reshape.

Rule: nothing that learns from data (MinMaxScaler, SelectKBest) ever sees the
test set. The split happens first; FeaturePipeline is fit on train only and
then applied unchanged to validation, test and production traffic.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass

import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from .config import Config
from .data import Dataset

log = logging.getLogger(__name__)


@dataclass
class Splits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def split_dataset(ds: Dataset, cfg: Config) -> Splits:
    X_tr, X_te, y_tr, y_te = train_test_split(
        ds.X, ds.y, test_size=cfg.test_size, stratify=ds.y, random_state=cfg.seed)
    log.info("Split: train %d | test %d", len(y_tr), len(y_te))
    return Splits(X_tr, y_tr, X_te, y_te)

class FeaturePipeline:
    """Min-Max scaling to [0, 1] followed by mutual-information top-k selection."""

    def __init__(self, k: int = 10, seed: int = 42):
        self.k = k
        self.seed = seed
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.selector: SelectKBest | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FeaturePipeline":
        self.scaler.fit(X)
        k = min(self.k, X.shape[1])
        # functools.partial (not a lambda) so the pipeline stays picklable.
        self.selector = SelectKBest(
            functools.partial(mutual_info_classif, random_state=self.seed), k=k)
        self.selector.fit(self._scale(X), y)
        return self

    def _scale(self, X: np.ndarray) -> np.ndarray:
        # Values outside the training range are clipped, never extrapolated.
        return np.clip(self.scaler.transform(X), 0.0, 1.0)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.selector is None:
            raise RuntimeError("FeaturePipeline is not fitted")
        return self.selector.transform(self._scale(X)).astype(np.float32)

def to_sequences(X2d: np.ndarray) -> np.ndarray:
    """
    (N, k) -> (N, k, 1): each selected feature becomes one 'timestep'.

    Be clear about what this is: feature order is arbitrary, so this is not
    real temporal structure. See windowing.py for actual flow sequences.
    """
    return X2d.reshape(X2d.shape[0], X2d.shape[1], 1).astype(np.float32)