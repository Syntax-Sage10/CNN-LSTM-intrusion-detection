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