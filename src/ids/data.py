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
# Processed cache
# ---------------------------------------------------------------------------
def save_processed(ds: Dataset, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, X=ds.X, y=ds.y,
        class_names=np.array(ds.class_names, dtype=str),
        feature_names=np.array(ds.feature_names, dtype=str),
        groups=ds.groups if ds.groups is not None else np.array([], dtype=str),
    )
    log.info("Saved processed dataset to %s", path)


def load_processed(path: str) -> Dataset:
    z = np.load(path, allow_pickle=False)
    groups = z["groups"] if z["groups"].size else None
    return Dataset(z["X"], z["y"], z["class_names"].tolist(),
                   z["feature_names"].tolist(), groups)


def get_dataset(cfg: Config) -> Dataset:
    """Load the processed cache; for dummy data, generate it on first use."""
    if Path(cfg.processed_path).exists():
        ds = load_processed(cfg.processed_path)
    elif cfg.dataset == "dummy":
        ds = generate_dummy_data(seed=cfg.seed)
        save_processed(ds, cfg.processed_path)
    else:
        raise FileNotFoundError(
            f"{cfg.processed_path} not found. Run scripts/prepare_data.py first.")
    counts = np.bincount(ds.y, minlength=len(ds.class_names)).tolist()
    log.info("Dataset '%s': %d samples, %d features, classes %s",
             cfg.name, len(ds.y), ds.X.shape[1], dict(zip(ds.class_names, counts)))
    return ds