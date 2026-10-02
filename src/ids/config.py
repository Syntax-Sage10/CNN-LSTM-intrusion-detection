"""
Experiment configuration.

One dataclass holds every setting. Configs are built by layering YAML files
(configs/default.yaml first, then a dataset file such as configs/nslkdd.yaml),
then applying command-line overrides. Unknown keys raise immediately, so a
typo in a YAML file cannot silently fall back to a default.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


@dataclass
class Config:
    # --- Identity ---
    name: str = "dummy"

    # --- Data source ---
    dataset: str = "dummy"                      # "dummy" or "csv"
    processed_path: str = "data/processed/dummy.npz"

    # --- Split ---
    test_size: float = 0.2

    # --- Features ---
    k_features: int = 10

    # --- Model ---
    conv_filters: int = 64
    conv_kernel: int = 1
    pool_size: int = 2
    lstm_units: int = 64
    dense_units: int = 32
    dropout: float = 0.3

    # --- Training ---
    batch_size: int = 32
    epochs: int = 10
    lr: float = 1e-3

    # --- Misc ---
    seed: int = 42
    runs_dir: str = "runs"


def load_config(paths: list[str | Path], overrides: dict | None = None) -> Config:
    """Merge YAML files left to right, then apply non-None overrides."""
    merged: dict = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            merged.update(yaml.safe_load(f) or {})
    if overrides:
        merged.update({k: v for k, v in overrides.items() if v is not None})

    known = {f.name for f in dataclasses.fields(Config)}
    unknown = set(merged) - known
    if unknown:
        raise KeyError(f"Unknown config keys: {sorted(unknown)}")
    return Config(**merged)