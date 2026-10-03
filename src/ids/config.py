"""
Experiment configuration.

One dataclass holds every setting. Configs are built by layering YAML files
(configs/default.yaml first, then a dataset file such as configs/nslkdd.yaml),
then applying command-line overrides. Unknown keys raise immediately, so a
typo in a YAML file cannot silently fall back to a default.
"""


import dataclasses
import datetime as dt
import logging
import random
from __future__ import annotations

import argparse
import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path
from pathlib import Path

import numpy as np
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

        # --- Baseline ---
    baseline: str = "random_forest"

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

def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """CLI flags shared by every script."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", action="append", default=[],
                   help="Extra YAML layered on top of configs/default.yaml "
                        "(repeatable), e.g. --config configs/nslkdd.yaml")
    p.add_argument("--no-show", action="store_true", help="Don't open plot windows.")
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    return load_config([DEFAULT_CONFIG, *args.config])

def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def create_run_dir(cfg: Config, model_name: str) -> Path:
    """
    runs/<timestamp>_<model>_<dataset>/ with the exact config used.
    Never overwrites: every experiment keeps its own record.
    """
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run = Path(cfg.runs_dir) / f"{stamp}_{model_name}_{cfg.name}"
    run.mkdir(parents=True, exist_ok=False)
    with open(run / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(dataclasses.asdict(cfg), f, sort_keys=False)
    log.info("Run directory: %s", run)
    return run


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")