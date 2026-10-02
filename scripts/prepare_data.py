#!/usr/bin/env python3
"""
raw/ -> processed/.  Run once per dataset.

    python scripts/prepare_data.py                              # dummy data
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ids.config import build_arg_parser, config_from_args, setup_logging
from ids.data import generate_dummy_data, save_processed


def main() -> None:
    setup_logging()
    cfg = config_from_args(build_arg_parser(__doc__).parse_args())
    ds = generate_dummy_data(seed=cfg.seed)
    save_processed(ds, cfg.processed_path)
    print(f"{len(ds.y)} samples, {ds.X.shape[1]} features, classes: {ds.class_names}")


if __name__ == "__main__":
    main()