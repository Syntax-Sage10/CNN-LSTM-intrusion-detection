#!/usr/bin/env python3
"""
Train the non-deep baseline on the SAME split and SAME 10 features as the
CNN-LSTM. Run this first; its macro-F1 is the number to beat.

    python scripts/train_baseline.py
    python scripts/train_baseline.py --config configs/nslkdd.yaml --baseline xgboost
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import logging

import joblib

from ids.config import (build_arg_parser, config_from_args, create_run_dir,
                        set_seed, setup_logging)
from ids.data import get_dataset
from ids.evaluate import full_metrics, plot_confusion, print_report, save_json
from ids.models.baselines import build_baseline
from ids.preprocessing import FeaturePipeline, save_pipeline, split_dataset

log = logging.getLogger("train_baseline")


def main() -> None:
    setup_logging()
    args = build_arg_parser(__doc__).parse_args()
    cfg = config_from_args(args)
    set_seed(cfg.seed)

    ds = get_dataset(cfg)
    s = split_dataset(ds, cfg)

    pipe = FeaturePipeline(cfg.k_features, cfg.seed).fit(s.X_train, s.y_train)
    log.info("Selected features: %s", pipe.selected(ds.feature_names))

    model = build_baseline(cfg.baseline, cfg.seed)
    model.fit(pipe.transform(s.X_train), s.y_train)
    y_pred = model.predict(pipe.transform(s.X_test))

    run = create_run_dir(cfg, cfg.baseline)
    metrics = full_metrics(s.y_test, y_pred, ds.class_names)
    metrics["selected_features"] = pipe.selected(ds.feature_names)
    print_report(s.y_test, y_pred, ds.class_names, metrics,
                 f"TEST - {cfg.baseline} on {cfg.name}")
    save_json(metrics, run / "metrics.json")
    plot_confusion(s.y_test, y_pred, ds.class_names, run / "confusion_matrix.png",
                   show=not args.no_show)
    joblib.dump(model, run / "model.joblib")
    save_pipeline(run / "preprocessing.joblib", pipe, ds.class_names, ds.feature_names)
    print(f"\nRun saved to {run}")


if __name__ == "__main__":
    main()