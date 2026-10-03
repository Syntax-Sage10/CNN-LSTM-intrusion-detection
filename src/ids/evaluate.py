"""
Evaluation: metrics, report, confusion matrix, JSON output.

Headline numbers are MACRO-averaged. On imbalanced IDS data, accuracy and
weighted averages are dominated by 'normal'/'dos' and hide failure on rare
attacks. Per-class recall is reported for every class.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import matplotlib

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix,
                             precision_recall_fscore_support)

log = logging.getLogger(__name__)


def summary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro",
                                                  zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(p), "recall_macro": float(r), "f1_macro": float(f1)}


def full_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                 class_names: list[str]) -> dict:
    labels = list(range(len(class_names)))
    m = summary_metrics(y_true, y_pred)
    m["f1_weighted"] = float(precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)[2])
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                  zero_division=0)
    m["per_class"] = {name: {"precision": float(p[i]), "recall": float(r[i]),
                             "f1": float(f1[i]), "support": int(s[i])}
                      for i, name in enumerate(class_names)}
    return m


def print_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str],
                 metrics: dict, title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(f"Accuracy          : {metrics['accuracy']:.4f}")
    print(f"Precision (macro) : {metrics['precision_macro']:.4f}")
    print(f"Recall    (macro) : {metrics['recall_macro']:.4f}")
    print(f"F1        (macro) : {metrics['f1_macro']:.4f}   <- headline number")
    print(f"F1     (weighted) : {metrics['f1_weighted']:.4f}   <- flattering on imbalanced data\n")
    print(classification_report(y_true, y_pred, labels=list(range(len(class_names))),
                                target_names=class_names, digits=4, zero_division=0))