"""
Non-deep baselines. Run these FIRST.

If the CNN-LSTM cannot beat a random forest on the same features and split,
the deep architecture is adding complexity, not detection ability.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier


def build_baseline(name: str, seed: int = 42, balanced: bool = False):
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=seed,
            class_weight="balanced" if balanced else None)
    raise ValueError(f"Unknown baseline '{name}'")