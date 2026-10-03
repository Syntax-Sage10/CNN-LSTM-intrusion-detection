# Hybrid CNN-LSTM Intrusion Detection System

A PyTorch CNN-LSTM network-intrusion classifier, evaluated against a non-deep
baseline on the same features and the same split.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # optional; scripts also work without it
pytest                    # model tests skip if torch isn't installed
```

Run everything from the project root.

## Workflow (in this order)

```bash
# 1. Prepare data (dummy data needs no download)
python scripts/prepare_data.py

# 2. Baseline FIRST. This is the number to beat.
python scripts/train_baseline.py
```

Each run writes `runs/<timestamp>_<model>_<dataset>/` containing `config.yaml`,
`metrics.json`, `confusion_matrix.png`, the model, and the fitted preprocessing.
Nothing is overwritten.