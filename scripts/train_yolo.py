"""Thin wrapper: ``python scripts/train_yolo.py --data data/yolo/data.yaml ...``."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from main import app  # noqa: E402

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "train-yolo", *sys.argv[1:]]
    app()
