"""Thin wrapper: ``python scripts/train_action.py --manifest data/action/manifest.csv ...``."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from main import app  # noqa: E402

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "train-action", *sys.argv[1:]]
    app()
