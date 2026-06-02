"""Thin wrapper: ``python scripts/convert_cvat.py ...``.

Forwards to ``main.py convert-cvat``. Exists so users can ``cd scripts/`` and
run conversion without remembering the typer sub-command.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from main import app  # noqa: E402

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "convert-cvat", *sys.argv[1:]]
    app()
