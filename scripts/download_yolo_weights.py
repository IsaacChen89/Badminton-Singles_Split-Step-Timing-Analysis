"""Materialize a stock YOLO checkpoint into ``models/``.

Usage::

    python scripts/download_yolo_weights.py            # default: yolo26n.pt
    python scripts/download_yolo_weights.py yolo26s.pt
    python scripts/download_yolo_weights.py yolo26n-cls.pt   # BoT-SORT Re-ID (strong tracking)
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "yolo26n.pt"
    target_dir = Path(__file__).resolve().parent.parent / "models"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / name

    if target.exists():
        print(f"Already present: {target}")
        return 0

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed. Run: pip install -r requirements.txt", file=sys.stderr)
        return 2

    print(f"Asking ultralytics to fetch {name} ...")
    model = YOLO(name)  # triggers download to ultralytics' weight cache
    src = Path(model.ckpt_path) if hasattr(model, "ckpt_path") and model.ckpt_path else None
    if src is None or not src.exists():
        # Fallback: ultralytics often writes to CWD on first download.
        cwd_candidate = Path.cwd() / name
        if cwd_candidate.exists():
            src = cwd_candidate
    if src is None or not src.exists():
        print("Could not locate the downloaded weights file. Check ultralytics cache.", file=sys.stderr)
        return 1

    shutil.copy2(src, target)
    print(f"Copied {src} -> {target}")
    # Ultralytics may also drop a copy in the project root on first fetch.
    stray = Path.cwd() / name
    if stray.exists() and stray.resolve() != target.resolve():
        stray.unlink()
        print(f"Removed duplicate in project root: {stray}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
