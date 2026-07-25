"""Resolve YOLO checkpoint paths and avoid stray root downloads.

Stock base weights (``yolo26n.pt``, ``yolo26n-cls.pt``) live under ``models/``.
Fine-tuned detector runs live under ``models/yolo_player_<N>/`` with
Ultralytics ``weights/best.pt`` (no promoted ``yolo_player_best.pt`` copy).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

MODELS_DIR = Path("models")
YOLO_PLAYER_DIR = MODELS_DIR / "yolo_player"


def resolve_yolo_weight(spec: str) -> Path:
    """Return an existing local checkpoint path, or ``spec`` if none found.

    Search order: as given → ``models/<name>`` → legacy ``models/yolo_player/<name>``.
    """
    if not spec:
        return Path(spec)
    p = Path(spec)
    if p.is_file():
        return p.resolve()
    name = Path(spec).name
    under_models = MODELS_DIR / name
    if under_models.is_file():
        return under_models.resolve()
    legacy = YOLO_PLAYER_DIR / name
    if legacy.is_file():
        return legacy.resolve()
    return p


def remove_stray_root_weight(basename: str, *, keep: Optional[Path] = None) -> None:
    """Delete ``<basename>`` in the project root if it duplicates ``keep``."""
    stray = Path.cwd() / basename
    if not stray.is_file():
        return
    if keep is not None and stray.resolve() == keep.resolve():
        return
    stray.unlink()


@contextmanager
def ultralytics_weights_cwd(directory: Path) -> Iterator[None]:
    """Run Ultralytics load/train with CWD in ``directory`` so downloads stay there."""
    directory.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous)