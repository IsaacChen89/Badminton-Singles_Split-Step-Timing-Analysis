"""Resolve YOLO checkpoint paths under ``models/yolo_player/`` and avoid stray root downloads."""

from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

YOLO_PLAYER_DIR = Path("models") / "yolo_player"
FINETUNED_BASENAME = "yolo_player_best.pt"


def resolve_yolo_weight(spec: str) -> Path:
    """Return an existing local checkpoint path, or ``spec`` if none found.

    Search order: as given → ``models/<spec>`` → ``models/yolo_player/<name>``.
    """
    if not spec:
        return Path(spec)
    p = Path(spec)
    if p.is_file():
        return p.resolve()
    under_models = Path("models") / spec
    if under_models.is_file():
        return under_models.resolve()
    in_yolo_player = YOLO_PLAYER_DIR / Path(spec).name
    if in_yolo_player.is_file():
        return in_yolo_player.resolve()
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


def promote_finetuned_best(save_dir: Path) -> Optional[Path]:
    """Copy ``<save_dir>/weights/best.pt`` → ``models/yolo_player/yolo_player_best.pt``."""
    best = save_dir / "weights" / "best.pt"
    if not best.is_file():
        return None
    YOLO_PLAYER_DIR.mkdir(parents=True, exist_ok=True)
    dest = YOLO_PLAYER_DIR / FINETUNED_BASENAME
    shutil.copy2(best, dest)
    return dest.resolve()
