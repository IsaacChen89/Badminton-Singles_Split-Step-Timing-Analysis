"""Versioned training runs under ``models/yolo_player_<N>`` and ``models/action_player_<N>``."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

MODELS_DIR = Path("models")
YOLO_KIND = "yolo_player"
ACTION_KIND = "action_player"
YOLO_BEST_NAME = "yolo_player_best.pt"
ACTION_BEST_NAME = "action_best.pt"
RUN_INFO_NAME = "run_info.json"

# Legacy flat folders (pre-versioning).
LEGACY_YOLO_DIR = MODELS_DIR / "yolo_player"
LEGACY_ACTION_DIR = MODELS_DIR / "action_player"


def _run_dir(kind: str, run: int) -> Path:
    return MODELS_DIR / f"{kind}_{run}"


def existing_run_numbers(kind: str) -> list[int]:
    """Return sorted run indices for ``yolo_player`` or ``action_player``."""
    prefix = f"{kind}_"
    nums: list[int] = []
    if not MODELS_DIR.is_dir():
        return nums
    for path in MODELS_DIR.iterdir():
        if not path.is_dir() or not path.name.startswith(prefix):
            continue
        suffix = path.name[len(prefix) :]
        if suffix.isdigit():
            nums.append(int(suffix))
    return sorted(nums)


def next_run_number(kind: str) -> int:
    """Pick ``max(existing) + 1`` (gaps are preserved)."""
    nums = existing_run_numbers(kind)
    return (max(nums) if nums else 0) + 1


def allocate_run_dir(kind: str) -> tuple[Path, int]:
    """Create and return ``(models/<kind>_<N>, N)`` for a new training run."""
    run = next_run_number(kind)
    out = _run_dir(kind, run)
    out.mkdir(parents=True, exist_ok=False)
    return out, run


def project_relative(path: str | Path) -> str:
    """Return ``path`` relative to the project root (cwd), if possible."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(p)


def write_run_info(run_dir: Path, payload: dict[str, Any]) -> Path:
    _verify_checkpoint_payload(run_dir, payload)
    path = run_dir / RUN_INFO_NAME
    path.write_text(json.dumps(payload, indent=2))
    return path


def _verify_checkpoint_payload(run_dir: Path, payload: dict[str, Any]) -> None:
    checkpoint = payload.get("checkpoint")
    if not checkpoint:
        return
    run_root = run_dir.resolve()
    checkpoint_path = Path(str(checkpoint))
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path.cwd() / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Run checkpoint does not exist: {checkpoint_path}")
    try:
        checkpoint_path.relative_to(run_root)
    except ValueError as exc:
        raise ValueError(
            f"Run checkpoint must live inside {run_root}, got {checkpoint_path}"
        ) from exc


def yolo_checkpoint_in_dir(run_dir: Path) -> Optional[Path]:
    """Best fine-tuned YOLO weights inside a run directory."""
    promoted = run_dir / YOLO_BEST_NAME
    if promoted.is_file():
        return promoted.resolve()
    weights_best = run_dir / "weights" / "best.pt"
    if weights_best.is_file():
        return weights_best.resolve()
    return None


def action_checkpoint_in_dir(run_dir: Path) -> Optional[Path]:
    path = run_dir / ACTION_BEST_NAME
    return path.resolve() if path.is_file() else None


def resolve_yolo_run_checkpoint(run: int) -> Path:
    run_dir = _run_dir(YOLO_KIND, run)
    ckpt = yolo_checkpoint_in_dir(run_dir)
    if ckpt is None:
        raise FileNotFoundError(
            f"No YOLO checkpoint found for run {run} (looked in {run_dir})"
        )
    return ckpt


def resolve_action_run_checkpoint(run: int) -> Path:
    run_dir = _run_dir(ACTION_KIND, run)
    ckpt = action_checkpoint_in_dir(run_dir)
    if ckpt is None:
        raise FileNotFoundError(
            f"No action checkpoint found for run {run} (looked in {run_dir})"
        )
    return ckpt


def legacy_yolo_checkpoint() -> Optional[Path]:
    return yolo_checkpoint_in_dir(LEGACY_YOLO_DIR)


def legacy_action_checkpoint() -> Optional[Path]:
    return action_checkpoint_in_dir(LEGACY_ACTION_DIR)


def latest_yolo_run() -> Optional[int]:
    nums = existing_run_numbers(YOLO_KIND)
    for run in reversed(nums):
        if yolo_checkpoint_in_dir(_run_dir(YOLO_KIND, run)) is not None:
            return run
    return None


def latest_action_run() -> Optional[int]:
    nums = existing_run_numbers(ACTION_KIND)
    for run in reversed(nums):
        if action_checkpoint_in_dir(_run_dir(ACTION_KIND, run)) is not None:
            return run
    return None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
