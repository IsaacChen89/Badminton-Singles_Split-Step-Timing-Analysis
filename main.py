"""Top-level Typer CLI for the Badminton Split Step Analyzer.

Sub-commands
------------
- ``analyze``           Run end-to-end inference on a video.
- ``convert-cvat``      Turn CVAT for Video 1.1 XML into YOLO + action datasets.
- ``train-yolo``        Fine-tune the player detector on a CVAT-derived dataset.
- ``train-action``      Train the split-step CNN-LSTM on a CVAT-derived dataset.
- ``plot-training``     Regenerate action training curve PNGs from history JSON.
- ``info``              Print resolved config, device, and discovered weights.

All commands accept ``--config <path>`` (defaults to ``config.yaml``).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import typer

from src.utils.config import load_config, resolve_device, mps_available, device_supports_pin_memory
from src.utils.logging import setup_logging, get_logger


app = typer.Typer(
    add_completion=False,
    help="Badminton Singles - Split Step Timing Analysis",
    no_args_is_help=True,
)


def _resolve_analyze_yolo(
    yolo_weights: Optional[Path],
    yolo_run: Optional[int],
    config_default: str,
) -> str:
    from src.utils.model_runs import (
        latest_yolo_run,
        legacy_yolo_checkpoint,
        resolve_yolo_run_checkpoint,
    )

    if yolo_weights is not None and yolo_run is not None:
        raise typer.BadParameter("Use either --yolo-weights or --yolo-run, not both.")
    if yolo_weights is not None:
        return str(yolo_weights)
    if yolo_run is not None:
        return str(resolve_yolo_run_checkpoint(yolo_run))
    if Path(config_default).is_file():
        return config_default
    latest = latest_yolo_run()
    if latest is not None:
        return str(resolve_yolo_run_checkpoint(latest))
    legacy = legacy_yolo_checkpoint()
    if legacy is not None:
        return str(legacy)
    return config_default


def _resolve_analyze_action(
    action_weights: Optional[Path],
    action_run: Optional[int],
    config_default: str,
) -> str:
    from src.utils.model_runs import (
        latest_action_run,
        legacy_action_checkpoint,
        resolve_action_run_checkpoint,
    )

    if action_weights is not None and action_run is not None:
        raise typer.BadParameter("Use either --action-weights or --action-run, not both.")
    if action_weights is not None:
        return str(action_weights)
    if action_run is not None:
        return str(resolve_action_run_checkpoint(action_run))
    if Path(config_default).is_file():
        return config_default
    latest = latest_action_run()
    if latest is not None:
        return str(resolve_action_run_checkpoint(latest))
    legacy = legacy_action_checkpoint()
    if legacy is not None:
        return str(legacy)
    return config_default


# -------------------------------------------------------------------- #
# analyze
# -------------------------------------------------------------------- #
@app.command()
def analyze(
    video: Path = typer.Option(..., "--video", exists=True, dir_okay=False, help="Input video file."),
    output: Path = typer.Option(..., "--output", help="Output annotated MP4."),
    config_path: Path = typer.Option(Path("config.yaml"), "--config"),
    frame_skip: Optional[int] = typer.Option(None, "--frame-skip", help="Process every Nth frame."),
    device: Optional[str] = typer.Option(None, "--device", help="auto|cpu|cuda|mps"),
    yolo_weights: Optional[Path] = typer.Option(None, "--yolo-weights", help="Override detector weights path."),
    yolo_run: Optional[int] = typer.Option(
        None, "--yolo-run", min=1, help="Use models/yolo_player_<N>/yolo_player_best.pt."
    ),
    action_weights: Optional[Path] = typer.Option(None, "--action-weights", help="Override action checkpoint path."),
    action_run: Optional[int] = typer.Option(
        None, "--action-run", min=1, help="Use models/action_player_<N>/action_best.pt."
    ),
    no_hud: bool = typer.Option(False, "--no-hud", help="Disable on-screen HUD."),
    tracking_mode: Optional[str] = typer.Option(
        None,
        "--tracking-mode",
        help=(
            "Tracking mode: 'strong' (BoT-SORT + Re-ID + trajectory + smoothing, "
            "default), 'normal' (BoT-SORT defaults + sticky map), or "
            "'court-side-fallback' (court-half rule each frame, ignores tracker IDs)."
        ),
    ),
    player1_position: Optional[str] = typer.Option(
        None,
        "--player1-position",
        help="Where Player 1 is: top|bottom|left|right (default from config).",
    ),
    debug_ids: bool = typer.Option(
        False, "--debug-ids", help="Show raw tracker IDs next to each player box."
    ),
    min_confidence: Optional[float] = typer.Option(
        None,
        "--min-confidence",
        help="Drop detections below this YOLO confidence (overrides config).",
    ),
    target_fps: Optional[float] = typer.Option(
        None,
        "--target-fps",
        help=(
            "Resample the input video to this FPS before processing so the "
            "action model sees the cadence it was trained on. Pass 0 to disable "
            "(use the source FPS). Default comes from pipeline.target_fps in "
            "config.yaml (30)."
        ),
    ),
    max_frames: Optional[int] = typer.Option(None, "--max-frames", help="Stop after N frames (debugging)."),
) -> None:
    """Run detection + tracking + split-step classification on a video."""
    cfg = load_config(config_path)
    if device:
        cfg.device = device  # type: ignore[assignment]
    if frame_skip is not None:
        cfg.pipeline.frame_skip = frame_skip
    if target_fps is not None:
        cfg.pipeline.target_fps = target_fps if target_fps > 0 else None
    if no_hud:
        cfg.pipeline.draw_hud = False
    cfg.detection.finetuned_model_path = _resolve_analyze_yolo(
        yolo_weights, yolo_run, cfg.detection.finetuned_model_path
    )
    cfg.action.model_checkpoint = _resolve_analyze_action(
        action_weights, action_run, cfg.action.model_checkpoint
    )
    if tracking_mode:
        # accept the user-friendly hyphenated form too
        normalized = tracking_mode.lower().replace("-", "_").strip()
        if normalized == "court_side_fallback":
            normalized = "court_side"
        if normalized not in {"strong", "normal", "court_side"}:
            raise typer.BadParameter(
                f"--tracking-mode must be one of strong|normal|court-side-fallback "
                f"(got '{tracking_mode}')"
            )
        cfg.tracking.mode = normalized
    if player1_position:
        if player1_position not in {"top", "bottom", "left", "right"}:
            raise typer.BadParameter(
                "--player1-position must be one of top|bottom|left|right"
            )
        cfg.assignment.player1_position = player1_position
    if debug_ids:
        cfg.tracking.show_tracker_ids = True
    if min_confidence is not None:
        cfg.assignment.min_confidence = min_confidence
        cfg.detection.conf_threshold = min(
            cfg.detection.conf_threshold, max(0.05, min_confidence)
        )

    logger = setup_logging(cfg.log_level)
    resolved_device = resolve_device(cfg.device)
    logger.info(
        f"Device={resolved_device}  tracking_mode={cfg.tracking.mode}  "
        f"player1_position={cfg.assignment.player1_position}"
    )
    logger.info(f"YOLO weights: {cfg.detection.finetuned_model_path}")
    logger.info(f"Action checkpoint: {cfg.action.model_checkpoint}")

    output.parent.mkdir(parents=True, exist_ok=True)

    # Lazy imports so `--help` and `info` don't pay for torch import time.
    from src.data.video_io import (
        VideoReader,
        VideoWriter,
        effective_fps,
        iter_frames_at_fps,
    )
    from src.detection.yolo_detector import YOLODetector
    from src.tracking.tracker import PlayerTracker
    from src.tracking.player_assigner import PlayerAssigner
    from src.action.model import load_checkpoint
    from src.action.inference import RollingActionInference
    from src.action.smoothing import LabelSmoother
    from src.utils.geometry import crop_with_padding
    from src.visualization.overlay import render_annotated_frame

    # Detection + tracking
    detector = YOLODetector(
        model_path=cfg.detection.finetuned_model_path,
        stock_model_path=cfg.detection.model_path,
        device=resolved_device,
        conf=cfg.detection.conf_threshold,
        iou=cfg.detection.iou_threshold,
        classes=cfg.detection.classes,
        imgsz=cfg.detection.imgsz,
        max_det=cfg.detection.max_det,
    )
    tracker = PlayerTracker(
        detector=detector,
        mode=cfg.tracking.mode,
        tracker_yaml=cfg.tracking.tracker_yaml,
        persist=cfg.tracking.persist,
    )
    assigner = PlayerAssigner(
        mode=cfg.tracking.mode,  # type: ignore[arg-type]
        player1_position=cfg.assignment.player1_position,  # type: ignore[arg-type]
        top_is_player1=cfg.assignment.top_is_player1,
        reassign_after_lost_frames=cfg.assignment.reassign_after_lost_frames,
        bbox_smoothing_alpha=cfg.assignment.bbox_smoothing_alpha,
        velocity_alpha=cfg.assignment.velocity_alpha,
        predict_max_frames=cfg.assignment.predict_max_frames,
        min_confidence=cfg.assignment.min_confidence,
        iou_recovery_threshold=cfg.assignment.iou_recovery_threshold,
    )

    # Action model (optional — runs without it but won't emit SPLIT STEP).
    action_model = None
    ckpt = Path(cfg.action.model_checkpoint)
    if ckpt.exists():
        try:
            action_model = load_checkpoint(ckpt, map_location=resolved_device)
            logger.info(f"Loaded action checkpoint: {ckpt}")
        except Exception as exc:
            logger.warning(f"Failed to load action checkpoint {ckpt}: {exc}")
    else:
        logger.warning(
            f"Action checkpoint not found at {ckpt}. "
            f"The video will still be annotated with player boxes, but "
            f"'SPLIT STEP' labels won't fire until you train and place a "
            f"checkpoint there."
        )

    rolling = RollingActionInference(
        model=action_model,
        clip_length=cfg.action.clip_length,
        input_size=cfg.action.input_size,
        device=resolved_device,
        stride=1,
    )
    smoother = LabelSmoother(
        ema_alpha=cfg.smoothing.ema_alpha,
        prob_on=cfg.smoothing.prob_on,
        prob_off=cfg.smoothing.prob_off,
        min_on_frames=cfg.smoothing.min_on_frames,
        cooldown_frames=cfg.smoothing.cooldown_frames,
    )

    last_assigned = []  # type: ignore[var-annotated]
    last_flags: dict[int, bool] = {1: False, 2: False}
    last_probs: dict[int, float] = {1: 0.0, 2: 0.0}

    started = time.perf_counter()
    n_processed = 0

    with VideoReader(video) as reader:
        # Required by the court-side rule and bbox clamping.
        assigner.set_frame_size(reader.width, reader.height)
        eff_fps = effective_fps(reader.fps, cfg.pipeline.target_fps)
        if eff_fps != reader.fps:
            logger.info(
                f"Resampling input {reader.fps:.2f} FPS -> {eff_fps:.2f} FPS "
                f"to match the action model's training cadence."
            )
        else:
            logger.info(f"Input FPS={reader.fps:.2f} (no resampling).")
        out_fps = cfg.pipeline.output_fps or eff_fps
        with VideoWriter(output, out_fps, (reader.width, reader.height)) as writer:
            for frame_idx, frame in iter_frames_at_fps(reader, cfg.pipeline.target_fps):
                if max_frames is not None and frame_idx >= max_frames:
                    break

                run_pipeline = (frame_idx % max(1, cfg.pipeline.frame_skip)) == 0
                if run_pipeline:
                    detections = tracker.update(frame)
                    assigned = assigner.assign(detections, frame_idx=frame_idx)
                    last_assigned = assigned

                    # Per-player rolling action inference + smoothing.
                    # We crop from the *raw* detection box (more accurate); the
                    # smoothed bbox is only for rendering. When a slot is being
                    # held over with a predicted bbox we don't push a new crop
                    # (avoids feeding stale frames into the model) but we do
                    # decay the probability via a 0.0 update so SPLIT STEP
                    # eventually resets.
                    for ap in assigned:
                        if ap.detection is not None and not ap.predicted:
                            crop = crop_with_padding(
                                frame, ap.detection.bbox, pad_ratio=0.15
                            )
                            prob = rolling.update(ap.player_id, crop)
                        else:
                            prob = 0.0
                        on, smoothed = smoother.update(ap.player_id, prob)
                        last_flags[ap.player_id] = on
                        last_probs[ap.player_id] = smoothed

                    # Decay probability for any player that disappeared this frame.
                    seen_pids = {ap.player_id for ap in assigned}
                    for pid in (1, 2):
                        if pid not in seen_pids:
                            on, smoothed = smoother.update(pid, 0.0)
                            last_flags[pid] = on
                            last_probs[pid] = smoothed

                t_sec = frame_idx / max(1e-6, eff_fps)
                runtime_fps = (frame_idx + 1) / max(1e-6, time.perf_counter() - started)
                render_annotated_frame(
                    frame,
                    last_assigned,
                    split_step_flags=last_flags,
                    split_probs=last_probs,
                    frame_idx=frame_idx,
                    time_seconds=t_sec,
                    runtime_fps=runtime_fps,
                    show_hud=cfg.pipeline.draw_hud,
                    show_tracker_ids=cfg.tracking.show_tracker_ids,
                    tracking_mode=cfg.tracking.mode,
                )
                writer.write(frame)

                n_processed += 1
                if n_processed % 200 == 0:
                    summary = assigner.slot_summary()
                    logger.info(
                        f"frame {frame_idx}  ({runtime_fps:5.1f} FPS)  "
                        f"P1_split={last_flags.get(1)} P2_split={last_flags.get(2)}  "
                        f"slots={summary}"
                    )

    elapsed = time.perf_counter() - started
    logger.info(f"Done in {elapsed:.1f}s. Annotated -> {output}")


# -------------------------------------------------------------------- #
# convert-cvat
# -------------------------------------------------------------------- #
@app.command("convert-cvat")
def convert_cvat(
    auto: bool = typer.Option(
        False,
        "--auto",
        is_flag=True,
        help=(
            "Auto-pair every video in --raw-dir with its matching CVAT export in "
            "--cvat-dir (e.g. rally_001.mp4 <-> rally_001_cvat.zip)."
        ),
    ),
    raw_dir: Path = typer.Option(Path("data/raw"), "--raw-dir", help="Directory of source videos (used by --auto)."),
    cvat_dir: Path = typer.Option(Path("data/cvat"), "--cvat-dir", help="Directory of CVAT zip/xml exports (used by --auto)."),
    suffix: str = typer.Option("_cvat", "--suffix", help="Suffix on CVAT files relative to video stem (e.g. '_cvat')."),
    video: Optional[Path] = typer.Option(None, "--video", exists=True, dir_okay=False, help="Single source video."),
    cvat: Optional[Path] = typer.Option(
        None,
        "--cvat",
        exists=True,
        dir_okay=False,
        help="Single CVAT export — accepts either .zip (auto-extracted) or .xml.",
    ),
    out: Path = typer.Option(Path("data"), "--out", help="Output root. YOLO -> <out>/yolo, action -> <out>/action."),
    mode: str = typer.Option(
        "both",
        "--mode",
        help="'yolo' for detection dataset, 'action' for split-step clips, 'both' to do both.",
    ),
    config_path: Path = typer.Option(Path("config.yaml"), "--config"),
    clip_len: Optional[int] = typer.Option(None, "--clip-len"),
    val_split: Optional[float] = typer.Option(None, "--val-split"),
    test_split: Optional[float] = typer.Option(
        None,
        "--test-split",
        help="Held-out test fraction (train share is implicit: 1 - val - test).",
    ),
    every_n: Optional[int] = typer.Option(None, "--every-n", help="YOLO: subsample every Nth labeled frame."),
    no_group_split: bool = typer.Option(
        False,
        "--no-group-split",
        help="Split clips/frames randomly within each video instead of by whole video.",
    ),
) -> None:
    """Convert CVAT for Video 1.1 exports into YOLO and/or action datasets.

    Two ways to use this command:

    \b
      # 1) Auto mode — drop files into the standard layout, then:
      data/raw/rally_001.mp4
      data/cvat/rally_001_cvat.zip
      python main.py convert-cvat --auto

    \b
      # 2) Explicit single pair (zip OR xml accepted):
      python main.py convert-cvat \
          --video data/raw/rally_001.mp4 \
          --cvat  data/cvat/rally_001_cvat.zip
    """
    cfg = load_config(config_path)
    setup_logging(cfg.log_level)

    from src.data.cvat_converter import (
        CvatJob,
        convert_cvat as run_convert,
        discover_jobs,
        find_cvat_for_video,
        parse_cvat_xml,
    )

    mode = mode.lower().strip()
    if mode not in {"yolo", "action", "both"}:
        raise typer.BadParameter(f"--mode must be one of yolo|action|both (got '{mode}')")

    if auto:
        if video is not None or cvat is not None:
            raise typer.BadParameter(
                "--auto cannot be combined with --video/--cvat. "
                "Use one mode or the other."
            )
        jobs = discover_jobs(raw_dir, cvat_dir, suffix=suffix)
    else:
        if video is None:
            raise typer.BadParameter(
                "Provide --video and --cvat, or pass --auto.\n\n"
                "Common mistake: there must be a SPACE between main.py and convert-cvat:\n"
                "  python main.py convert-cvat --auto\n\n"
                "Or use the wrapper script (no sub-command needed):\n"
                "  python scripts/convert_cvat.py --auto"
            )
        # Allow the user to skip --cvat and let us look it up by stem.
        if cvat is None:
            located = find_cvat_for_video(video, cvat_dir, suffix=suffix)
            if located is None:
                raise typer.BadParameter(
                    f"--cvat not given and no CVAT export found in {cvat_dir} "
                    f"matching '{video.stem}{suffix}.zip|.xml'."
                )
            cvat = located
        annotations = parse_cvat_xml(cvat)
        jobs = [CvatJob(video_path=video, annotations=annotations)]

    run_convert(
        jobs,
        output_root=out,
        do_yolo=mode in {"yolo", "both"},
        do_action=mode in {"action", "both"},
        val_split=val_split if val_split is not None else cfg.cvat.val_split,
        test_split=test_split if test_split is not None else cfg.cvat.test_split,
        every_n_frames=every_n if every_n is not None else cfg.cvat.every_n_frames,
        yolo_class_name=cfg.cvat.yolo_class_name,
        clip_length=clip_len if clip_len is not None else cfg.action.clip_length,
        clip_stride=cfg.action.clip_stride,
        crop_size=cfg.action.input_size,
        player1_label=cfg.cvat.player1_label,
        player2_label=cfg.cvat.player2_label,
        split_attribute=cfg.cvat.split_attribute,
        seed=cfg.seed,
        group_split=cfg.cvat.group_split and not no_group_split,
    )


# -------------------------------------------------------------------- #
# train-yolo
# -------------------------------------------------------------------- #
@app.command("train-yolo")
def train_yolo(
    data: Path = typer.Option(..., "--data", exists=True, dir_okay=False, help="data.yaml from convert-cvat."),
    config_path: Path = typer.Option(Path("config.yaml"), "--config"),
    base_model: Optional[str] = typer.Option(None, "--base-model"),
    epochs: Optional[int] = typer.Option(None, "--epochs"),
    imgsz: Optional[int] = typer.Option(None, "--imgsz"),
    batch: Optional[int] = typer.Option(None, "--batch"),
    device: Optional[str] = typer.Option(None, "--device"),
) -> None:
    """Fine-tune the YOLO player detector on a CVAT-derived dataset."""
    cfg = load_config(config_path)
    setup_logging(cfg.log_level)
    if device:
        cfg.device = device  # type: ignore[assignment]
    resolved_device = resolve_device(cfg.device)

    from ultralytics import YOLO

    from src.utils.model_runs import (
        YOLO_KIND,
        allocate_run_dir,
        project_relative,
        utc_now_iso,
        write_run_info,
    )
    from src.utils.yolo_weights import (
        MODELS_DIR,
        promote_finetuned_best,
        remove_stray_root_weight,
        resolve_yolo_weight,
        ultralytics_weights_cwd,
    )

    base_spec = base_model or cfg.train_yolo.base_model
    base_path = resolve_yolo_weight(base_spec)
    save_dir, run = allocate_run_dir(YOLO_KIND)
    # Resolve before ultralytics_weights_cwd chdirs into models/.
    data_path = data.resolve()
    save_dir = save_dir.resolve()
    n_epochs = epochs or cfg.train_yolo.epochs
    n_imgsz = imgsz or cfg.train_yolo.imgsz
    n_batch = batch or cfg.train_yolo.batch
    logger = get_logger("train_yolo")
    logger.info(f"Training YOLO run {run} from base '{base_path}' on {data_path}")
    logger.info(f"Saving run artifacts to {save_dir}")

    load_target = str(base_path) if base_path.is_file() else base_spec
    with ultralytics_weights_cwd(MODELS_DIR.resolve()):
        model = YOLO(load_target)
        remove_stray_root_weight(Path(load_target).name, keep=base_path if base_path.is_file() else None)
        results = model.train(
            data=str(data_path),
            epochs=n_epochs,
            imgsz=n_imgsz,
            batch=n_batch,
            device=resolved_device,
            save_dir=str(save_dir),
            exist_ok=True,
        )
    remove_stray_root_weight("yolo26n.pt", keep=MODELS_DIR / "yolo26n.pt")
    remove_stray_root_weight("yolo26n-cls.pt", keep=MODELS_DIR / "yolo26n-cls.pt")

    best_src = save_dir / "weights" / "best.pt"
    promoted = promote_finetuned_best(save_dir)
    if promoted:
        logger.info(f"Fine-tuned detector for analyze: {promoted}")
    write_run_info(
        save_dir,
        {
            "run": run,
            "kind": YOLO_KIND,
            "created_at": utc_now_iso(),
            "epochs": n_epochs,
            "imgsz": n_imgsz,
            "batch": n_batch,
            "data": project_relative(data_path),
            "base_model": project_relative(base_path),
            "checkpoint": project_relative(promoted or best_src),
        },
    )
    logger.info(
        f"YOLO training complete (run {run}). "
        f"Analyze with: --yolo-run {run}"
    )


# -------------------------------------------------------------------- #
# train-action
# -------------------------------------------------------------------- #
@app.command("train-action")
def train_action_cmd(
    manifest: Path = typer.Option(..., "--manifest", exists=True, dir_okay=False, help="manifest.csv from convert-cvat."),
    config_path: Path = typer.Option(Path("config.yaml"), "--config"),
    epochs: Optional[int] = typer.Option(None, "--epochs"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size"),
    lr: Optional[float] = typer.Option(None, "--lr"),
    device: Optional[str] = typer.Option(None, "--device"),
    no_pretrained: bool = typer.Option(
        False, "--no-pretrained", help="Disable best-effort ImageNet weight load."
    ),
) -> None:
    """Train the split-step CNN-LSTM on the action dataset."""
    cfg = load_config(config_path)
    setup_logging(cfg.log_level)
    if device:
        cfg.device = device  # type: ignore[assignment]
    if epochs is not None:
        cfg.train_action.epochs = epochs
    if batch_size is not None:
        cfg.train_action.batch_size = batch_size
    if lr is not None:
        cfg.train_action.lr = lr
    resolved_device = resolve_device(cfg.device)

    from src.action.train import train as run_train
    from src.utils.model_runs import (
        ACTION_KIND,
        allocate_run_dir,
        project_relative,
        utc_now_iso,
        write_run_info,
    )

    run_dir, run = allocate_run_dir(ACTION_KIND)
    cfg.train_action.output_dir = str(run_dir)
    logger = get_logger("train_action")
    logger.info(f"Training action run {run} -> {run_dir}")

    result = run_train(
        manifest=manifest,
        action_cfg=cfg.action,
        train_cfg=cfg.train_action,
        device=resolved_device,
        seed=cfg.seed,
        pretrained_imagenet=not no_pretrained,
    )
    write_run_info(
        run_dir,
        {
            "run": run,
            "kind": ACTION_KIND,
            "created_at": utc_now_iso(),
            "epochs": cfg.train_action.epochs,
            "batch_size": cfg.train_action.batch_size,
            "lr": cfg.train_action.lr,
            "weight_decay": cfg.train_action.weight_decay,
            "loss": cfg.train_action.loss,
            "class_weight_balance": cfg.train_action.class_weight_balance,
            "manifest": project_relative(manifest),
            "best_val_f1": result.best_val_f1,
            "best_val_acc": result.best_val_acc,
            "best_epoch": result.best_epoch,
            "best_metric": result.best_metric,
            "best_metric_value": result.best_metric_value,
            "early_stopping_metric": cfg.train_action.early_stopping_metric,
            "best_threshold": result.best_threshold,
            "test_f1": result.test_f1,
            "test_acc": result.test_acc,
            "checkpoint": project_relative(result.checkpoint_path),
        },
    )
    logger.info(
        f"Action training complete (run {run}). "
        f"Analyze with: --action-run {run}"
    )


# -------------------------------------------------------------------- #
# plot-training
# -------------------------------------------------------------------- #
@app.command("plot-training")
def plot_training_cmd(
    action_run: Optional[int] = typer.Option(
        None,
        "--action-run",
        min=1,
        help="Use models/action_player_<N>/ (overrides default --history/--out).",
    ),
    history: Optional[Path] = typer.Option(
        None,
        "--history",
        exists=True,
        dir_okay=False,
        help="train_history.json from train-action.",
    ),
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        help="Directory for PNG output (defaults to the history file's parent).",
    ),
    test_metrics: Optional[Path] = typer.Option(
        None,
        "--test-metrics",
        exists=True,
        dir_okay=False,
        help="Optional test_metrics.json (defaults to <out>/test_metrics.json).",
    ),
) -> None:
    """Regenerate training curve PNGs from train_history.json."""
    setup_logging("INFO")
    from src.action.plots import save_training_plots_from_files
    from src.utils.model_runs import MODELS_DIR, ACTION_KIND

    if action_run is not None:
        run_dir = MODELS_DIR / f"{ACTION_KIND}_{action_run}"
        history = history or (run_dir / "train_history.json")
        out = out or run_dir
    else:
        history = history or Path("models/action_player/train_history.json")
        out = out or history.parent

    paths = save_training_plots_from_files(
        history_path=history,
        output_dir=out,
        test_metrics_path=test_metrics or (out / "test_metrics.json"),
    )
    if not paths:
        raise typer.Exit(code=1)
    for p in paths:
        typer.echo(str(p))


# -------------------------------------------------------------------- #
# info
# -------------------------------------------------------------------- #
@app.command()
def info(
    config_path: Path = typer.Option(Path("config.yaml"), "--config"),
) -> None:
    """Print resolved config + environment summary."""
    cfg = load_config(config_path)
    setup_logging(cfg.log_level)
    import torch

    typer.echo("=== Environment ===")
    typer.echo(f"torch: {torch.__version__}  cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        typer.echo(f"cuda_device: {torch.cuda.get_device_name(0)}")
    typer.echo(f"mps_available: {mps_available()}")
    typer.echo(f"resolved device: {resolve_device(cfg.device)}")

    from src.utils.model_runs import (
        ACTION_KIND,
        YOLO_KIND,
        existing_run_numbers,
        latest_action_run,
        latest_yolo_run,
    )

    typer.echo("\n=== Detection ===")
    typer.echo(f"  finetuned: {cfg.detection.finetuned_model_path} (exists={Path(cfg.detection.finetuned_model_path).exists()})")
    typer.echo(f"  stock:     {cfg.detection.model_path} (exists={Path(cfg.detection.model_path).exists()})")
    latest_yolo = latest_yolo_run()
    n_yolo = len(existing_run_numbers(YOLO_KIND))
    typer.echo(f"  latest yolo run: {latest_yolo if latest_yolo is not None else '—'}  ({n_yolo} total)")

    typer.echo("\n=== Action ===")
    typer.echo(f"  checkpoint: {cfg.action.model_checkpoint} (exists={Path(cfg.action.model_checkpoint).exists()})")
    typer.echo(f"  clip_length={cfg.action.clip_length}  input_size={cfg.action.input_size}")
    latest_action = latest_action_run()
    n_action = len(existing_run_numbers(ACTION_KIND))
    typer.echo(f"  latest action run: {latest_action if latest_action is not None else '—'}  ({n_action} total)")

    from src.tracking.tracker import resolve_tracker_yaml

    typer.echo("\n=== Tracking ===")
    typer.echo(f"  mode={cfg.tracking.mode}")
    typer.echo(
        f"  tracker_yaml={resolve_tracker_yaml(cfg.tracking.mode, cfg.tracking.tracker_yaml)}"
    )
    typer.echo(f"  player1_position={cfg.assignment.player1_position}")
    typer.echo(
        f"  bbox_smoothing_alpha={cfg.assignment.bbox_smoothing_alpha}  "
        f"predict_max_frames={cfg.assignment.predict_max_frames}"
    )

    typer.echo("\n=== Pipeline ===")
    typer.echo(f"  frame_skip={cfg.pipeline.frame_skip}  draw_hud={cfg.pipeline.draw_hud}")


if __name__ == "__main__":
    app()
