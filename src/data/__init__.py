"""Dataset I/O: CVAT parsing/conversion and frame readers."""

__all__ = [
    "VideoReader",
    "VideoWriter",
    "effective_fps",
    "iter_frames_at_fps",
    "CvatAnnotations",
    "CvatBox",
    "CvatJob",
    "CvatTrack",
    "DEFAULT_CVAT_SUFFIX",
    "auto_convert",
    "convert_cvat",
    "discover_jobs",
    "export_action_dataset",
    "export_yolo_detection",
    "find_cvat_for_video",
    "find_video_for_cvat",
    "pair_videos_and_cvat",
    "parse_cvat_xml",
]


def __getattr__(name: str):
    """Lazily import video/CVAT helpers to avoid native imports for pure utilities."""
    if name in {"VideoReader", "VideoWriter", "effective_fps", "iter_frames_at_fps"}:
        from . import video_io

        return getattr(video_io, name)
    if name in {
        "CvatAnnotations",
        "CvatBox",
        "CvatJob",
        "CvatTrack",
        "DEFAULT_CVAT_SUFFIX",
        "auto_convert",
        "convert_cvat",
        "discover_jobs",
        "export_action_dataset",
        "export_yolo_detection",
        "find_cvat_for_video",
        "find_video_for_cvat",
        "pair_videos_and_cvat",
        "parse_cvat_xml",
    }:
        from . import cvat_converter

        return getattr(cvat_converter, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
