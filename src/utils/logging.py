"""Logging helpers.

Uses ``rich`` for nice console output; gracefully degrades to ``logging``'s
default handler if rich is unavailable.
"""

from __future__ import annotations

import logging
from typing import Optional


_LOGGER_NAME = "splitstep"


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure the package-wide logger.

    Idempotent: repeated calls won't stack handlers.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level.upper())
    if logger.handlers:
        return logger

    try:
        from rich.logging import RichHandler

        handler: logging.Handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
            markup=False,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
    except ImportError:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )

    logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a child logger under the package namespace."""
    if name is None:
        return logging.getLogger(_LOGGER_NAME)
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")
