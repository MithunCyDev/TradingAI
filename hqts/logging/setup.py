"""
Centralized logging configuration for HQTS.

Provides structured logs for data pulls, predictions, decisions, and executions.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[Path | str] = None,
    format_string: Optional[str] = None,
) -> None:
    """
    Configure root logger with console and optional file output.

    Args:
        level: Logging level (e.g., logging.INFO).
        log_file: Optional path to log file.
        format_string: Optional custom format. Default includes timestamp, level, name, message.
    """
    fmt = format_string or "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.setLevel(level)

    # Remove existing handlers to avoid duplicates
    for h in root.handlers[:]:
        root.removeHandler(h)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
    root.addHandler(console)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
        root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module name."""
    return logging.getLogger(name)
