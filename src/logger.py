"""Logger module for logging messages to the console and to a file."""

from __future__ import annotations

import logging
import os
import sys
from logging import Logger
from pathlib import Path


def setup_logger(name: str, class_file: str, level: type[int | str] = logging.INFO) -> Logger:
    """Define logger with the given name, file, and level."""
    if not Path("logs").exists():
        Path("logs").mkdir()

    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(class_file, encoding="utf-8")
    global_handler = logging.FileHandler("logs/global.log", encoding="utf-8")

    # Create formatters
    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    global_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(global_handler)

    return logger


def get_relative_path(full_path: str) -> str:
    """Get the relative path of a file."""
    normalized_path = os.path.normpath(full_path)
    position = normalized_path.find(os.sep + "strata" + os.sep)
    return normalized_path[position + 1 :] if position != -1 else normalized_path


class StrataLogger:
    """Logger class for logging."""

    def __init__(self: StrataLogger, class_name: str) -> None:
        """Initialize the Logger class."""
        self.logger = setup_logger(f"{class_name}.class", f"logs/{class_name}.log")
