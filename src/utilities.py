"""Utility functions for the project."""

import time
from typing import Literal

import torch

from src.logger import setup_logger

logger = setup_logger("StrataUtilities", logger_type="class")


class StrataUtilities:
    """Utility functions for the Strata project."""

    def __init__(self: "StrataUtilities") -> None:
        """Initialize the StrataUtilities class."""
        self.timers = {}

    def get_best_device(self: "StrataUtilities") -> Literal["cpu", "cuda", "mps"]:
        """Return the best device available (CPU, CUDA, MPS)."""
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        logger.info(f"Using device: {device.upper()}")
        return device

    def start_timer(self: "StrataUtilities", timer_id: str) -> None:
        """Start a timer with the given ID."""
        if timer_id in self.timers:
            error_message = f"Timer with ID {timer_id} already exists."
            raise ValueError(error_message)
        self.timers[timer_id] = time.time()

    def end_timer(self: "StrataUtilities", timer_id: str) -> None:
        """End the timer with the given ID and print the elapsed time."""
        if timer_id not in self.timers:
            error_message = f"Timer with ID {timer_id} does not exist."
            raise ValueError(error_message)
        elapsed_time = time.time() - self.timers[timer_id]
        logger.info(f"Elapsed time for {timer_id}: {elapsed_time}s")
        del self.timers[timer_id]
