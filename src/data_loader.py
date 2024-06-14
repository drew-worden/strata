"""Data loader for the Strata model."""

from __future__ import annotations

from pathlib import Path

import tiktoken
import torch

from src.logger import setup_logger

# Define the logger.
logger = setup_logger("StrataDataLoader", logger_type="class")


class StrataDataLoader:
    """Data loader for the Strata model."""

    def __init__(self: StrataDataLoader, batch: int, t: int) -> None:
        """Initialize the Strata data loader."""
        self.batch = batch
        self.t = t
        self.batch_time = self.batch * self.t

        with Path("data/shake.txt").open("r") as file:
            text = file.read()

        encoding = tiktoken.get_encoding("gpt2")
        loaded_tokens = encoding.encode(text)
        self.tokens = torch.tensor(loaded_tokens)
        logger.info(f"Loaded {len(self.tokens)} tokens.")
        logger.info(f"Epoch is defined as {len(self.tokens) // self.batch_time} batches.")
        self.position = 0

    def next_batch(self: StrataDataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the next batch of inputs and targets."""
        buffer = self.tokens[self.position : self.position + self.batch_time + 1]
        inputs = (buffer[:-1]).view(self.batch, self.t)
        targets = (buffer[1:]).view(self.batch, self.t)

        self.position += self.batch_time

        if self.position + self.batch_time + 1 > len(self.tokens):
            self.position = 0
        return inputs, targets
