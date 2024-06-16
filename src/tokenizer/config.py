"""Defines the configuration for the Strata tokenizer."""

from dataclasses import dataclass


@dataclass
class StrataTokenizerConfig:
    """Configuration for the Strata tokenizer."""

    vocabulary_size: int = 86400  # number of unique tokens
