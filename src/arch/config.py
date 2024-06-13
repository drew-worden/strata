"""Defines the configuration for the Strata model."""

from dataclasses import dataclass


@dataclass
class StrataConfig:
    """Configuration for the Strata model."""

    block_size: int = 1024  # max sequence length
    vocabulary_size: int = 50257  # number of unique tokens
    num_layers: int = 24  # number of layers in the model
    num_head: int = 24  # number of heads in the model
    num_embedding_dim: int = 768  # embedding dimension
