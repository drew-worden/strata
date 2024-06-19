"""Defines the configuration for the Strata tokenizer."""

import typing
from dataclasses import dataclass


@dataclass
class StrataTokenizerConfig:
    """Configuration for the Strata tokenizer."""

    vocab_size: int = 86400  # number of unique tokens
    pattern: str = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]
    ++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""  # regex pattern for tokenization
    special_tokens: typing.ClassVar = {
        "<|endoftext|>": 100257,
        "<|fim_prefix|>": 100258,
        "<|fim_middle|>": 100259,
        "<|fim_suffix|>": 100260,
        "<|endofprompt|>": 100276,
    }  # special tokens
    dataset_checkpoint_path: str = (
        "./tokenizer-trainer-checkpoint.json"  # path to save the dataset checkpoint
    )
