"""Defines the Strata tokenizer."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.tokenizer.config import StrataTokenizerConfig


class StrataTokenizer:
    """Strata tokenizer definition."""

    def __init__(self: StrataTokenizer, config: StrataTokenizerConfig) -> None:
        """Initialize the Strata tokenizer."""
        self.config = config

    def encode(self: StrataTokenizer, text: str) -> None:
        """Encode text into tokens."""

    def decode(self: StrataTokenizer, tokens: list[str]) -> str:
        """Decode tokens into text."""
