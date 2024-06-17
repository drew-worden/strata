"""Defines the Strata tokenizer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import regex

from src.logger import setup_logger

if TYPE_CHECKING:
    from src.tokenizer.config import StrataTokenizerConfig

logger = setup_logger("StrataTokenizer", logger_type="class")


class StrataTokenizer:
    """Strata tokenizer definition."""

    def __init__(self: StrataTokenizer, config: StrataTokenizerConfig) -> None:
        """Initialize the Strata tokenizer."""
        self.config = config
        self.merges = {}
        self.vocab = self.create_vocab()
        self.pattern = regex.compile(self.config.pattern)

    def create_vocab(self: StrataTokenizer) -> dict[int, bytes]:
        """Create the vocabulary."""
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.config.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def encode(self: StrataTokenizer, text: str) -> None:
        """Encode text into tokens."""

    def decode(self: StrataTokenizer, tokens: list[str]) -> str:
        """Decode tokens into text."""

    def train(self: StrataTokenizer, text: str) -> None:
        """Train the tokenizer."""
        min_vocab_size = 256
        if self.config.vocab_size < min_vocab_size:
            error_message = "Vocabulary size must be at least 256."
            raise ValueError(error_message)

        num_merges = self.config.vocab_size - min_vocab_size
        text_chunks = regex.findall(self.pattern, text)
        ids = [list(c.encode("utf-8")) for c in text_chunks]

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            counts = {}
            for chunk_ids in ids:
                counts = self.get_counts(chunk_ids, counts)
            pair = max(counts, key=counts.get)
            idx = 256 + i
            ids = [self.merge(chunk_ids, pair, idx) for chunk_ids in ids]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            logger.info(
                f"""Merge {i + 1}/{num_merges}: {pair} -> {idx} ({vocab[idx]})
                had {counts[pair]} occurrences."""
            )

    def save(self: StrataTokenizer, prefix: str) -> None:
        """Save the tokenizer to disk."""

    def load(self: StrataTokenizer, path: str) -> None:
        """Load the tokenizer from disk."""

    def merge(self: StrataTokenizer, ids: list, pair: tuple, idx: int) -> list:
        """Merge a pair of tokens."""
        new_ids = []
        i = 0
        while i < len(ids):
            if ids[i] == pair[0] and i < len(ids) - 1 and ids[i + 1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def get_counts(self: StrataTokenizer, ids: [int], counts: dict | None = None) -> dict:
        """Get the counts of pairs of tokens."""
        counts = {} if counts is None else counts
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
