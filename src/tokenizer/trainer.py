"""Tokenizer trainer."""

import json
from pathlib import Path

from datasets import IterableDataset, load_dataset

from src.logger import setup_logger
from src.tokenizer.config import StrataTokenizerConfig
from src.utilities import StrataUtilities

logger = setup_logger("StrataTokenizerTrainer", logger_type="class")
utils = StrataUtilities()


class StrataTokenizerTrainer:
    """Strata tokenizer trainer."""

    def __init__(self: "StrataTokenizerTrainer", config: StrataTokenizerConfig) -> None:
        """Initialize the tokenizer trainer."""
        self.config = config
        self.dataset_name = "HuggingFaceFW/fineweb"

    def get_dataset(self: "StrataTokenizerTrainer") -> IterableDataset:
        """Get the dataset."""
        return load_dataset(
            self.dataset_name, streaming=True, split="train", trust_remote_code=True
        )

    def save_checkpoint(self: "StrataTokenizerTrainer", state_dict: dict) -> None:
        """Save the dataset state to disk."""
        with Path(self.config.dataset_checkpoint_path).open("w") as f:
            f.write(json.dumps(state_dict))

    def load_checkpoint(self: "StrataTokenizerTrainer") -> dict:
        """Load the dataset state from disk."""
        with Path(self.config.dataset_checkpoint_path).open("r") as f:
            return json.loads(f.read())

    def does_checkpoint_exist(self: "StrataTokenizerTrainer") -> bool:
        """Check if the dataset checkpoint exists."""
        return (
            Path(self.config.dataset_checkpoint_path).is_file()
            and Path(self.config.dataset_checkpoint_path).stat().st_size > 0
        )

    def train(self: "StrataTokenizerTrainer") -> None:
        """Train the tokenizer."""
        logger.info(f"Loading dataset [{self.dataset_name}]...")
        dataset = self.get_dataset()

        if self.does_checkpoint_exist():
            logger.info(
                f"Checkpoint found. Loading dataset [{self.dataset_name}] from checkpoint..."
            )
            dataset.load_state_dict(self.load_checkpoint())

        for data in dataset:
            logger.info(data["text"])
            self.save_checkpoint(dataset.state_dict())
            logger.info(f"Checkpoint saved for dataset [{self.dataset_name}].")


tokenizer_trainer = StrataTokenizerTrainer(StrataTokenizerConfig())
tokenizer_trainer.train()
