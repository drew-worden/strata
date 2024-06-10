"""Defines the Statra model."""

from torch import nn

from src.train.block import StrataBlock
from src.train.config import StrataConfig


class Strata:
    """Strata model definition."""

    def __init__(self: "Strata", config: StrataConfig) -> None:
        """Initialize the Strata model."""
        super().__init__()
        self.config = config
        self.transformer = nn.Module(
            token_embedding_weights=nn.Embedding(config.vocabulary_size, config.num_embedding_dim),
            position_embedding_weights=nn.Embedding(config.block_size, config.num_embedding_dim),
            hidden_layers=nn.ModuleList([StrataBlock(config) for _ in range(config.num_layers)]),
            layer_norm_func=nn.LayerNorm(config.num_embedding_dim),
        )
        self.lang_model_head = nn.Linear(
            config.num_embedding_dim, config.vocabulary_size, bias=False
        )
