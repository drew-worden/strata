"""File contains the implementation of the StrataBlock class."""

from torch import Tensor, nn

from src.train.config import StrataConfig


class StrataBlock(nn.Module):
    """StrataBlock class is a block that is used in the Strata model."""

    def __init__(self: "StrataBlock", config: StrataConfig) -> None:
        """Initialize the StrataBlock class."""
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.num_embedding_dim)
        self.attention = StrataCasualAttention(config)
        self.layer_norm_2 = nn.LayerNorm(config.num_embedding_dim)
        self.feed_forward_network = StrataFeedForwardNetwork(config)

    def forward(self: "StrataBlock", x: Tensor) -> Tensor:
        """Forward pass of the StrataBlock class. Effectively performs map reduce operation."""
        x = x + self.attention(self.layer_norm_1(x))
        return x + self.feed_forward_network(self.layer_norm_2(x))
