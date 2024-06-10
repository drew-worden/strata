"""File contains the implementation of the StrataBlock class."""

from torch import Tensor, nn

from src.arch.causal_attention import StrataCausalAttention
from src.arch.config import StrataConfig
from src.arch.feed_forward_network import StrataFeedForwardNetwork


class StrataBlock(nn.Module):
    """StrataBlock class is a block that is used in the Strata model."""

    def __init__(self: "StrataBlock", config: StrataConfig) -> None:
        """Initialize the StrataBlock class."""
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.num_embedding_dim)
        self.attention = StrataCausalAttention(config)
        self.layer_norm_2 = nn.LayerNorm(config.num_embedding_dim)
        self.feed_forward_network = StrataFeedForwardNetwork(config)

    def forward(self: "StrataBlock", x: Tensor) -> Tensor:
        """Forward pass of the StrataBlock class. Effectively performs map reduce operation."""
        x = x + self.attention(self.layer_norm_1(x))
        return x + self.feed_forward_network(self.layer_norm_2(x))
