"""File contains the implementation of the StrataFeedForwardNetwork class."""

from torch import nn

from src.arch.config import StrataConfig


class StrataFeedForwardNetwork(nn.Module):
    """StrataFeedForwardNetwork class is a feed forward network that is used in the Strata model."""

    def __init__(self: "StrataFeedForwardNetwork", config: StrataConfig) -> None:
        """Initialize the StrataFeedForwardNetwork class."""
        super().__init__()

        # Using GELU as apposed to RELU for the contribution of a local gradient.
        # Might want to try without tanh to turn off the approximation calculation.
        self.linear_projection_1 = nn.Linear(config.num_embedding_dim, config.num_embedding_dim * 4)
        self.gelu = nn.GELU(approximate="tanh")
        self.linear_projection_2 = nn.Linear(config.num_embedding_dim * 4, config.num_embedding_dim)

    def forward(self: "StrataFeedForwardNetwork", x: nn.Tensor) -> nn.Tensor:
        """Forward pass of the StrataFeedForwardNetwork class."""
        x = self.linear_projection_1(x)
        x = self.gelu(x)
        return self.linear_projection_2(x)
