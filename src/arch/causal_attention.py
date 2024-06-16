"""File contains the implementation of the StrataCasualAttention class."""

from torch import Tensor, nn, ones, tril

from src.arch.config import StrataModelConfig


class StrataCausalAttention(nn.Module):
    """StrataCausalAttention class is a causal attention mechanism used in the Strata model."""

    def __init__(self: "StrataCausalAttention", config: StrataModelConfig) -> None:
        """Initialize the StrataCausalAttention class."""
        super().__init__()

        # Check if the number of embedding dimensions is divisible by the number of heads.
        if (config.num_embedding_dim % config.num_head) != 0:
            error_msg = """
            The number of embedding dimensions must be divisible by the number of heads.
            """
            raise ValueError(error_msg)

        self.causal_attention = nn.Linear(config.num_embedding_dim, 3 * config.num_embedding_dim)
        self.causal_projection = nn.Linear(config.num_embedding_dim, config.num_embedding_dim)

        self.num_heads = config.num_head
        self.num_embedding_dim = config.num_embedding_dim

        # Not really a bias.
        # We use it to mask out the upper triangular part of the attention matrix.
        self.register_buffer(
            "bias",
            tril(ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self: "StrataCausalAttention", x: Tensor) -> Tensor:
        """Forward pass of the StrataCausalAttention class."""
        # Batch size, sequence length, embedding dimension.
        b, t, c = x.size()

        # Project the input to the query, key, and value.
        query_key_value = self.causal_attention(x)
        query, key, value = query_key_value.split(self.num_embedding_dim, dim=2)
        key = key.view(b, t, self.num_heads, c // self.num_heads).transpose(1, 2)
        query = query.view(b, t, self.num_heads, c // self.num_heads).transpose(1, 2)
        value = value.view(b, t, self.num_heads, c // self.num_heads).transpose(1, 2)

        # Flash attention
        y = nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True)

        # Apply the attention scores to the value.
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.causal_projection(y)
