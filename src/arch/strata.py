"""Defines the Strata model."""

from torch import Tensor, arange, long, nn

from src.arch.block import StrataBlock
from src.arch.config import StrataConfig
from src.logger import setup_logger

logger = setup_logger("Strata", logger_type="class")


class Strata(nn.Module):
    """Strata model definition."""

    def __init__(self: "Strata", config: StrataConfig) -> None:
        """Initialize the Strata model."""
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "token_embedding_weights": nn.Embedding(
                    config.vocabulary_size, config.num_embedding_dim
                ),
                "position_embedding_weights": nn.Embedding(
                    config.block_size, config.num_embedding_dim
                ),
                "hidden_layers": nn.ModuleList(
                    [StrataBlock(config) for _ in range(config.num_layers)]
                ),
                "layer_norm_func": nn.LayerNorm(config.num_embedding_dim),
            }
        )
        self.lang_model_head = nn.Linear(
            config.num_embedding_dim, config.vocabulary_size, bias=False
        )

        self.transformer.token_embedding_weights.weight = self.lang_model_head.weight
        self.apply(self.init_weights)

    def init_weights(self: "Strata", module: nn.Module) -> None:
        """Initialize the weights of the Strata model."""
        std = 0.02 * (2 * self.config.num_layers) ** -0.5
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=std)

    def forward(self: "Strata", idx: Tensor, targets: Tensor = None) -> (Tensor, Tensor):
        """Forward pass of the Strata model."""
        b, t = idx.size()
        if t > self.config.block_size:
            error_msg = "Input sequence length exceeds the block size."
            raise ValueError(error_msg)

        positions = arange(0, t, dtype=long, device=idx.device)
        position_embeddings = self.transformer.position_embedding_weights(positions)
        token_embeddings = self.transformer.token_embedding_weights(idx)

        x = token_embeddings + position_embeddings

        for block in self.transformer.hidden_layers:
            x = block(x)

        x = self.transformer.layer_norm_func(x)

        logits = self.lang_model_head(x)

        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

        return logits
