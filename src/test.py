"""Test the Strata model by generating text using a completion prompt."""

import sys

import tiktoken
import torch

from src.arch.config import StrataConfig
from src.arch.strata import Strata

# Define the number of return sequences and the maximum length.
num_return_sequences = 5
max_length = 30

# Create a Strata model.
config = StrataConfig()
model = Strata(config)
model.eval()

# Create a tokenizer and convert the input text to tokens.
encoding = tiktoken.get_encoding("gpt2")
tokens = encoding.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens

# Generate text using the Strata model.
torch.manual_seed(42)

while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        next_tokens = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, next_tokens)
        x = torch.cat((x, xcol), dim=1)

# Decode the generated tokens to text.
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded_text = encoding.decode(tokens)
    sys.stderr.write(f">{decoded_text}\n")
