"""Test the Strata model by generating text using a completion prompt."""

import sys

import tiktoken
import torch

from src.arch.config import StrataConfig
from src.arch.strata import Strata
from src.data_loader import StrataDataLoader
from src.logger import get_relative_path, setup_logger
from src.utilities import StrataUtilities

# Setup logger using filename
logger = setup_logger(get_relative_path(__file__), logger_type="script")

# Get utils
utils = StrataUtilities()

# Get the best device available
device = utils.get_best_device()

torch.set_float32_matmul_precision("high")

config = StrataConfig()
data_loader = StrataDataLoader(8, 32)
model = Strata(config)
model.to(device)
model = torch.compile(model)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

for i in range(50):
    inputs, targets = data_loader.next_batch()
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(inputs, targets)
    loss.backward()
    optimizer.step()
    logger.info(f"Step {i + 1}, Loss: {loss.item()}")

sys.exit(0)
# Define the number of return sequences and the maximum length.
num_return_sequences = 5
max_length = 30

# Create a Strata model.

model = Strata(config)
model.to(device)

# Create a tokenizer and convert the input text to tokens.
encoding = tiktoken.get_encoding("gpt2")
tokens = encoding.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)

x = tokens.to(device)

# Generate text using the Strata model.
torch.manual_seed(42)

utils.start_timer("generation")
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
    logger.info(f">{decoded_text}")

utils.end_timer("generation")
