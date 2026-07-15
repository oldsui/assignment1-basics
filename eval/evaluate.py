import torch
import numpy as np
from cs336_basics.training_loop import get_device, load_tokenized_dataset
from cs336_basics.language_model import TransformerLM
from cs336_basics.utils import cross_entropy, load_checkpoint

# Load checkpoint
device = get_device()
model = TransformerLM(
    vocab_size=10_000,
    context_length=256,
    d_model=64,
    num_layers=3,
    num_heads=4,
    d_ff=128,
    theta=10_000.0,
).to(device)

load_checkpoint("checkpoints/checkpoint-iter1000.pt", model, optimizer=None)
model.eval()

# Load validation data
_, validation_data = load_tokenized_dataset("data/tinystories_small")

# Compute loss on validation set
with torch.no_grad():
    all_losses = []
    for i in range(0, len(validation_data) - 256, 256):
        batch = torch.from_numpy(np.array(validation_data[i:i+256], copy=True)).long().to(device)
        inputs, targets = batch[:-1].unsqueeze(0), batch[1:].view(-1)
        
        logits = model(inputs)
        loss = cross_entropy(logits.view(-1, 10_000), targets)
        all_losses.append(loss.item())
    
    avg_loss = np.mean(all_losses)
    perplexity = np.exp(avg_loss)
    print(f"Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
