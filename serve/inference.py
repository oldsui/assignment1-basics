import numpy as np
import torch

from cs336_basics.language_model import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.training_loop import get_device, load_tokenized_dataset
from cs336_basics.utils import cross_entropy, load_checkpoint


PROMPTS = [
    "Once upon a time",
    "The little girl",
    "A brave knight",
]


def build_model(device: torch.device) -> TransformerLM:
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
    return model


def build_tokenizer() -> Tokenizer:
    return Tokenizer.from_files(
        vocab_filepath="data/tinystories_small/vocab.pkl",
        merges_filepath="data/tinystories_small/merges.pkl",
        special_tokens=["<|endoftext|>"],
    )


def evaluate_model(model: TransformerLM, device: torch.device) -> None:
    _, validation_data = load_tokenized_dataset("data/tinystories_small")

    with torch.no_grad():
        all_losses = []
        for i in range(0, len(validation_data) - 256, 256):
            batch = torch.from_numpy(np.array(validation_data[i:i + 256], copy=True)).long().to(device)
            inputs, targets = batch[:-1].unsqueeze(0), batch[1:].view(-1)

            logits = model(inputs)
            loss = cross_entropy(logits.view(-1, 10_000), targets)
            all_losses.append(loss.item())

        avg_loss = np.mean(all_losses)
        perplexity = np.exp(avg_loss)
        print(f"Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")


def generate_text(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 40,
    device: torch.device | None = None,
) -> str:
    if device is None:
        device = next(model.parameters()).device

    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        return prompt

    if len(token_ids) > model.context_length:
        token_ids = token_ids[-model.context_length:]

    generated_ids = list(token_ids)
    end_token_id = tokenizer.byte_to_id.get("<|endoftext|>".encode("utf-8"))

    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_ids = torch.tensor([generated_ids[-model.context_length:]], device=device, dtype=torch.long)
            logits = model(input_ids)
            next_token = logits[0, -1].argmax().item()
            generated_ids.append(next_token)
            if end_token_id is not None and next_token == end_token_id:
                break

    return tokenizer.decode(generated_ids)


def main() -> None:
    device = get_device()
    model = build_model(device)
    tokenizer = build_tokenizer()

    evaluate_model(model, device)

    for prompt in PROMPTS:
        completion = generate_text(model, tokenizer, prompt, max_new_tokens=30, device=device)
        print(f"Prompt: {prompt}")
        print(f"Completion: {completion}")
        print()


if __name__ == "__main__":
    main()