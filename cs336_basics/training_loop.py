import argparse
import os
import pathlib
import uuid

import numpy as np
import torch

from cs336_basics.adamw import AdamW
from cs336_basics.data_loader import get_batch
from cs336_basics.language_model import TransformerLM
from cs336_basics.utils import (
    cross_entropy,
    cosine_lr_schedule,
    get_device,
    gradient_clipping,
    load_checkpoint,
    save_checkpoint,
)

VOCAB_SIZE = 10_000
CONTEXT_LEN = 256
D_MODEL = 64
NUM_LAYERS = 3
NUM_HEADS = 4
D_FF = 128
ROPE_THETA = 10_000.0

DEFAULT_BATCH_SIZE = 32
DEFAULT_TOTAL_TRAINING_TOKENS = 10_000_000
DEFAULT_LR_COSINE_NEPOCHS = 5
DEFAULT_CHECKPOINT_FREQ = 100
DEFAULT_VALIDATION_LOSS_FREQ = 10
DEFAULT_DATASET_DIR = pathlib.Path(__file__).resolve().parents[1] / "data"


def combined_gradient_norm(parameters) -> float:
    """Compute the combined L2 norm of gradients for a set of parameters."""
    total_norm = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        total_norm += grad.norm(2).item() ** 2
    return float(total_norm ** 0.5)


def compute_train_validation_losses(
    model: TransformerLM,
    training_data: np.ndarray,
    validation_data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: torch.device,
) -> tuple[float, float]:
    """Compute a train loss and validation loss on fresh batches from each split."""
    model.eval()
    with torch.no_grad():
        train_inputs, train_targets = get_batch(
            training_data, batch_size, context_length, str(device)
        )
        valid_inputs, valid_targets = get_batch(
            validation_data, batch_size, context_length, str(device)
        )

        train_logits = model(train_inputs)
        valid_logits = model(valid_inputs)

        vocab_size = train_logits.size(-1)
        train_loss = cross_entropy(train_logits.view(-1, vocab_size), train_targets.view(-1))
        validation_loss = cross_entropy(valid_logits.view(-1, vocab_size), valid_targets.view(-1))

    model.train()
    return float(train_loss.item()), float(validation_loss.item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer language model.")
    parser.add_argument("--dataset-dir", type=pathlib.Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--context-length", type=int, default=CONTEXT_LEN)
    parser.add_argument("--d-model", type=int, default=D_MODEL)
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--num-heads", type=int, default=NUM_HEADS)
    parser.add_argument("--d-ff", type=int, default=D_FF)
    parser.add_argument("--rope-theta", type=float, default=ROPE_THETA)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--total-training-tokens", type=int, default=DEFAULT_TOTAL_TRAINING_TOKENS)
    parser.add_argument("--lr-max", type=float, default=1e-3)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--lr-warmup-iters", type=int, default=100)
    parser.add_argument("--beta-1", type=float, default=0.9)
    parser.add_argument("--beta-2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--lr-cosine-nepochs", type=int, default=DEFAULT_LR_COSINE_NEPOCHS)
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, default=pathlib.Path("checkpoints"))
    parser.add_argument("--checkpoint-freq", type=int, default=DEFAULT_CHECKPOINT_FREQ)
    parser.add_argument("--validation-loss-freq", type=int, default=DEFAULT_VALIDATION_LOSS_FREQ)
    parser.add_argument("--resume-from", type=pathlib.Path, default=None)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_tokenized_dataset(dataset_dir: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    dataset_dir = dataset_dir.resolve()
    base_name = dataset_dir.name
    train_path = dataset_dir / f"{base_name}-train-tokens.npy"
    valid_path = dataset_dir / f"{base_name}-valid-tokens.npy"

    if not train_path.exists() or not valid_path.exists():
        raise FileNotFoundError(
            f"Could not find training or validation token files in {dataset_dir}."
        )

    training_data = np.load(str(train_path), mmap_mode="r")
    validation_data = np.load(str(valid_path), mmap_mode="r")
    return training_data, validation_data


def train(args: argparse.Namespace) -> None:
    device = get_device()
    torch.manual_seed(args.seed)

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.rope_theta,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_max,
        betas=(args.beta_1, args.beta_2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    if args.resume_from is not None:
        if args.resume_from.exists():
            loaded_iter = load_checkpoint(str(args.resume_from), model, optimizer)
            start_iter = int(loaded_iter) + 1
            print(f"Resumed checkpoint {args.resume_from} from iteration {loaded_iter}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {args.resume_from}")
    else:
        start_iter = 1

    training_data, validation_data = load_tokenized_dataset(args.dataset_dir)
    print("Data set ready")

    num_tokens = len(training_data)
    tokens_per_iter = float(args.batch_size) * float(args.context_length)
    iterations_per_epoch = max(1, int(num_tokens / tokens_per_iter))
    cosine_cycle_iters = iterations_per_epoch * args.lr_cosine_nepochs
    max_num_iters = int(float(args.total_training_tokens) / tokens_per_iter)
    training_epochs = float(max_num_iters) / iterations_per_epoch

    print(
        f"Training configured for {max_num_iters} iterations with batch size {args.batch_size} "
        f"and approx. {training_epochs:.2f} epochs."
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    model.train()
    for iteration in range(start_iter, max_num_iters + 1):
        input_tensor, target_tensor = get_batch(
            training_data,
            args.batch_size,
            args.context_length,
            str(device),
        )

        lr_now = cosine_lr_schedule(
            iteration,
            args.lr_max,
            args.lr_min,
            args.lr_warmup_iters,
            cosine_cycle_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_now

        logits = model(input_tensor)
        vocab_size = logits.size(-1)
        loss = cross_entropy(logits.view(-1, vocab_size), target_tensor.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_clipping(model.parameters(), args.max_gradient_norm)
        gradient_norm = combined_gradient_norm(model.parameters())
        optimizer.step()

        if iteration % args.validation_loss_freq == 0 or iteration == start_iter:
            train_loss, validation_loss = compute_train_validation_losses(
                model,
                training_data,
                validation_data,
                args.batch_size,
                args.context_length,
                device,
            )
            log_dict = {
                "iteration": iteration,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "gradient_norm": gradient_norm,
                "learning_rate": lr_now,
            }
            print(log_dict)

        if iteration % args.checkpoint_freq == 0:
            checkpoint_path = args.checkpoint_dir / f"checkpoint-iter{iteration}.pt"
            save_checkpoint(model, optimizer, iteration, str(checkpoint_path))
            print(f"Saved checkpoint to {checkpoint_path}")

    print("Training complete")


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
