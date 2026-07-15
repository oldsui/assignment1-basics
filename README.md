# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

### Train BPE and Generate Token Datasets

Before training the language model, you need to tokenize your data using a trained BPE tokenizer. Create a Python script to train BPE on your corpus and encode it to token IDs.

#### Quick Start (Small Dataset)

For a quick experiment with a small subset:

```sh
mkdir -p data/tinystories_small
# Place your TinyStories-valid.txt in this directory
cp /path/to/TinyStories-valid.txt data/tinystories_small/
```

Create `data/tinystories_small/make_dataset.py`:

```python
from pathlib import Path
import pickle
import numpy as np
from cs336_basics.tokenizer import train_bpe, Tokenizer

DATA_DIR = Path(__file__).resolve().parent
corpus_path = DATA_DIR / "TinyStories-valid.txt"

special_tokens = ["<|endoftext|>"]
vocab_size = 10000

# Train BPE tokenizer
vocab, merges = train_bpe(
    input_path=corpus_path,
    vocab_size=vocab_size,
    special_tokens=special_tokens,
)

# Create tokenizer and encode text
tokenizer = Tokenizer(vocab, merges, special_tokens)
text = corpus_path.read_text(encoding="utf-8")
token_ids = np.array(tokenizer.encode(text), dtype=np.int64)

# Split into train/validation (90/10)
split = int(len(token_ids) * 0.9)
train_ids = token_ids[:split]
valid_ids = token_ids[split:]

# Save token arrays
np.save(DATA_DIR / "tinystories_small-train-tokens.npy", train_ids)
np.save(DATA_DIR / "tinystories_small-valid-tokens.npy", valid_ids)

# Save vocab and merges for later use
with open(DATA_DIR / "vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)
with open(DATA_DIR / "merges.pkl", "wb") as f:
    pickle.dump(merges, f)

print("Saved token datasets and BPE artifacts")
```

Run it:

```sh
uv run data/tinystories_small/make_dataset.py
```

#### Full Dataset

For the complete TinyStories dataset, modify the script to use both train and validation splits, then combine them or keep them separate as needed.

### Run Training Loop

After tokenized datasets are ready, train the Transformer language model:

```sh
# Using the small dataset
uv run python -m cs336_basics.training_loop --dataset-dir data/tinystories_small

# With custom hyperparameters
uv run python -m cs336_basics.training_loop \
  --dataset-dir data/tinystories_small \
  --batch-size 32 \
  --total-training-tokens 100000 \
  --lr-max 1e-3 \
  --lr-min 1e-5 \
  --checkpoint-freq 100 \
  --validation-loss-freq 10

# Resume from checkpoint
uv run python -m cs336_basics.training_loop \
  --dataset-dir data/tinystories_small \
  --resume-from checkpoints/checkpoint-iter1000.pt
```

Training will log iteration metrics (loss, validation loss, gradient norm, learning rate) and save checkpoints to `checkpoints/` directory.

