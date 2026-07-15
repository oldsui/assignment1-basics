from pathlib import Path
import pickle
import numpy as np
from cs336_basics.tokenizer import train_bpe, Tokenizer

DATA_DIR = Path(__file__).resolve().parent
corpus_path = DATA_DIR / "TinyStories-valid.txt"

special_tokens = ["<|endoftext|>"]
vocab_size = 10000

vocab, merges = train_bpe(
    input_path=corpus_path,
    vocab_size=vocab_size,
    special_tokens=special_tokens,
)

tokenizer = Tokenizer(vocab, merges, special_tokens)

text = corpus_path.read_text(encoding="utf-8")
token_ids = np.array(tokenizer.encode(text), dtype=np.int64)

split = int(len(token_ids) * 0.9)
train_ids = token_ids[:split]
valid_ids = token_ids[split:]

np.save(DATA_DIR / "tinystories_small-train-tokens.npy", train_ids)
np.save(DATA_DIR / "tinystories_small-valid-tokens.npy", valid_ids)

with open(DATA_DIR / "vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)
with open(DATA_DIR / "merges.pkl", "wb") as f:
    pickle.dump(merges, f)

print("Saved tinystories_small-train-tokens.npy and tinystories_small-valid-tokens.npy")