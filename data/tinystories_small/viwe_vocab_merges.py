from pathlib import Path
import pickle

DATA_DIR = Path(__file__).resolve().parent

with open(DATA_DIR / "vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

with open(DATA_DIR / "merges.pkl", "rb") as f:
    merges = pickle.load(f)

print(f"Vocab size: {len(vocab)}")
print(f"Number of merges: {len(merges)}")
print(f"\nFirst 5 vocab entries:")
for token_id, token_bytes in list(vocab.items())[:5]:
    print(f"  {token_id}: {token_bytes}")
print(f"\nFirst 5 merges:")
for i, merge in enumerate(merges[:5]):
    print(f"  {i}: {merge}")