"""Quick check of the saved checkpoints to see accuracy progression."""
import torch
import numpy as np
from pathlib import Path
from model import Grokker
from train import make_dataset

p = 113
train_frac = 0.3

rng = np.random.default_rng(42)
data = make_dataset(p)
rng.shuffle(data)
n_train = int(len(data) * train_frac)
train_data, test_data = data[:n_train], data[n_train:]

# sanity check: no overlap between train and test
train_set = set((a, b) for a, b, _ in train_data)
test_set  = set((a, b) for a, b, _ in test_data)
overlap = train_set & test_set
print(f"train: {len(train_data)}  test: {len(test_data)}  overlap: {len(overlap)}")

eq = p
x_train = torch.tensor([[a, b, eq] for a, b, _ in train_data], dtype=torch.long)
y_train = torch.tensor([c for _, _, c in train_data], dtype=torch.long)
x_test  = torch.tensor([[a, b, eq] for a, b, _ in test_data],  dtype=torch.long)
y_test  = torch.tensor([c for _, _, c in test_data],  dtype=torch.long)

for ckpt in sorted(Path("checkpoints").glob("step_*.pt")):
    model = Grokker(p)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        train_acc = (model(x_train).argmax(-1) == y_train).float().mean().item()
        test_acc  = (model(x_test).argmax(-1)  == y_test).float().mean().item()
    print(f"{ckpt.stem}  train {train_acc:.4f}  test {test_acc:.4f}")
