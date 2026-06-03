import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path

from model import Grokker


def make_dataset(p, op="add"):
    if op == "add":
        return [(a, b, (a + b) % p) for a in range(p) for b in range(p)]
    elif op == "mul":
        # exclude pairs with 0 — multiplication by 0 is trivial and doesn't require the circuit
        return [(a, b, (a * b) % p) for a in range(1, p) for b in range(1, p)]
    else:
        raise ValueError(f"unknown op: {op}")


def run(
    p=113,
    d_model=128,
    n_heads=4,
    d_mlp=512,
    op="add",
    train_frac=0.3,
    lr=1e-3,
    weight_decay=0.5,
    steps=50000,
    log_every=50,
    checkpoint_every=1000,
    save_dir="checkpoints",
):
    Path(save_dir).mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    rng = np.random.default_rng(42)
    data = make_dataset(p, op)
    rng.shuffle(data)
    n_train = int(len(data) * train_frac)
    train_data, test_data = data[:n_train], data[n_train:]

    eq = p  # '=' token index

    def to_tensors(subset):
        x = torch.tensor([[a, b, eq] for a, b, _ in subset], dtype=torch.long, device=device)
        y = torch.tensor([c for _, _, c in subset], dtype=torch.long, device=device)
        return x, y

    x_train, y_train = to_tensors(train_data)
    x_test, y_test = to_tensors(test_data)

    model = Grokker(p, d_model, n_heads, d_mlp).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    log = []

    for step in range(steps + 1):
        model.train()
        loss = F.cross_entropy(model(x_train), y_train)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % log_every == 0:
            model.eval()
            with torch.no_grad():
                train_acc = (model(x_train).argmax(-1) == y_train).float().mean().item()
                test_acc = (model(x_test).argmax(-1) == y_test).float().mean().item()
            log.append({"step": step, "loss": round(loss.item(), 4),
                         "train_acc": round(train_acc, 4), "test_acc": round(test_acc, 4)})
            with open(f"{save_dir}/log.json", "w") as f:
                json.dump(log, f)
            if step % 1000 == 0:
                print(f"step {step:6d}  loss {loss.item():.3f}  train {train_acc:.3f}  test {test_acc:.3f}")

        if step % checkpoint_every == 0:
            torch.save(model.state_dict(), f"{save_dir}/step_{step:06d}.pt")

    with open(f"{save_dir}/log.json", "w") as f:
        json.dump(log, f)

    torch.save(model.state_dict(), f"{save_dir}/final.pt")
    print("done")
    return model, log


if __name__ == "__main__":
    import sys
    op = sys.argv[1] if len(sys.argv) > 1 else "add"
    save_dir = f"checkpoints_{op}"
    print(f"op: {op}")
    run(op=op, save_dir=save_dir)
