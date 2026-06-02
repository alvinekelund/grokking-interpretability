"""
Load a trained checkpoint and look at what the model actually learned.

The key finding from Nanda et al. 2023: after grokking, the embedding matrix
is structured along a small set of Fourier frequencies. The model implements
modular addition via trig identities:
    cos(w(a+b)) = cos(wa)cos(wb) - sin(wa)sin(wb)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

from model import Grokker


def load(checkpoint, p=113, **kwargs):
    model = Grokker(p, **kwargs)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    return model


def fourier_power(W, p):
    """
    W: (p, d_model) embedding matrix (rows = token embeddings for 0..p-1)
    Returns: (p,) array of power per frequency (freq 0 = constant, freq k = k-th harmonic)
    """
    # Fourier transform over the token axis
    # fft gives complex coefficients; power = magnitude squared summed over d_model
    F = np.fft.rfft(W, axis=0)  # (p//2+1, d_model)
    power = (np.abs(F) ** 2).sum(axis=1)
    power /= power.sum()
    return power


def plot_training_curve(log_path="checkpoints/log.json", out="training_curve.png"):
    with open(log_path) as f:
        log = json.load(f)

    steps = [e["step"] for e in log]
    train_acc = [e["train_acc"] for e in log]
    test_acc = [e["test_acc"] for e in log]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, train_acc, label="train")
    ax.plot(steps, test_acc, label="test")
    ax.set_xlabel("step")
    ax.set_ylabel("accuracy")
    ax.legend()
    ax.set_title("grokking: delayed generalization on (a+b) mod 113")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    plt.show()


def plot_embedding_spectrum(model, out="embedding_spectrum.png"):
    W = model.embed.weight[:model.p].detach().numpy()  # (p, d_model)
    power = fourier_power(W, model.p)
    freqs = np.arange(len(power))

    fig, ax = plt.subplots(figsize=(11, 3))
    ax.bar(freqs, power, width=0.8)
    ax.set_xlabel("frequency")
    ax.set_ylabel("fraction of embedding power")
    ax.set_title("embedding matrix: Fourier spectrum — sparse = model learned the trig circuit")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    plt.show()


def activation_patch_accuracy(model, x, y, component="mlp"):
    """
    Zero out a component and measure accuracy drop.
    Rough way to check how load-bearing each part is.
    """
    with torch.no_grad():
        baseline = (model(x).argmax(-1) == y).float().mean().item()

    if component == "mlp":
        orig = [p.data.clone() for p in model.mlp.parameters()]
        for p in model.mlp.parameters():
            p.data.zero_()
        with torch.no_grad():
            patched = (model(x).argmax(-1) == y).float().mean().item()
        for p, o in zip(model.mlp.parameters(), orig):
            p.data.copy_(o)
    else:
        raise ValueError(f"unknown component: {component}")

    return {"baseline": baseline, "patched": patched, "drop": baseline - patched}


if __name__ == "__main__":
    plot_training_curve()

    model = load("checkpoints/final.pt")

    plot_embedding_spectrum(model)

    # sanity check: what happens if we zero the MLP
    import json
    with open("checkpoints/log.json") as f:
        log = json.load(f)
    # build test set
    from train import make_dataset
    import numpy as np
    rng = np.random.default_rng(42)
    data = make_dataset(113)
    rng.shuffle(data)
    n_train = int(len(data) * 0.5)
    test_data = data[n_train:]
    eq = 113
    x = torch.tensor([[a, b, eq] for a, b, _ in test_data], dtype=torch.long)
    y = torch.tensor([c for _, _, c in test_data], dtype=torch.long)

    result = activation_patch_accuracy(model, x, y, "mlp")
    print(f"accuracy with full model: {result['baseline']:.3f}")
    print(f"accuracy without MLP:     {result['patched']:.3f}  (drop: {result['drop']:.3f})")
