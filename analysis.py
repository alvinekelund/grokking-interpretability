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
    plt.close(fig)
    print(f"saved {out}")


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
    plt.close(fig)
    print(f"saved {out}")


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


def plot_spectra_comparison(out="spectra_comparison.png"):
    """Side-by-side embedding spectra for addition and multiplication."""
    add_ckpt = Path("checkpoints_add/final.pt")
    mul_ckpt = Path("checkpoints_mul/final.pt")

    if not add_ckpt.exists() or not mul_ckpt.exists():
        missing = [s for s, p in [("add", add_ckpt), ("mul", mul_ckpt)] if not p.exists()]
        print(f"missing checkpoints for: {missing} — run train.py with those ops first")
        return

    model_add = load(add_ckpt)
    model_mul = load(mul_ckpt)

    p = 113
    W_add = model_add.embed.weight[:p].detach().numpy()
    W_mul = model_mul.embed.weight[:p].detach().numpy()

    power_add = fourier_power(W_add, p)
    power_mul = fourier_power(W_mul, p)
    freqs = np.arange(len(power_add))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    ax1.bar(freqs, power_add, width=0.8)
    ax1.set_ylabel("power fraction")
    ax1.set_title("addition: (a + b) mod 113")

    ax2.bar(freqs, power_mul, width=0.8, color="orange")
    ax2.set_ylabel("power fraction")
    ax2.set_title("multiplication: (a × b) mod 113")
    ax2.set_xlabel("frequency")

    fig.suptitle("Embedding Fourier spectra — do both operations learn the same structure?")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    import sys
    op = sys.argv[1] if len(sys.argv) > 1 else None

    if op in ("add", None):
        plot_training_curve(
            log_path="checkpoints_add/log.json",
            out="training_curve_add.png",
        )
        model_add = load("checkpoints_add/final.pt")
        plot_embedding_spectrum(model_add, out="embedding_spectrum_add.png")

    if op in ("mul", None):
        plot_training_curve(
            log_path="checkpoints_mul/log.json",
            out="training_curve_mul.png",
        )
        model_mul = load("checkpoints_mul/final.pt")
        plot_embedding_spectrum(model_mul, out="embedding_spectrum_mul.png")

    if op is None:
        plot_spectra_comparison()
