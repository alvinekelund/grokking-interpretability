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


def primitive_root(p):
    """Smallest primitive root mod p (p must be prime)."""
    for g in range(2, p):
        order = p - 1
        n = order
        is_primitive = True
        q = 2
        while q * q <= n:
            if n % q == 0:
                if pow(g, order // q, p) == 1:
                    is_primitive = False
                    break
                while n % q == 0:
                    n //= q
            q += 1
        if n > 1 and pow(g, order // n, p) == 1:
            is_primitive = False
        if is_primitive:
            return g
    raise ValueError(f"no primitive root for p={p}")


def dlog_order(p):
    """
    Returns list of tokens in discrete log order: [g^0, g^1, ..., g^(p-2)] mod p.
    Under this reordering, a×b mod p becomes dlog(a)+dlog(b) mod (p-1),
    so the Fourier analysis over the additive group applies.
    """
    g = primitive_root(p)
    return [pow(g, k, p) for k in range(p - 1)]


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
    g = primitive_root(p)

    # addition: tokens 0..p-1 in natural order
    W_add = model_add.embed.weight[:p].detach().numpy()
    power_add = fourier_power(W_add, p)

    # multiplication: tokens 1..p-1 in natural order (flat — wrong basis)
    W_mul_natural = model_mul.embed.weight[1:p].detach().numpy()
    power_mul_natural = fourier_power(W_mul_natural, p - 1)

    # multiplication: tokens reordered by discrete log (g^0, g^1, ..., g^(p-2))
    order = dlog_order(p)
    W_mul_dlog = np.stack([model_mul.embed.weight[t].detach().numpy() for t in order])
    power_mul_dlog = fourier_power(W_mul_dlog, p - 1)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 7))
    ax1.bar(np.arange(len(power_add)), power_add, width=0.8)
    ax1.set_ylabel("power fraction")
    ax1.set_title("addition — natural token order")

    ax2.bar(np.arange(len(power_mul_natural)), power_mul_natural, width=0.8, color="orange")
    ax2.set_ylabel("power fraction")
    ax2.set_title("multiplication — natural token order (flat: wrong basis)")

    ax3.bar(np.arange(len(power_mul_dlog)), power_mul_dlog, width=0.8, color="green")
    ax3.set_ylabel("power fraction")
    ax3.set_title(f"multiplication — discrete log order (generator g={g})")
    ax3.set_xlabel("frequency")

    fig.suptitle("Embedding Fourier spectra: same analysis, different bases")
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
