# Grokking

Train a small transformer on modular arithmetic. Watch it memorize the training data first, then generalize — sometimes thousands of steps later — in what looks like a phase transition.

The interesting question is what changes inside the model when that happens. The answer (from [Nanda et al. 2023](https://arxiv.org/abs/2301.05217)) is that the model learns to implement the Discrete Fourier Transform: it represents numbers as Fourier frequencies and uses trig identities to compute addition.

This repo reproduces that finding cleanly and adds a few probing tools for looking at the weights directly.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python train.py          # trains for 60k steps, saves checkpoints/
python analysis.py       # loads final checkpoint, plots training curve + embedding spectrum
```

Training takes about 10 minutes on a laptop GPU, longer on CPU.

## What you see

After training: the test accuracy jumps from near-zero to near-100% around step 30k–50k, well after the model has already perfectly memorized the training set.

The embedding spectrum shows why: the model's number representations concentrate on a small set of Fourier frequencies. That's the footprint of the trig circuit.

## Files

- `model.py` — one-layer transformer
- `train.py` — dataset generation + training loop
- `analysis.py` — Fourier analysis, training curve, activation patching

## References

- Power et al. 2022 — [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177)
- Nanda et al. 2023 — [Progress Measures for Grokking via Mechanistic Interpretability](https://arxiv.org/abs/2301.05217)
