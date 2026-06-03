# Grokking: Watching a Neural Network Learn to Generalize

**Status:** Complete  
**Built:** June 2026

---

## What was built

A one-layer transformer (~100k parameters) trained from scratch on modular arithmetic, with Fourier analysis tools to inspect what the model learned internally.

Two tasks: `(a + b) mod 113` and `(a × b) mod 113`. Both trained with the same architecture and hyperparameters (AdamW, weight decay 0.5, gradient clipping, 50k steps, 30% train split).

---

## Key findings

**1. Grokking reproduced cleanly**

Both tasks show the same pattern: training accuracy hits 100% around step 1000, then test accuracy sits flat near zero for ~10k more steps before jumping to 100% in a phase transition. The gap between memorization and generalization is the grokking phenomenon.

**2. Addition uses a Fourier circuit**

After grokking, the embedding matrix for the addition model has a sparse Fourier spectrum — only ~8-9 frequencies carry significant power. This is the footprint of the trig circuit:

```
cos(w(a+b)) = cos(wa)·cos(wb) - sin(wa)·sin(wb)
```

The model represents each number as a superposition of sine/cosine waves at those frequencies, and the MLP computes the sum via trig identities. Visible directly in the weights.

**3. Multiplication uses the same idea, in a different basis**

The multiplication model's embedding spectrum looks flat and unstructured when analyzed in natural token order (1, 2, 3, ..., 112). Reordering the tokens by discrete logarithm (powers of the primitive root g=3 mod 113) reveals sparse spikes — the same kind of structure as addition.

This works because discrete log maps multiplication to addition: `dlog(a×b) = dlog(a) + dlog(b) mod 112`. Under that reordering, the same Fourier analysis applies. Both models found the algebraically natural representation for their operation — the structure in the multiplication model was always there, just invisible from the wrong angle.

---

## Files

| File | Description |
|------|-------------|
| `model.py` | One-layer transformer, raw `nn.Module` |
| `train.py` | Dataset generation, training loop, gradient clipping, checkpointing |
| `analysis.py` | Fourier probes, training curves, discrete log reordering, 3-panel comparison |
| `diagnose.py` | Data split verification, checkpoint accuracy evaluation |

---

## What didn't get built

The original spec included weight norm tracking over training, effective rank analysis, and a Jupyter notebook walkthrough. These were deprioritised — the core Fourier analysis and the cross-task extension tell the story more directly.

---

## References

- Power et al. 2022 — [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177)
- Nanda et al. 2023 — [Progress Measures for Grokking via Mechanistic Interpretability](https://arxiv.org/abs/2301.05217)
