# Grokking: Watching a Neural Network Learn to Generalize

**Status:** Speccing  
**Target:** Late Q2 2026 — recruiter-ready portfolio project  
**Audience:** ML research engineers at Anthropic, OpenAI, Google DeepMind, Meta AI

---

## The Hook

Train a tiny transformer (2 layers, ~100k parameters) on a simple task — modular arithmetic, like "37 + 42 mod 97 = ?" — and something strange happens.

The model first **memorizes** the training data perfectly. Training accuracy: 100%. Test accuracy: ~5%. Useless.

Then, after thousands of additional training steps with no new data, something clicks. Test accuracy suddenly jumps from 5% to 99% in what looks like a phase transition. The model has "grokked" the task.

**The question this project answers:** *What actually changed inside the model during that jump?*

The answer is beautiful: the model learned to implement the **Discrete Fourier Transform**. It represents numbers as superpositions of Fourier frequencies, and uses trigonometric identities to compute modular addition. You can see this directly in the weights.

---

## Why This Matters (the honest version)

If you're deploying an LLM in production, you need to know when it generalizes and when it's just memorizing. Grokking shows there's a phase transition between these two regimes that isn't visible from training loss alone. Understanding *what circuit the model builds* when it generalizes is the first step toward predicting and guaranteeing behavior.

This is the core problem behind model reliability. It's not theoretical — it's what makes the difference between a model that works in demos and one that works in production.

---

## What Gets Built

### 1. Minimal transformer from scratch
- No PyTorch Lightning, no HuggingFace. Raw `nn.Module`.
- 2-layer, 4-head transformer. Embedding + unembedding. ~100k params.
- Trained on the full dataset of modular arithmetic pairs: `(a + b) mod p` for all `a, b < p`.

### 2. Training dynamics visualization
- Loss curves that show the memorization-then-generalization transition
- Weight norm evolution over training (Nanda et al. showed L2 norm drops right before grokking)
- Effective rank of weight matrices over time

### 3. Fourier analysis probes
- Show that the residual stream at the final layer decomposes along specific Fourier frequencies
- Plot which frequencies are active and when they appear during training
- Show that the same frequencies are used for the embedding and the output

### 4. Circuit dissection
- Identify which attention heads and MLP layers implement which parts of the computation
- Activation patching: zero out components and measure the drop in accuracy
- Visualize the "trig circuit": how `cos(w(a+b)) = cos(wa)cos(wb) - sin(wa)sin(wb)` is being computed

### 5. Extension: grokking on a new task
- Pick a task not in the original paper (e.g., permutation composition, or multi-step arithmetic)
- Show whether the same dynamics appear, and what circuit the model builds

---

## Deliverables

| File | Description |
|------|-------------|
| `train.py` | Clean training loop, checkpointing |
| `model.py` | Transformer implementation from scratch |
| `analysis.py` | Probing tools: Fourier decomposition, activation patching |
| `notebooks/grokking.ipynb` | Full narrative walkthrough with all plots |
| `README.md` | Project overview, key findings, how to reproduce |

The notebook is the main artifact — a self-contained story from "here's the mystery" to "here's what the model actually learned."

---

## Key Papers

- Power et al. 2022 — "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets" (the original phenomenon)
- Nanda et al. 2023 — "Progress Measures for Grokking via Mechanistic Interpretability" (the Fourier circuit finding)
- Liu et al. 2023 — follow-up work on grokking in other architectures

---

## Why Alvin Is Doing This (the recruiter version)

His thesis was about LLM reliability in production at enterprises. The fundamental open question underneath that work: *when does a model generalize, and how do you know?* Grokking is the cleanest version of that question in a setting small enough to study completely. This project is about building the intuition and tooling to actually look inside a model and read what it learned.

---

## Potential Extensions (pick one after core is done)

### A — Grokking speedup
Show that weight decay or targeted L1 regularization on the embedding can cut time-to-grokking by 10x. Explain *why* mechanistically (it forces the model to commit to sparse Fourier representations earlier). Practical implication: curriculum design for efficient generalization.

### B — Cross-task universality
Show that the same Fourier-basis circuit emerges for mod multiplication, not just mod addition. If true: this suggests the circuit is a stable attractor, not an artifact of one task. If false: characterize what's different. Either outcome is interesting.

### C — Grokking in a real LLM
Find evidence of grokking-like dynamics in a small GPT-2 trained on structured text (e.g., arithmetic word problems). Does the phase transition appear? What does the circuit look like? Bridges toy-model findings to production-relevant models.
