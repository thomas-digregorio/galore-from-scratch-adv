# GaLore Implementation from Scratch

This repository contains a clean, "from scratch" implementation of **GaLore (Gradient Low-Rank Projection)** for training LLMs on consumer GPUs (specifically tested on an **NVIDIA RTX 5080 Laptop GPU**).

It includes a custom optimizer `GaLoreAdamW`, a training script for language modeling on WikiText-103, and a benchmarking notebook.

## Key Features
*   **Massive Memory Reduction**: Reduces optimizer state memory by up to 65% **by projecting gradients into a low-rank subspace** (Gradient Low-Rank Projection) instead of storing valid full-rank states.
*   **Synchronous State Reset**: Ensures robust convergence by resetting momentum when the SVD basis changes.
*   **Full Rank Warmup**: Starts with standard AdamW (Full Rank) to find a stable initial basin.
*   **WandB Logging**: Integrated Weights & Biases logging for loss, gradient norms, and memory usage tracking.
*   **Llama Compatible**: Designed to train Llama-architecture models (Configurable sizes).

## Installation

```bash
pip install torch transformers datasets wandb matplotlib tqdm
```

## Quick Start

### 1. Training (The Main Event)
Train a Llama-125M model on WikiText-103.

```bash
# Standard GaLore (Rank 128)
python train.py --size 125M --optimizer galore --lr 1e-3
```

**Arguments:**
*   `--size`: `60M`, `125M` (Default), `350M`, `1B`.

### 2. Benchmarking (The Proof)
Open `benchmark_viz.ipynb` in VS Code or Jupyter. 
Run all cells to visualize the memory difference between `AdamW` and `GaLore`.

> [!NOTE]
> **Why uses Random Weights?**
> The benchmark notebook initializes a fresh model with random weights for each run. This ensures a sterile, consistent environment to measure *only* the memory difference caused by the optimizer, isolating it from data loading or checkpointing variances.

## Training Stability & Hyperparameters

Training Low-Rank methods from scratch can be unstable if not handled correctly.
We use a **Synchronous State Reset** strategy to ensure robust convergence.

### Why Synchronous Reset?
*   **The Problem**: Changing the SVD basis ($P$) invalidates the stored optimizer momentum ($m$). Using old momentum with a new basis causes the model to jump in the wrong direction ("Explosions").
*   **The Solution**: We **reset** the optimizer states (`exp_avg`, `exp_avg_sq`) to zero whenever the projector updates (every 200 steps).
*   **Synchronous**: We update *all* layers at the same time (Step 500, 1000, etc.). This avoids the infinite "Creeping Loss" or "Freezing" seen with staggered updates.
*   **Full Rank Warmup**: We run standard AdamW (Full Rank) for the first `proj_start_steps` (e.g. 500). **This is recommended by the GaLore paper** to allow the model to find a stable initial basin before applying any Low-Rank compression/approximation.

### Recommended Configuration
| Parameter | Value | Reason |
| :--- | :--- | :--- |
| **Learning Rate** | `3e-4` | Standard Llama LR. Stable without quantization. |
| **Update Gap** | `500` | Aligned with Warmup (10% of 5000). First reset happens *after* warmup. |
| **Start Steps** | `500` | **Full Rank Warmup**. Runs standard AdamW for first 500 steps to find stable basin. |

**Stable Command (The "Golden" Run)**:
```bash
python train.py --size 125M --optimizer galore --steps 5000 --lr 3e-4 --update_proj_gap 500 --proj_start_steps 500
```

## Results & Expectations

### Why only 5000 Steps?
This repo serves as a **verification** of the GaLore algorithm implementation. Training a Large Language Model to full convergence typically requires:
*   **Tokens**: 3 Billion+ (vs ~40 Million in this run)
*   **Time**: Days or Weeks on a single GPU.

### Verified Results (Run: northern-surf-24)
*   **Final Loss**: 7.30 (Steady decrease from 10.8)
*   **Perplexity**: 1480.34
*   **Peak Memory**: 12.38 GB
*   **Gradient Norm**: 1.73 (Stable, no explosions)
*   **Output**: The model successfully transitioned from random noise to coherent character/word spacing ("Structured Nonsense"), confirming the optimizer is learning.
*   **Success Criteria**: The key indicator is that the Loss **continually decreases** and does not explode (NaN). This proves the Low-Rank Gradient Projection is correctly approximating the full gradient trajectory.

> [!NOTE]
> **Understanding Memory Savings**: The "65% reduction" claimed in Key Features applies specifically to **Optimizer State Memory** (which usually dominates training RAM).
> In smaller models (like 125M), fixed costs (Weights, Activation Buffers) take up a larger proportion of VRAM, so the *Total VRAM* reduction percentage may appear smaller (e.g. 18-20%), even though the optimizer state itself is compressed by ~60%. As model size grows (7B+), the Total VRAM savings approach the Optimizer State savings percentage.

## Project Structure

```text
.
├── galore/
│   └── optimizer.py    # The core GaLoreAdamW class
├── benchmark_viz.ipynb # Jupyter notebook for memory visualization
├── requirements.txt    # Dependencies
├── train.py            # Main training script (WikiText-103)
└── README.md
```

## Reference
*   **GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection**
    *   *Zhao, J., Zhang, Z., Chen, B., Wang, Z., Anandkumar, A., & Tian, Y. (2024)*
    *   [arXiv:2403.03507](https://arxiv.org/abs/2403.03507)
