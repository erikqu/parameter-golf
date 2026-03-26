# 11L Compile Warmup + Int6 + EMA

**val_bpb: 1.1570** (3-seed mean, std 0.0008) | **~15.9 MB** | 8×H100 SXM

## Summary

When `torch.compile` kicks in, the first training step takes 20-30 seconds instead of ~100ms. The wall-clock based LR warmdown schedule was using this inflated timing to estimate total training duration, causing it to immediately enter warmdown with `scale ≈ 0.005` at step 1.

**The fix**: Let the compile settle before measuring. Skip the first 20 steps when computing wall-clock warmdown.

## Results (8×H100 80GB SXM)

| Seed | Steps | Step Avg | **val_bpb** | Artifact |
|------|-------|----------|-------------|----------|
| 1337 | 5,135 | 117ms | **1.1568** | 15.9 MB |
| 42 | 5,198 | 115ms | **1.1563** | 15.9 MB |
| 2025 | 5,030 | 119ms | **1.1578** | 15.9 MB |
| **Mean** | **~5,120** | **117ms** | **1.1570** | |

## The Problem

Before the fix, at step 1:
```
elapsed_ms = 31529        (torch.compile overhead)
step_ms = 31529 / 1       = 31529 ms/step (wildly wrong!)
warmdown_ms = 3500 × 31529 = 110M ms
remaining_ms = 600000 - 31529 = 568471 ms
scale = 568471 / 110351500 ≈ 0.005
```

This triggered `late_qat` immediately with near-zero learning rate, completely breaking training.

## The Fix

```python
compile_warmup_steps = 20  # torch.compile overhead skews timing

def lr_mul(step: int, elapsed_ms: float) -> float:
    # ...existing checks...
    # Don't start wall-clock warmdown until torch.compile has settled
    if step < compile_warmup_steps:
        return 1.0
    # ...rest unchanged...
```

## Architecture

| Component | Setting |
|-----------|---------|
| Layers | 11 |
| Model dim | 512 |
| Heads | 8 (4 KV) |
| MLP | 3× |
| XSA | Last 4 layers |
| Quantization | int6 (GPTQ-lite) |
| EMA | Enabled |

## Run Command

```bash
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
