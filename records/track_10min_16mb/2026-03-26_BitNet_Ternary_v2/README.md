# LR Schedule Fix for torch.compile Warmup

## Summary

This submission fixes a critical bug in the LR schedule that caused training to fail when using `torch.compile`. The first step takes 20-30 seconds due to compilation overhead, which caused the wall-clock based warmdown schedule to immediately think training was almost complete.

**Key fix**: Skip the first 20 steps when computing wall-clock based LR warmdown, allowing torch.compile to settle before timing-based scheduling kicks in.

## Results (3 seeds)

| Seed | val_bpb (int6 roundtrip) | Model Size |
|------|--------------------------|------------|
| 1337 | 1.1568 | 16.0 MB |
| 42 | 1.1563 | 15.9 MB |
| 2025 | 1.1578 | 15.9 MB |

**Average: 1.1570 bpb**

## The Bug

Before the fix, at step 1:
- `elapsed_ms = 31529` (torch.compile overhead)
- `step_ms = 31529 / 1 = 31529` ms per step (wildly overestimated)
- `warmdown_ms = 3500 * 31529 = 110M ms`
- `remaining_ms = 600000 - 31529 = 568471` ms
- Since `remaining <= warmdown_ms`: `scale = 568471 / 110351500 ≈ 0.005`

This triggered `late_qat` at step 1 with near-zero learning rate, causing training to fail completely.

## The Fix

```python
compile_warmup_steps = 20  # torch.compile first-step overhead skews timing
def lr_mul(step: int, elapsed_ms: float) -> float:
    # ...existing checks...
    # Don't start wall-clock warmdown until torch.compile has settled
    if step < compile_warmup_steps:
        return 1.0
    # ...rest of function unchanged...
```

## Run Command

```bash
DATA_PATH=/path/to/fineweb10B_sp1024 \
TOKENIZER_PATH=/path/to/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Notes

- This is a non-record submission focused on fixing a training stability bug
- The int6 baseline achieves ~1.157 bpb with the fix applied
- Model architecture: 11 layers, 512 dim, 8 heads, 4 KV heads
