"""NPU benchmark: compact projector vs full-vocab LM Head.

Measures the latency win from the target-token-scoring compact path on an
Ascend NPU: ``index_select`` K candidate weight rows + ``F.linear`` ([B, K])
versus the full-vocab ``F.linear`` ([B, V]) that the native path runs.

Run on a 910B box with torch_npu installed:

    .venv/bin/python examples/target_token_scoring/npu_projector_benchmark.py

The numbers are *device-side projection only* (no sampler, no D2H). They show
the upper bound the compact path removes from the hot path; the end-to-end win
is smaller once the sampler/output copy is included.
"""

from __future__ import annotations

import argparse
import time

import torch

try:
    import torch_npu  # noqa: F401  -- registers NPU backend on import
    DEVICE = torch.device("npu:0")
except ImportError:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[warn] torch_npu not found; falling back to {DEVICE}")


def _bench(fn, iters: int = 50, warmup: int = 5) -> float:
    """Return median ms over ``iters`` timed runs after ``warmup`` untimed."""
    for _ in range(warmup):
        fn()
    if DEVICE.type != "cpu":
        torch.npu.synchronize() if DEVICE.type == "npu" else torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if DEVICE.type != "cpu":
            torch.npu.synchronize() if DEVICE.type == "npu" else torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return sorted(times)[len(times) // 2]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--hidden", type=int, default=896)      # Qwen2.5-0.5B
    p.add_argument("--vocab", type=int, default=151557)    # Qwen2.5 tokenizer
    p.add_argument("--num-candidates", type=int, default=2)
    args = p.parse_args()

    B, H, V, K = args.batch, args.hidden, args.vocab, args.num_candidates
    torch.manual_seed(0)
    weight = torch.randn(V, H, dtype=torch.float16, device=DEVICE)
    bias = torch.randn(V, dtype=torch.float16, device=DEVICE)
    hidden = torch.randn(B, H, dtype=torch.float16, device=DEVICE)
    target_ids = torch.as_tensor([100, 200][:K] if K <= 2 else list(range(K)),
                                  dtype=torch.long, device=DEVICE)

    def full_vocab():
        return torch.nn.functional.linear(hidden, weight, bias)  # [B, V]

    def compact():
        sel_w = weight.index_select(0, target_ids).contiguous()  # [K, H]
        sel_b = bias.index_select(0, target_ids).contiguous()    # [K]
        return torch.nn.functional.linear(hidden, sel_w, sel_b)  # [B, K]

    # Equivalence (FP32 on a small slice).
    full = full_vocab().float()
    comp = compact().float()
    exp = full[:, target_ids]
    assert torch.allclose(comp, exp, atol=1e-3), "compact != full candidate cols"

    t_full = _bench(full_vocab)
    t_compact = _bench(compact)
    print(f"device={DEVICE}  B={B} H={H} V={V} K={K}")
    print(f"full-vocab  F.linear [B,V]: {t_full:8.3f} ms")
    print(f"compact     F.linear [B,K]: {t_compact:8.3f} ms")
    print(f"speedup: {t_full / max(t_compact, 1e-6):.1f}x  (projection only)")


if __name__ == "__main__":
    main()
