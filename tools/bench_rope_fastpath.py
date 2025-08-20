# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time

import torch

from vllm._rope_fastpath import rope_complex_fast, rope_torch_baseline


def bench(
    fn,
    B=1,
    H=16,
    T=2048,
    hd=64,
    steps=100,
    device="cuda",
    dtype=torch.float16,
):
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; skip")
        return None
    q = torch.randn(B, H, T, hd, device=device, dtype=dtype)
    k = torch.randn(B, H, T, hd, device=device, dtype=dtype)
    cos = torch.randn(T, hd // 2, device=device, dtype=dtype)
    sin = torch.randn(T, hd // 2, device=device, dtype=dtype)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        qb, kb = fn(q, k, cos, sin)
    if device == "cuda":
        torch.cuda.synchronize()
    return time.time() - t0


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for hd in (64, 128, 256):
        tb = bench(rope_torch_baseline, hd=hd, device=dev) or 0.0
        tf = bench(rope_complex_fast, hd=hd, device=dev) or 0.0
        if tb and tf:
            print(
                f"hd={hd} base={tb:.4f}s fast={tf:.4f}s "
                f"speedup={(tb/tf):.2f}x"
            )
        else:
            print(f"hd={hd}: device={dev} no timing (no CUDA?)")
