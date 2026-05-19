#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A/B bench for the tiny-dot scalar projection in ``rocm_unquantized_gemm``.

Times three implementations at the shared_expert_gate shape (1x1xK)
by calling each backend directly:

  - BLAS path: ``torch.nn.functional.linear`` (hipBLASLt on ROCm).  This
    is what the dispatcher in ``vllm/model_executor/layers/utils.py``
    would route to for any tiny ``Linear(hidden, 1)`` before the
    SEED-6 / Triton fast paths were added.
  - SEED-6 eager fused: ``(x*w).sum(dtype=x.dtype)`` -- the 2-launch
    Triton-free fast path that SEED-6 wired into the dispatcher.
  - Triton: ``_tiny_dot_triton`` -- the single-block Triton kernel added
    by this PR (production pick for K <= 4096 + bias=None).

Usage:
    python benchmarks/kernels/bench_tiny_dot.py
    python benchmarks/kernels/bench_tiny_dot.py --k 4096 --dtype fp16
"""

from __future__ import annotations

import argparse

import torch

from vllm.model_executor.layers.utils import _tiny_dot_triton
from vllm.triton_utils import triton


def time_us(fn, warmup_ms: int = 25, rep_ms: int = 80) -> float:
    return (
        triton.testing.do_bench(fn, warmup=warmup_ms, rep=rep_ms, return_mode="median")
        * 1000.0
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--k",
        type=int,
        default=2048,
        help="Hidden dim K (default 2048 = Qwen3.5-A3B shared_expert_gate)",
    )
    p.add_argument(
        "--dtype",
        choices=["bf16", "fp16"],
        default="bf16",
        help="Activation/weight dtype",
    )
    args = p.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    x = torch.randn(1, args.k, dtype=dtype, device="cuda") * 0.01
    w = torch.randn(1, args.k, dtype=dtype, device="cuda") * 0.01
    x_flat = x.reshape(-1).contiguous()
    w_flat = w.reshape(-1).contiguous()

    print(f"Shape: 1x1xK={args.k}, dtype={dtype}")
    print()

    def blas() -> torch.Tensor:
        return torch.nn.functional.linear(x, w)

    def eager_fused() -> torch.Tensor:
        return (x.reshape(-1) * w.reshape(-1)).sum(dtype=x.dtype)

    def triton_fast() -> torch.Tensor:
        return _tiny_dot_triton(x_flat, w_flat)

    for fn in (blas, eager_fused, triton_fast):
        fn()
    torch.accelerator.synchronize()

    t_blas = time_us(blas)
    t_eager = time_us(eager_fused)
    t_triton = time_us(triton_fast)

    print(f"  {'config':<24} {'time_us':>10}  {'vs BLAS':>10}")
    print("  " + "-" * 50)
    print(f"  {'BLAS (gfx11 baseline)':<24} {t_blas:>10.2f}  {'(ref)':>10}")
    print(f"  {'SEED-6 eager fused':<24} {t_eager:>10.2f}  {t_blas / t_eager:>9.2f}x")
    print(f"  {'Triton (this PR)':<24} {t_triton:>10.2f}  {t_blas / t_triton:>9.2f}x")


if __name__ == "__main__":
    main()
