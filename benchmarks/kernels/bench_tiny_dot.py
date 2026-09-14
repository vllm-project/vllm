# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A/B bench for the tiny scalar projection in rocm_unquantized_gemm.

Times the three implementations of a 1x1xK dot, the shape a Qwen MoE
shared_expert_gate produces:

  - BLAS: torch.nn.functional.linear, i.e. hipBLASLt on ROCm, what the
    dispatcher routed to before the fast path existed.
  - eager fused: (x*w).sum(dtype=x.dtype), the two-launch fallback.
  - Triton: _tiny_dot_triton, the single-launch production pick.

Usage:
    python benchmarks/kernels/bench_tiny_dot.py --k 2048 --dtype bf16
"""

from __future__ import annotations

import argparse

import torch

from vllm.model_executor.layers.utils import _tiny_dot_triton
from vllm.triton_utils import triton


def time_us(fn) -> float:
    return triton.testing.do_bench(fn, warmup=25, rep=80, return_mode="median") * 1000.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--k",
        type=int,
        default=2048,
        help="Hidden dim K (2048 = Qwen3.5-A3B shared_expert_gate)",
    )
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    args = p.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    x = torch.randn(1, args.k, dtype=dtype, device="cuda") * 0.01
    w = torch.randn(1, args.k, dtype=dtype, device="cuda") * 0.01
    x_flat = x.reshape(-1).contiguous()
    w_flat = w.reshape(-1).contiguous()

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

    print(f"Shape 1x1xK={args.k}, dtype={dtype}")
    print(f"  {'config':<20} {'time_us':>10}  {'vs BLAS':>10}")
    print("  " + "-" * 44)
    print(f"  {'BLAS':<20} {t_blas:>10.2f}  {'(ref)':>10}")
    print(f"  {'eager fused':<20} {t_eager:>10.2f}  {t_blas / t_eager:>9.2f}x")
    print(f"  {'Triton':<20} {t_triton:>10.2f}  {t_blas / t_triton:>9.2f}x")


if __name__ == "__main__":
    main()
