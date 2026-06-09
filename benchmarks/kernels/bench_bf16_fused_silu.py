#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A/B bench for the META3-2 wvSplitK_fused_silu_{mul,gate_mul} fast path
exercised from Qwen2MoeMLP.forward at the shared-expert decode shape (N=1).

Times the production ``Qwen2MoeMLP.forward`` call:
  - Baseline (eager chain): silu_and_mul -> down_proj wvSplitK -> sigmoid*out
  - This PR (fused, gate=1): wvSplitK_fused_silu_gate_mul (single launch)
  - This PR (fused, no gate): wvSplitK_fused_silu_mul + post-mul sigmoid

A/B by toggling ``VLLM_BF16_WVSPLITK_FUSED_SILU`` in-process (the env var
is read every forward call via os.environ.get inside the probe).

Usage:
    python benchmarks/kernels/bench_bf16_fused_silu.py
    python benchmarks/kernels/bench_bf16_fused_silu.py --hidden 2048 --intermediate 512
"""

from __future__ import annotations

import argparse
import os

import torch

from vllm.triton_utils import triton

os.environ.setdefault("VLLM_ROCM_USE_SKINNY_GEMM", "1")

from vllm import _custom_ops as ops  # noqa: E402


def time_us(fn, warmup_ms: int = 25, rep_ms: int = 80) -> float:
    return (
        triton.testing.do_bench(fn, warmup=warmup_ms, rep=rep_ms, return_mode="median")
        * 1000.0
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--hidden", type=int, default=2048, help="Hidden dim (default Qwen3.5-A3B)"
    )
    p.add_argument(
        "--intermediate",
        type=int,
        default=512,
        help="Shared-expert intermediate dim (default Qwen3.5-A3B)",
    )
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16", help="dtype")
    args = p.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    H, IN = args.hidden, args.intermediate
    dev = "cuda"

    # gate_up is the output of gate_up_proj: shape [N=1, 2*IN]
    gate_up = (torch.randn(1, 2 * IN, dtype=dtype, device=dev) * 0.01).contiguous()
    # down_proj weight: [H, IN]
    down_weight = (torch.randn(H, IN, dtype=dtype, device=dev) * 0.01).contiguous()
    # expert_gate weight: [1, H]
    gate_scalar = torch.randn(1, 1, dtype=dtype, device=dev) * 0.5
    from vllm.utils.platform_utils import num_compute_units

    cu = num_compute_units()

    print(
        f"Shape: gate_up=[1, 2*{IN}={2 * IN}], down=[{H}, {IN}], gate=[1, 1] | cu={cu}"
    )
    print()

    # Baseline: eager silu_and_mul + wvSplitK + sigmoid*out
    def eager() -> torch.Tensor:
        gate = gate_up[:, :IN]
        up = gate_up[:, IN:]
        x = torch.nn.functional.silu(gate) * up
        out = ops.wvSplitK(down_weight, x, cu)
        return torch.sigmoid(gate_scalar) * out

    # Phase 1: wvSplitK_fused_silu_mul + post-multiply
    def fused_p1() -> torch.Tensor:
        out = ops.wvSplitK_fused_silu_mul(down_weight, gate_up, cu)
        return torch.sigmoid(gate_scalar) * out

    # Phase 2: wvSplitK_fused_silu_gate_mul (sigmoid+mul fused into epilogue)
    def fused_p2() -> torch.Tensor:
        return ops.wvSplitK_fused_silu_gate_mul(
            down_weight, gate_up, torch.sigmoid(gate_scalar), cu
        )

    # Warm and time each.
    for fn in (eager, fused_p1, fused_p2):
        fn()
    torch.accelerator.synchronize()

    t_eager = time_us(eager)
    t_p1 = time_us(fused_p1)
    t_p2 = time_us(fused_p2)

    print(f"  {'config':<36} {'time_us':>10}  {'vs eager':>9}")
    print("  " + "-" * 60)
    print(f"  {'eager (silu_mul+wv+sig*out)':<32} {t_eager:>10.2f}  {'(ref)':>9}")
    print(f"  {'fused P1 (silu_mul)':<32} {t_p1:>10.2f}  {t_eager / t_p1:>8.2f}×")
    print(
        f"  {'fused P2 (silu_mul+gate*out)':<32} {t_p2:>10.2f}  {t_eager / t_p2:>8.2f}×"
    )


if __name__ == "__main__":
    main()
