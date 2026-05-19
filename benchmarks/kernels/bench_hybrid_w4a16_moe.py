#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A/B bench for the hybrid_w4a16 MoE Triton prefill kernel.

Compares the BLOCK_SIZE_M choices at the Qwen3.5-A3B prefill shape
(E=256 experts, top_k=8, M=128 prefill batch, K=2048, intermediate N=1024)
by mutating ``HybridW4A16MoEExperts.TRITON_BLOCK_SIZE_M`` in-process and
re-running the full prefill (no kernel-internal compile cache leakage
because each BLOCK_SIZE_M selects a distinct Triton config tuple).

Also prints the workspace footprint (``num_slots`` in
``workspace_shapes()``) for each BLOCK_M so reviewers can verify the
``E·(BLOCK_M-1)`` padding-term shrink argument without running the GPU.

Usage:
    python benchmarks/kernels/bench_hybrid_w4a16_moe.py
    python benchmarks/kernels/bench_hybrid_w4a16_moe.py --m 256 --topk 4

Limitation: end-to-end timing includes weight quantization, sort, and the
two GEMMs back-to-back — the kernel-level numbers are drowned by the
host-side setup. For kernel-only timings, use the
``sweep_int4g_moe_kernel.py`` rig from the stacked int4-MoE PR (which
drives ``fused_moe_wvSplitK_int4_gemm_sweep`` directly).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from vllm.triton_utils import triton

# Make the in-tree tests helper importable from a stock checkout.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tests.kernels.moe.test_hybrid_w4a16_moe import _run_hybrid_moe  # noqa: E402
from vllm.model_executor.layers.fused_moe.hybrid_w4a16_moe import (  # noqa: E402
    HybridW4A16MoEExperts,
)
from vllm.utils.math_utils import round_up  # noqa: E402


def time_us(fn, warmup_ms: int = 25, rep_ms: int = 80) -> float:
    return (
        triton.testing.do_bench(fn, warmup=warmup_ms, rep=rep_ms, return_mode="median")
        * 1000.0
    )


def bench(m: int, n: int, k: int, e: int, topk: int, group_size: int) -> float:
    fn = lambda m=m, n=n, k=k, e=e, topk=topk, gs=group_size: _run_hybrid_moe(  # noqa: E731
        m=m, n=n, k=k, e=e, topk=topk, group_size=gs, force_triton=True
    )
    fn()
    torch.accelerator.synchronize()
    return time_us(fn)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m", type=int, default=128, help="Prefill batch (rows)")
    p.add_argument("--n", type=int, default=1024, help="Intermediate dim")
    p.add_argument("--k", type=int, default=2048, help="Hidden dim")
    p.add_argument("--e", type=int, default=256, help="Number of experts")
    p.add_argument("--topk", type=int, default=8, help="Experts per token")
    p.add_argument("--group-size", type=int, default=128, help="W4A16 group size")
    p.add_argument(
        "--block-sizes",
        type=int,
        nargs="+",
        default=[32, 64, 128],
        help="BLOCK_SIZE_M values to bench (first is treated as the reference)",
    )
    args = p.parse_args()

    shape = dict(
        m=args.m,
        n=args.n,
        k=args.k,
        e=args.e,
        topk=args.topk,
        group_size=args.group_size,
    )
    print(f"hybrid_w4a16 MoE, force-Triton: {shape}")
    print()

    timings: list[tuple[int, float]] = []
    for bm in args.block_sizes:
        HybridW4A16MoEExperts.TRITON_BLOCK_SIZE_M = bm
        timings.append((bm, bench(**shape)))

    ref_bm, ref_t = timings[0]
    print(f"  {'BLOCK_SIZE_M':<20} {'time_us':>10}  {'vs BM=' + str(ref_bm):>10}")
    print("  " + "-" * 46)
    for bm, t in timings:
        ratio = "(ref)" if bm == ref_bm else f"{t / ref_t:.2f}×"
        print(f"  {bm:<20} {t:>10.1f}  {ratio:>10}")
    print()

    # Workspace footprint: num_slots = ceil_to_blockm(M*topk + E*(BLOCK_M-1))
    header = (
        f"  Workspace shrink (gemm1 num_slots) at "
        f"M={args.m}, E={args.e}, top_k={args.topk}:"
    )
    print(header)
    for bm in args.block_sizes:
        slots = round_up(args.m * args.topk + args.e * (bm - 1), bm)
        print(f"    BLOCK_M={bm:<4}  num_slots = {slots}")


if __name__ == "__main__":
    main()
