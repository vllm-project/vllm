#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel-only sweep for the hybrid_w4a16 MoE prefill Triton kernel.

Calls ``invoke_fused_moe_kernel_hybrid_triton`` directly (skipping the
HybridW4A16MoEExperts.apply host setup so kernel time is not drowned
out, unlike benchmarks/kernels/bench_hybrid_w4a16_moe.py).  Sweeps the
full Triton config space and ranks results per gemm.

Default shapes target Qwen3-Omni-30B-A3B (E=128, group_size=32,
hidden=2048, moe_intermediate=768) so the tune for this model can be
re-derived from scratch.  Override with --n / --k / --e / --group-size
to retune for other shapes (e.g. --n 512 --k 2048 --e 256
--group-size 128 for Qwen3.5-A3B).

Usage:
    python benchmarks/kernels/sweep_hybrid_w4a16_moe_triton.py
    python benchmarks/kernels/sweep_hybrid_w4a16_moe_triton.py \\
        --m 2048 --n 512 --k 2048 --e 256 --topk 8 --group-size 128 \\
        --csv /scratch/mgehre/tmp/sweep_qwen35.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tests.kernels.moe.test_hybrid_w4a16_moe import (  # noqa: E402
    _make_hybrid_moe_weights,
)
from vllm.model_executor.layers.fused_moe import fused_topk  # noqa: E402
from vllm.model_executor.layers.fused_moe.fused_moe import (  # noqa: E402
    invoke_fused_moe_kernel_hybrid_triton,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: E402
    moe_align_block_size,
)
from vllm.triton_utils import tl, triton  # noqa: E402


def build_inputs(
    M: int,
    N_inter: int,
    K_hidden: int,
    E: int,
    topk: int,
    group_size: int,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
):
    """Build weights, hidden, and topk_ids once for the whole sweep."""
    w1, w1_s, _ = _make_hybrid_moe_weights(E, K_hidden, 2 * N_inter, group_size, device)
    w2, w2_s, _ = _make_hybrid_moe_weights(E, N_inter, K_hidden, group_size, device)
    hidden = torch.randn(M, K_hidden, device=device, dtype=dtype) / 10.0
    scores = torch.randn(M, E, device=device, dtype=dtype)
    topk_weights, topk_ids, _ = fused_topk(hidden, scores, topk, False)
    return dict(
        w1=w1,
        w1_s=w1_s,
        w2=w2,
        w2_s=w2_s,
        hidden=hidden,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )


def time_us(fn, warmup_ms: int = 25, rep_ms: int = 80) -> float:
    return (
        triton.testing.do_bench(fn, warmup=warmup_ms, rep=rep_ms, return_mode="median")
        * 1000.0
    )


def make_gemm1_fn(inputs, cfg, group_size, E):
    """Closure that runs only gemm1 (hidden -> 2*N_inter)."""
    block_m = cfg["BLOCK_SIZE_M"]
    sorted_ids, expert_ids, npp = moe_align_block_size(
        inputs["topk_ids"], block_m, E, None, ignore_invalid_experts=True
    )
    num_slots = sorted_ids.size(0)
    N_w = inputs["w1"].size(1)  # 2*N_inter
    out = torch.empty(
        num_slots, N_w, device=inputs["hidden"].device, dtype=inputs["hidden"].dtype
    )
    compute_type = (
        tl.float16 if inputs["hidden"].dtype == torch.float16 else tl.bfloat16
    )

    def run():
        invoke_fused_moe_kernel_hybrid_triton(
            A=inputs["hidden"],
            B=inputs["w1"],
            C=out,
            B_scale=inputs["w1_s"],
            topk_weights=None,
            sorted_token_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=npp,
            mul_routed_weight=False,
            top_k=inputs["topk_ids"].size(1),
            config=cfg,
            compute_type=compute_type,
            group_size=group_size,
            align_block_size_m=block_m,
        )

    return run


def make_gemm2_fn(inputs, cfg, group_size, E):
    """Closure that runs only gemm2 (N_inter -> hidden).

    Mimics apply()'s second call: A is in slot-space (one row per slot
    with the post-activation activations), B is w2, top_k is 1 so the
    kernel reads A[slot] directly.
    """
    block_m = cfg["BLOCK_SIZE_M"]
    sorted_ids, expert_ids, npp = moe_align_block_size(
        inputs["topk_ids"], block_m, E, None, ignore_invalid_experts=True
    )
    num_slots = sorted_ids.size(0)
    # w2: [E, N=K_hidden, K_in=N_inter//8 (int32, holds N_inter//8 int4 elems each)]
    K_in = inputs["w2"].size(2) * 8
    K_hidden = inputs["w2"].size(1)
    act = (
        torch.randn(
            num_slots,
            K_in,
            device=inputs["hidden"].device,
            dtype=inputs["hidden"].dtype,
        )
        / 10.0
    )
    out = torch.empty(
        num_slots,
        K_hidden,
        device=inputs["hidden"].device,
        dtype=inputs["hidden"].dtype,
    )
    compute_type = (
        tl.float16 if inputs["hidden"].dtype == torch.float16 else tl.bfloat16
    )

    def run():
        invoke_fused_moe_kernel_hybrid_triton(
            A=act,
            B=inputs["w2"],
            C=out,
            B_scale=inputs["w2_s"],
            topk_weights=None,
            sorted_token_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=npp,
            mul_routed_weight=False,
            top_k=1,
            config=cfg,
            compute_type=compute_type,
            group_size=group_size,
            align_block_size_m=block_m,
        )

    return run


def sweep(args):
    inputs = build_inputs(args.m, args.n, args.k, args.e, args.topk, args.group_size)
    torch.accelerator.synchronize()

    # Each (BM, BN, BK, GM, nw, ns) candidate.  BK is capped to
    # group_size inside the wrapper, so listing values > group_size is
    # equivalent to BK=group_size.
    bm_list = [int(x) for x in args.block_m]
    bn_list = [int(x) for x in args.block_n]
    bk_list = [int(x) for x in args.block_k]
    gm_list = [int(x) for x in args.group_m]
    nw_list = [int(x) for x in args.num_warps]
    ns_list = [int(x) for x in args.num_stages]

    gemms: list[tuple[str, callable]] = []
    if "gemm1" in args.gemms:
        gemms.append(("gemm1", make_gemm1_fn))
    if "gemm2" in args.gemms:
        gemms.append(("gemm2", make_gemm2_fn))

    rows: list[dict] = []

    for gname, maker in gemms:
        print(
            f"\n=== {gname} sweep "
            f"(M={args.m}, N={args.n}, K={args.k}, E={args.e}, "
            f"topk={args.topk}, group_size={args.group_size}) ==="
        )
        results: list[tuple[dict, float]] = []
        for bm, bn, bk, gm, nw, ns in itertools.product(
            bm_list, bn_list, bk_list, gm_list, nw_list, ns_list
        ):
            cfg = dict(
                BLOCK_SIZE_M=bm,
                BLOCK_SIZE_N=bn,
                BLOCK_SIZE_K=bk,
                GROUP_SIZE_M=gm,
                num_warps=nw,
                num_stages=ns,
            )
            try:
                fn = maker(inputs, cfg, args.group_size, args.e)
                fn()  # warmup + correctness sanity
                torch.accelerator.synchronize()
                t = time_us(fn)
            except Exception as e:
                print(f"  SKIP {cfg}: {type(e).__name__}: {e}")
                continue
            results.append((cfg, t))
            rows.append({"gemm": gname, **cfg, "us": t})

        results.sort(key=lambda x: x[1])
        ref_cfg, ref_t = results[0]
        print(
            f"  {'rank':>4}  {'BM':>4} {'BN':>4} {'BK':>4} {'GM':>4} "
            f"{'nw':>3} {'ns':>3}  {'us':>9}  {'vs best':>8}"
        )
        for i, (cfg, t) in enumerate(results):
            mark = "*" if i == 0 else " "
            print(
                f"  {i + 1:>4}{mark} {cfg['BLOCK_SIZE_M']:>4} "
                f"{cfg['BLOCK_SIZE_N']:>4} {cfg['BLOCK_SIZE_K']:>4} "
                f"{cfg['GROUP_SIZE_M']:>4} {cfg['num_warps']:>3} "
                f"{cfg['num_stages']:>3}  {t:>9.1f}  {t / ref_t:>7.2f}x"
            )

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "gemm",
                    "BLOCK_SIZE_M",
                    "BLOCK_SIZE_N",
                    "BLOCK_SIZE_K",
                    "GROUP_SIZE_M",
                    "num_warps",
                    "num_stages",
                    "us",
                ],
            )
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV: {args.csv}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    # Default = Qwen3-Omni-30B-A3B thinker shape.
    p.add_argument("--m", type=int, default=2048, help="Prefill batch (rows)")
    p.add_argument(
        "--n",
        type=int,
        default=768,
        help="moe_intermediate_size (single, not 2x).  768 for Qwen3-Omni, "
        "512 for Qwen3.5-A3B.",
    )
    p.add_argument("--k", type=int, default=2048, help="hidden_size")
    p.add_argument(
        "--e",
        type=int,
        default=128,
        help="num_experts.  128 for Qwen3-Omni, 256 for Qwen3.5-A3B.",
    )
    p.add_argument("--topk", type=int, default=8, help="experts per token")
    p.add_argument(
        "--group-size",
        type=int,
        default=32,
        help="W4A16 group size.  32 for Qwen3-Omni, 128 for Qwen3.5-A3B.",
    )
    p.add_argument(
        "--gemms",
        nargs="+",
        default=["gemm1", "gemm2"],
        choices=["gemm1", "gemm2"],
    )
    p.add_argument("--block-m", nargs="+", default=[16, 32, 64, 128])
    p.add_argument("--block-n", nargs="+", default=[16, 32, 64, 128])
    p.add_argument(
        "--block-k",
        nargs="+",
        default=[32, 64, 128],
        help="Capped to group_size inside the wrapper",
    )
    p.add_argument("--group-m", nargs="+", default=[1, 4, 8])
    p.add_argument("--num-warps", nargs="+", default=[2, 4, 8])
    p.add_argument("--num-stages", nargs="+", default=[1, 2])
    p.add_argument(
        "--csv", type=str, default=None, help="Write all results to this CSV path"
    )
    args = p.parse_args()
    sweep(args)


if __name__ == "__main__":
    main()
