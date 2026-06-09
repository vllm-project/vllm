#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sweep tuning parameters for the fused-MoE int4 skinny GEMM kernel.

Counterpart to ``sweep_int4g_kernel.py`` / ``sweep_w8a8_kernel.py`` /
``sweep_bf16_kernel.py`` but for the MoE-routed int4 path:
``fused_moe_wvSplitK_int4_gemm``.  Uses the 4-axis sweep op
``fused_moe_wvSplitK_int4_gemm_sweep`` (gated behind
``VLLM_SKINNY_GEMM_SWEEP=1``; build with that flag to expose it).

Each cell sweeps ``YTILE x UNRL x A_CHUNK x WvPrGrp`` and anchors the
result with one call to the production ``fused_moe_wvSplitK_int4_gemm``
op — that anchor is the dispatcher's pick at the same shape, useful both
as a baseline and as the verification number any external rig should
match.

Default shape list targets the Qwen3.5-A3B int4 MoE decode shapes:
GEMM1 (gate_up, K=hidden=2048, N=2*intermediate=1024) and GEMM2 (down,
K=intermediate=512, N=hidden=2048), at the routing batch (block_size_m=1)
used in production.  Single-token decode with top_k=8 yields num_slots=8.

Usage:
    python sweep_int4g_moe_kernel.py
    python sweep_int4g_moe_kernel.py --shapes 2048x512    # K=512 down only
    python sweep_int4g_moe_kernel.py --block-sizes 1 2 4
"""

import argparse
import csv
import itertools
import time

import torch

import vllm._custom_ops as ops
from vllm.triton_utils import triton
from vllm.utils.platform_utils import num_compute_units as get_cu_count

# (M = N_weight, K, label) — for MoE, M is the weight's N_weight dim
# (gate+up output for GEMM1, hidden for GEMM2).
SHAPES = [
    (1024, 2048, "Qwen3.5-A3B MoE gate_up (E=256, K=hidden)"),
    (2048, 512, "Qwen3.5-A3B MoE down    (E=256, K=intermediate)"),
]

YTILES = [1, 2, 4]
UNRLS = [1, 2, 4]
ACHUNKS = [8, 16, 32]
WVPRGRPS = [8, 12, 16, 32]  # MoE sweep op exposes W=32 (production grid stops at 16)

GROUP_SIZE = 128
NUM_EXPERTS = 256
TOP_K = 8


def pack_int4(values_int8: torch.Tensor) -> torch.Tensor:
    """Pack signed int4 (-8..7) using the ExLlama shuffle the production
    kernel expects.  Mirrors sweep_int4g_kernel.py exactly."""
    M, K = values_int8.shape
    unsigned = (values_int8.to(torch.int16) + 8).to(torch.uint8)
    g = unsigned.view(M, K // 8, 8).to(torch.int32)
    shuffled = (
        g[:, :, 0]
        | (g[:, :, 2] << 4)
        | (g[:, :, 4] << 8)
        | (g[:, :, 6] << 12)
        | (g[:, :, 1] << 16)
        | (g[:, :, 3] << 20)
        | (g[:, :, 5] << 24)
        | (g[:, :, 7] << 28)
    )
    return shuffled.contiguous().view(torch.int8).contiguous()


def parse_shape(s: str) -> tuple[int, int, str]:
    parts = s.split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Shape must be MxK, got '{s}'")
    return (int(parts[0]), int(parts[1]), s)


def build_moe_tensors(M, K, block_size_m, dtype, device):
    """Construct (a, w, scales, c, expert_ids) matching production layout."""
    num_groups = K // GROUP_SIZE
    num_slots = block_size_m * TOP_K  # one token, top_k expert assignments

    # Per-expert packed weights.  pack_int4 returns [M, K//2] of int8
    # (each int8 holds 2 int4 nibbles).  Stack -> [E, M, K//2] int8,
    # reinterpret as int32 -> [E, M, K//8] which is the layout the
    # production op expects (w.size(2) * 8 == K_in).
    w_int8 = torch.randint(-8, 8, (NUM_EXPERTS, M, K), dtype=torch.int8, device=device)
    w_packed_i8 = torch.stack([pack_int4(w_int8[e]) for e in range(NUM_EXPERTS)], dim=0)
    # Bit-reinterpret int8 -> int32; element count divides by 4.
    w = w_packed_i8.view(torch.int32).reshape(NUM_EXPERTS, M, K // 8).contiguous()

    scales = torch.rand(NUM_EXPERTS, M, num_groups, dtype=dtype, device=device)
    scales = scales * 0.02 - 0.01
    a = (torch.rand(num_slots, K, dtype=dtype, device=device) - 0.5) * 0.01
    c = torch.empty(num_slots, M, dtype=dtype, device=device)
    # Round-robin expert assignment for the sweep (real router would
    # produce a different distribution; for kernel timing the assignment
    # pattern is irrelevant — the kernel processes one block per WG).
    expert_ids = torch.arange(num_slots, dtype=torch.int32, device=device) % NUM_EXPERTS
    return a, w, scales, c, expert_ids


def time_us(fn, warmup: int, rep: int) -> float:
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")
    return ms * 1000.0


def run_sweep(shapes, block_sizes, warmup, rep, dtype, csv_path):
    cu_count = get_cu_count()
    gpu_name = torch.cuda.get_device_name(0)

    # torch.ops attribute lookup is lazy; check via the dispatcher's op name
    # registry instead of hasattr.
    if (
        "_rocm_C::fused_moe_wvSplitK_int4_gemm_sweep"
        not in torch._C._dispatch_get_all_op_names()
    ):
        raise SystemExit(
            "fused_moe_wvSplitK_int4_gemm_sweep is not registered -- build vLLM "
            "with VLLM_SKINNY_GEMM_SWEEP=1 to expose the sweep op."
        )

    print(f"GPU: {gpu_name}, CU count: {cu_count}")
    print(f"Dtype: {dtype}, group_size: {GROUP_SIZE}, E: {NUM_EXPERTS}, top_k: {TOP_K}")
    print(f"Shapes: {len(shapes)}, block_size_m: {block_sizes}")
    print(
        f"Grid: YTILE={YTILES} x UNRL={UNRLS} x A_CHUNK={ACHUNKS} x WvPrGrp={WVPRGRPS}"
    )
    print(f"warmup={warmup}ms, rep={rep}ms")
    print()

    csv_file = open(csv_path, "w", newline="")  # noqa: SIM115
    writer = csv.writer(csv_file)
    writer.writerow(
        [
            "M",
            "K",
            "block_size_m",
            "label",
            "ytile",
            "unrl",
            "achunk",
            "wvprgrp",
            "time_us",
            "weight_bw_gibs",
            "is_production_pick",
        ]
    )

    best_per_shape = []
    tested = skipped = 0
    t0 = time.time()

    for M, K, label in shapes:
        if K % GROUP_SIZE != 0:
            print(f"  SKIP {M}x{K}: K not divisible by group_size={GROUP_SIZE}")
            continue

        for bsm in block_sizes:
            a, w, scales, c, expert_ids = build_moe_tensors(
                M, K, bsm, dtype, device="cuda"
            )
            weight_bytes_per_block = M * K // 2  # one expert's int4 weight

            # Production anchor — dispatcher chooses ytile/unrl
            prod_us = time_us(
                lambda a=a,
                w=w,
                scales=scales,
                c=c,
                expert_ids=expert_ids,
                bsm=bsm: ops.fused_moe_wvSplitK_int4_gemm(
                    a, w, scales, c, expert_ids, bsm, cu_count, GROUP_SIZE
                ),
                warmup=warmup,
                rep=rep,
            )
            num_blocks = expert_ids.numel()
            prod_bw = weight_bytes_per_block * num_blocks / (prod_us * 1e-6) / (1 << 30)

            shape_rows = []
            for yt, un, ac, wv in itertools.product(YTILES, UNRLS, ACHUNKS, WVPRGRPS):
                if M % yt != 0 or K % ac != 0:
                    skipped += 1
                    continue
                try:
                    us = time_us(
                        lambda yt=yt,
                        un=un,
                        ac=ac,
                        wv=wv,
                        a=a,
                        w=w,
                        scales=scales,
                        c=c,
                        expert_ids=expert_ids,
                        bsm=bsm: ops.fused_moe_wvSplitK_int4_gemm_sweep(
                            a,
                            w,
                            scales,
                            c,
                            expert_ids,
                            bsm,
                            cu_count,
                            GROUP_SIZE,
                            yt,
                            un,
                            ac,
                            wv,
                        ),
                        warmup=warmup,
                        rep=rep,
                    )
                    bw = weight_bytes_per_block * num_blocks / (us * 1e-6) / (1 << 30)
                except Exception as e:
                    print(
                        f"  ERROR bsm={bsm} {M}x{K} "
                        f"yt={yt} un={un} ac={ac} wv={wv}: {e}"
                    )
                    us, bw = float("inf"), 0.0
                writer.writerow(
                    [M, K, bsm, label, yt, un, ac, wv, f"{us:.1f}", f"{bw:.1f}", ""]
                )
                shape_rows.append((yt, un, ac, wv, us, bw))
                tested += 1

            # Production anchor row.
            writer.writerow(
                [
                    M,
                    K,
                    bsm,
                    label,
                    "",
                    "",
                    "",
                    "",
                    f"{prod_us:.1f}",
                    f"{prod_bw:.1f}",
                    "True",
                ]
            )

            if shape_rows:
                best = min(shape_rows, key=lambda r: r[4])
                speedup = prod_us / best[4] if best[4] > 0 else 0.0
                elapsed = time.time() - t0
                print(
                    f"  bsm={bsm} {M:>5}x{K:<5}  {label:<48}  "
                    f"prod={prod_us:>7.1f}us  "
                    f"best yt={best[0]} un={best[1]} ac={best[2]} wv={best[3]} "
                    f"-> {best[4]:>7.1f}us ({speedup:.2f}x)  "
                    f"[{tested} tested, {elapsed:.0f}s]"
                )
                best_per_shape.append(
                    {
                        "M": M,
                        "K": K,
                        "bsm": bsm,
                        "label": label,
                        "ytile": best[0],
                        "unrl": best[1],
                        "achunk": best[2],
                        "wvprgrp": best[3],
                        "time_us": best[4],
                        "bw_gibs": best[5],
                        "prod_us": prod_us,
                    }
                )

            csv_file.flush()

    csv_file.close()
    elapsed = time.time() - t0
    print()
    print(f"Done: {tested} combos tested, {skipped} skipped, {elapsed:.0f}s total")
    print(f"Full CSV: {csv_path}")
    print()

    print("=" * 120)
    print(
        "BEST CONFIG PER SHAPE  "
        "(fused_moe_sweep best vs production fused_moe_wvSplitK_int4_gemm)"
    )
    print("=" * 120)
    print(
        f"{'bsm':>3} {'M':>5}x{'K':<5}  {'Label':<48}  "
        f"{'yt':>3} {'un':>3} {'ac':>3} {'wv':>3}  "
        f"{'best_us':>9}  {'prod_us':>9}  {'speedup':>8}"
    )
    print("-" * 120)
    for r in best_per_shape:
        speedup = r["prod_us"] / r["time_us"] if r["time_us"] > 0 else 0.0
        print(
            f"{r['bsm']:>3} {r['M']:>5}x{r['K']:<5}  {r['label']:<48}  "
            f"{r['ytile']:>3} {r['unrl']:>3} {r['achunk']:>3} {r['wvprgrp']:>3}  "
            f"{r['time_us']:>9.1f}  {r['prod_us']:>9.1f}  {speedup:>7.2f}x"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--block-sizes",
        type=int,
        nargs="+",
        default=[1],
        help="block_size_m values to sweep (default: 1, decode)",
    )
    ap.add_argument(
        "--shapes",
        type=parse_shape,
        nargs="+",
        default=None,
        help="Override the default shape list. e.g. --shapes 2048x512",
    )
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--rep", type=int, default=100)
    ap.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    ap.add_argument("--csv", default="int4g_moe_sweep_results.csv")
    args = ap.parse_args()

    shapes = args.shapes if args.shapes else SHAPES
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    run_sweep(shapes, args.block_sizes, args.warmup, args.rep, dtype, args.csv)


if __name__ == "__main__":
    main()
