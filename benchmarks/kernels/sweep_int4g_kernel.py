#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sweep all tuning parameters for the int4 grouped (group_size=128) skinny GEMM kernel.

Sweeps YTILE x UNRL x A_CHUNK x WvPrGrp for all shapes and batch sizes,
producing a CSV of results and summary tables.

Usage:
    python sweep_int4g_kernel.py
    python sweep_int4g_kernel.py --batch-sizes 1 4
    python sweep_int4g_kernel.py --shapes 151936x2560
    python sweep_int4g_kernel.py --medium   # sweep medium (hf) variant only
"""

import argparse
import csv
import itertools
import time

import torch

import vllm._custom_ops as ops
from vllm.triton_utils import triton
from vllm.utils.platform_utils import num_compute_units as get_cu_count

SHAPES = [
    (9728, 896, "Qwen0.5B gate_up"),
    (2048, 2048, "Gemma2B q/o"),
    (2560, 2048, "Gemma2B qkv"),
    (32768, 2048, "Gemma2B gate_up"),
    (2048, 16384, "Gemma2B down"),
    (4096, 2048, "Qwen1.7B qkv"),
    (12288, 2048, "Qwen1.7B gate_up"),
    (2048, 6144, "Qwen1.7B down"),
    (2560, 2560, "Qwen3-4B q/o"),
    (6144, 2560, "Qwen3-4B qkv"),
    (19456, 2560, "Qwen3-4B gate_up"),
    (2560, 9728, "Qwen3-4B down"),
    (151936, 2560, "Qwen3-4B lm_head"),
    (22016, 2048, "Qwen2.5VL-3B gate_up"),
    (2048, 11008, "Qwen2.5VL-3B down"),
    (3584, 3584, "Qwen7B q/o"),
    (4608, 3584, "Qwen7B qkv"),
    (37888, 3584, "Qwen7B gate_up"),
    (3584, 18944, "Qwen7B down"),
    (152064, 3584, "Qwen7B lm_head"),
    (4096, 4096, "LLaMA8B q/o"),
    (6144, 4096, "LLaMA8B qkv"),
    (28672, 4096, "LLaMA8B gate_up"),
    (4096, 14336, "LLaMA8B down"),
    (11008, 4096, "LLaMA2-7B up/gate"),
    (22016, 4096, "LLaMA2-7B gate_up"),
    (4096, 11008, "LLaMA2-7B down"),
]

YTILES = [1, 2, 4]
UNRLS = [1, 2, 4]
ACHUNKS = [8, 16, 32]
WVPRGRPS = [8, 12, 16]

GROUP_SIZE = 128
LDS_CAPACITY = 64 * 1024 // 2
LDS_MEDIUM = int(LDS_CAPACITY * 1.2)


def fits_sml(K, N):
    return K * N <= LDS_CAPACITY


def fits_medium(K, N):
    return K * N <= LDS_MEDIUM


def pack_int4(values_int8):
    """Pack signed int4 values using ExLlama shuffle for fast fp16 dequant."""
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


def parse_shape(s):
    parts = s.split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Shape must be MxK, got '{s}'")
    return (int(parts[0]), int(parts[1]), s)


def run_sweep(shapes, batch_sizes, warmup, rep, medium_only=False):
    cu_count = get_cu_count()
    gpu_name = torch.cuda.get_device_name(0)
    dtype = torch.float16

    variant_label = "medium (hf)" if medium_only else "sml"
    total_combos = (
        len(shapes)
        * len(batch_sizes)
        * len(YTILES)
        * len(UNRLS)
        * len(ACHUNKS)
        * len(WVPRGRPS)
    )
    print(f"GPU: {gpu_name}, CU count: {cu_count}")
    print(f"Kernel variant: {variant_label}")
    print(f"Shapes: {len(shapes)}, Batch sizes: {batch_sizes}")
    print(f"Group size: {GROUP_SIZE}")
    print(
        f"Param grid: YTILE={YTILES} x UNRL={UNRLS}"
        f" x A_CHUNK={ACHUNKS} x WvPrGrp={WVPRGRPS}"
    )
    print(f"Max combos (before filtering): {total_combos}")
    print(f"warmup={warmup}, rep={rep}")
    print()

    suffix = "_medium" if medium_only else ""
    csv_path = f"/scratch/mgehre/tmp/int4g128_sweep{suffix}_results.csv"
    csv_file = open(csv_path, "w", newline="")  # noqa: SIM115
    writer = csv.writer(csv_file)
    writer.writerow(
        [
            "M",
            "K",
            "N",
            "label",
            "ytile",
            "unrl",
            "achunk",
            "wvprgrp",
            "time_us",
            "weight_bw_gibs",
        ]
    )

    results = []
    skipped = 0
    tested = 0
    t0 = time.time()

    for M, K, label in shapes:
        if K % GROUP_SIZE != 0:
            for N in batch_sizes:
                skipped += len(YTILES) * len(UNRLS) * len(ACHUNKS) * len(WVPRGRPS)
            continue

        num_groups = K // GROUP_SIZE

        for N in batch_sizes:
            if medium_only:
                if fits_sml(K, N) or not fits_medium(K, N):
                    skipped += len(YTILES) * len(UNRLS) * len(ACHUNKS) * len(WVPRGRPS)
                    continue
            else:
                if not fits_sml(K, N):
                    skipped += len(YTILES) * len(UNRLS) * len(ACHUNKS) * len(WVPRGRPS)
                    continue

            values_int4 = torch.randint(-8, 8, (M, K), dtype=torch.int8, device="cuda")
            weight_packed = pack_int4(values_int4)
            scale = torch.rand(M, num_groups, dtype=dtype, device="cuda") * 0.02 - 0.01
            activation = (torch.rand(N, K, dtype=dtype, device="cuda") - 0.5) * 0.01

            weight_bytes = M * K // 2

            sweep_fn = (
                ops.wvSplitK_int4g_hf_sweep if medium_only else ops.wvSplitK_int4g_sweep
            )

            shape_results = []
            for ytile, unrl, achunk, wvprgrp in itertools.product(
                YTILES, UNRLS, ACHUNKS, WVPRGRPS
            ):
                if K % achunk != 0:
                    skipped += 1
                    continue
                if not medium_only and M % ytile != 0:
                    skipped += 1
                    continue

                try:
                    fn = (
                        lambda yt=ytile,
                        ur=unrl,
                        ac=achunk,
                        wv=wvprgrp,
                        w=weight_packed,
                        a=activation,
                        s=scale,
                        _sf=sweep_fn: (
                            _sf(
                                w,
                                a,
                                s,
                                cu_count,
                                GROUP_SIZE,
                                yt,
                                ur,
                                ac,
                                wv,
                            )
                        )
                    )
                    ms = triton.testing.do_bench(
                        fn, warmup=warmup, rep=rep, return_mode="median"
                    )
                    time_us = ms * 1000
                    bw_gibs = weight_bytes / (ms * 1e-3) / (1 << 30)
                except Exception as e:
                    time_us = float("inf")
                    bw_gibs = 0.0
                    print(
                        f"  ERROR: N={N} {M}x{K} yt={ytile}"
                        f" ur={unrl} ac={achunk} wv={wvprgrp}: {e}"
                    )

                writer.writerow(
                    [
                        M,
                        K,
                        N,
                        label,
                        ytile,
                        unrl,
                        achunk,
                        wvprgrp,
                        f"{time_us:.1f}",
                        f"{bw_gibs:.1f}",
                    ]
                )
                shape_results.append(
                    {
                        "M": M,
                        "K": K,
                        "N": N,
                        "label": label,
                        "ytile": ytile,
                        "unrl": unrl,
                        "achunk": achunk,
                        "wvprgrp": wvprgrp,
                        "time_us": time_us,
                        "bw_gibs": bw_gibs,
                    }
                )
                tested += 1

            if shape_results:
                best = min(shape_results, key=lambda r: r["time_us"])
                results.append(best)
                elapsed = time.time() - t0
                print(
                    f"  N={N} {M:>6}x{K:<6} {label:<22} "
                    f"best: yt={best['ytile']} ur={best['unrl']} "
                    f"ac={best['achunk']} wv={best['wvprgrp']}  "
                    f"{best['time_us']:>8.1f} us  {best['bw_gibs']:>6.1f} GiB/s  "
                    f"[{tested} tested, {elapsed:.0f}s elapsed]"
                )

            csv_file.flush()

    csv_file.close()
    elapsed = time.time() - t0
    print()
    print(f"Done: {tested} combos tested, {skipped} skipped, {elapsed:.0f}s total")
    print(f"Full CSV: {csv_path}")
    print()

    analyze_results(csv_path, results, medium_only)


def analyze_results(csv_path, best_per_shape, medium_only=False):
    import collections

    all_rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["time_us"] = float(row["time_us"])
            row["weight_bw_gibs"] = float(row["weight_bw_gibs"])
            row["M"] = int(row["M"])
            row["K"] = int(row["K"])
            row["N"] = int(row["N"])
            row["ytile"] = int(row["ytile"])
            row["unrl"] = int(row["unrl"])
            row["achunk"] = int(row["achunk"])
            row["wvprgrp"] = int(row["wvprgrp"])
            all_rows.append(row)

    variant_label = "MEDIUM (hf)" if medium_only else "SML"
    print("=" * 100)
    print(f"BEST CONFIG PER SHAPE [{variant_label}]")
    print("=" * 100)
    print(
        f"{'N':>2} {'M':>6}x{'K':<6} {'Label':<22} "
        f"{'yt':>3} {'ur':>3} {'ac':>3} {'wv':>3} "
        f"{'time_us':>9} {'BW GiB/s':>9}"
    )
    print("-" * 100)
    for r in best_per_shape:
        print(
            f"{r['N']:>2} {r['M']:>6}x{r['K']:<6} {r['label']:<22} "
            f"{r['ytile']:>3} {r['unrl']:>3} {r['achunk']:>3} {r['wvprgrp']:>3} "
            f"{r['time_us']:>9.1f} {r['bw_gibs']:>9.1f}"
        )

    print()
    print("=" * 100)
    print("A_CHUNK ANALYSIS (is one value universally best?)")
    print("=" * 100)
    shape_keys = set()
    for row in all_rows:
        shape_keys.add((row["M"], row["K"], row["N"], row["label"]))

    achunk_wins = collections.Counter()
    achunk_regret = {ac: [] for ac in ACHUNKS}

    for sk in sorted(shape_keys):
        M, K, N, label = sk
        rows_for_shape = [
            r for r in all_rows if r["M"] == M and r["K"] == K and r["N"] == N
        ]
        if not rows_for_shape:
            continue
        best_time = min(r["time_us"] for r in rows_for_shape)
        best_per_ac = {}
        for ac in ACHUNKS:
            ac_rows = [r for r in rows_for_shape if r["achunk"] == ac]
            if ac_rows:
                best_per_ac[ac] = min(r["time_us"] for r in ac_rows)
        if not best_per_ac:
            continue
        winner_ac = min(best_per_ac, key=best_per_ac.get)
        achunk_wins[winner_ac] += 1
        for ac in ACHUNKS:
            if ac in best_per_ac:
                regret = (best_per_ac[ac] - best_time) / best_time * 100
                achunk_regret[ac].append(regret)

    for ac in ACHUNKS:
        regrets = achunk_regret[ac]
        if regrets:
            avg_r = sum(regrets) / len(regrets)
            max_r = max(regrets)
            print(
                f"  A_CHUNK={ac:>2}: wins {achunk_wins[ac]:>3} shapes, "
                f"avg regret {avg_r:.1f}%, max regret {max_r:.1f}%"
            )

    print()
    print("=" * 100)
    print("WvPrGrp ANALYSIS (is one value universally best?)")
    print("=" * 100)

    wv_wins = collections.Counter()
    wv_regret = {wv: [] for wv in WVPRGRPS}

    for sk in sorted(shape_keys):
        M, K, N, label = sk
        rows_for_shape = [
            r for r in all_rows if r["M"] == M and r["K"] == K and r["N"] == N
        ]
        if not rows_for_shape:
            continue
        best_time = min(r["time_us"] for r in rows_for_shape)
        best_per_wv = {}
        for wv in WVPRGRPS:
            wv_rows = [r for r in rows_for_shape if r["wvprgrp"] == wv]
            if wv_rows:
                best_per_wv[wv] = min(r["time_us"] for r in wv_rows)
        if not best_per_wv:
            continue
        winner_wv = min(best_per_wv, key=best_per_wv.get)
        wv_wins[winner_wv] += 1
        for wv in WVPRGRPS:
            if wv in best_per_wv:
                regret = (best_per_wv[wv] - best_time) / best_time * 100
                wv_regret[wv].append(regret)

    for wv in WVPRGRPS:
        regrets = wv_regret[wv]
        if regrets:
            avg_r = sum(regrets) / len(regrets)
            max_r = max(regrets)
            print(
                f"  WvPrGrp={wv:>2}: wins {wv_wins[wv]:>3} shapes, "
                f"avg regret {avg_r:.1f}%, max regret {max_r:.1f}%"
            )

    print()
    print("=" * 100)
    print("YTILE/UNRL ANALYSIS (best per shape vs current heuristic)")
    print("=" * 100)

    cu_count = get_cu_count()
    for sk in sorted(shape_keys):
        M, K, N, label = sk
        rows_for_shape = [
            r for r in all_rows if r["M"] == M and r["K"] == K and r["N"] == N
        ]
        if not rows_for_shape:
            continue
        best = min(rows_for_shape, key=lambda r: r["time_us"])

        sYT = (M + cu_count * 4 - 1) // (cu_count * 4)
        if N >= 4 and sYT >= 480:
            heur_yt, heur_ur = 4, 1
        elif N >= 2:
            heur_yt, heur_ur = 2, 2
        elif sYT >= 30:
            heur_yt, heur_ur = 2, 4
        else:
            heur_yt, heur_ur = 1, 4

        heur_rows = [
            r for r in rows_for_shape if r["ytile"] == heur_yt and r["unrl"] == heur_ur
        ]
        heur_best = min(heur_rows, key=lambda r: r["time_us"]) if heur_rows else None

        marker = ""
        if heur_best and heur_best["time_us"] > best["time_us"] * 1.05:
            regret = (heur_best["time_us"] - best["time_us"]) / best["time_us"] * 100
            marker = f"  <-- heuristic {regret:.0f}% slower"

        heur_str = f" ({heur_best['time_us']:.1f} us)" if heur_best else " (N/A)"
        print(
            f"  N={N} {M:>6}x{K:<6} {label:<22} "
            f"best: yt={best['ytile']} ur={best['unrl']} "
            f"ac={best['achunk']} wv={best['wvprgrp']} "
            f"({best['time_us']:.1f} us) "
            f"heur: yt={heur_yt} ur={heur_ur}{heur_str}{marker}"
        )

    print()
    suffix = "_medium" if medium_only else ""
    summary_path = f"/scratch/mgehre/tmp/int4g128_sweep{suffix}_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Int4 group-128 {variant_label} sweep summary\n")
        for r in best_per_shape:
            f.write(
                f"N={r['N']} {r['M']}x{r['K']} {r['label']}: "
                f"yt={r['ytile']} ur={r['unrl']} ac={r['achunk']} wv={r['wvprgrp']} "
                f"{r['time_us']:.1f} us {r['bw_gibs']:.1f} GiB/s\n"
            )
    print(f"Summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Sweep int4g kernel tuning parameters")
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 4],
        help="Batch sizes (N) to sweep (default: 1 4)",
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        type=parse_shape,
        default=None,
        help="Shapes as MxK (default: all built-in shapes)",
    )
    parser.add_argument(
        "--medium",
        action="store_true",
        help="Sweep medium (hf) kernel variant for shapes where K*N exceeds sml LDS",
    )
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations")
    parser.add_argument("--rep", type=int, default=100, help="Benchmark repetitions")
    args = parser.parse_args()

    shapes = args.shapes if args.shapes else SHAPES
    run_sweep(shapes, args.batch_sizes, args.warmup, args.rep, medium_only=args.medium)


if __name__ == "__main__":
    main()
