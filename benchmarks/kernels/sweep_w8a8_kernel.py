#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sweep all tuning parameters for the W8A8 skinny GEMM kernel.

Sweeps YTILE x UNRL x A_CHUNK x WvPrGrp for all shapes and batch sizes,
producing a CSV of results and summary tables.

W8A8: int8 weights, bf16/fp16 activations (fused quantization inside kernel),
per-channel weight scale (fp16/bf16), per-tensor activation scale (float32).

Usage:
    python sweep_w8a8_kernel.py
    python sweep_w8a8_kernel.py --batch-sizes 1 4
    python sweep_w8a8_kernel.py --shapes 4096x4096
    python sweep_w8a8_kernel.py --dtype bfloat16
"""

import argparse
import csv
import itertools
import os
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
    (1024, 1024, "Internal-0.9B-VLM q/o"),
    (1280, 1024, "Internal-0.9B-VLM qkv"),
    (8192, 1024, "Internal-0.9B-VLM gate_up"),
    (1024, 4096, "Internal-0.9B-VLM down"),
    (73448, 1024, "Internal-0.9B-VLM lm_head"),
    (2560, 2560, "Qwen3-4B q/o"),
    (6144, 2560, "Qwen3-4B qkv"),
    (19456, 2560, "Qwen3-4B gate_up"),
    (2560, 9728, "Qwen3-4B down"),
    (151936, 2560, "Qwen3-4B lm_head"),
    (4608, 2560, "Internal-4B-VLM qkv"),
    (20480, 2560, "Internal-4B-VLM gate_up"),
    (2560, 10240, "Internal-4B-VLM down"),
    (73448, 2560, "Internal-4B-VLM lm_head"),
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

# LDS capacity for int8 activations (1 byte each).
# gfx9: 64 KB, gfx95x: 160 KB.  Use conservative default; detect at runtime.
LDS_CAPACITY_GFX9 = 64 * 1024
LDS_CAPACITY_GFX95 = 160 * 1024


def get_lds_capacity():
    name = torch.cuda.get_device_name(0).lower()
    if "gfx95" in name or "mi355" in name or "mi350" in name:
        return LDS_CAPACITY_GFX95
    return LDS_CAPACITY_GFX9


def fits_lds(K, N, lds_cap):
    return lds_cap >= K * N


def quantize_symmetric(tensor):
    """Symmetric per-channel int8 quantization."""
    amax = tensor.abs().amax(dim=1)
    scale = (amax / 127.0).clamp(min=1e-10)
    quantized = (tensor / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
    return quantized, scale


def quantize_per_tensor(tensor):
    """Symmetric per-tensor int8 quantization."""
    amax = tensor.abs().max()
    scale = (amax / 127.0).clamp(min=1e-10)
    quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
    return quantized, scale.reshape(1)


def parse_shape(s):
    parts = s.split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Shape must be MxK, got '{s}'")
    return (int(parts[0]), int(parts[1]), s)


def run_sweep(shapes, batch_sizes, warmup, rep, dtype):
    cu_count = get_cu_count()
    gpu_name = torch.cuda.get_device_name(0)
    lds_cap = get_lds_capacity()

    total_combos = (
        len(shapes)
        * len(batch_sizes)
        * len(YTILES)
        * len(UNRLS)
        * len(ACHUNKS)
        * len(WVPRGRPS)
    )
    dtype_str = "fp16" if dtype == torch.float16 else "bf16"
    print(f"GPU: {gpu_name}, CU count: {cu_count}")
    print(f"LDS capacity: {lds_cap // 1024} KB (int8 activations, 1 byte each)")
    print(f"Output dtype: {dtype_str}")
    print(f"Shapes: {len(shapes)}, Batch sizes: {batch_sizes}")
    print(
        f"Param grid: YTILE={YTILES} x UNRL={UNRLS}"
        f" x A_CHUNK={ACHUNKS} x WvPrGrp={WVPRGRPS}"
    )
    print(f"Max combos (before filtering): {total_combos}")
    print(f"warmup={warmup}, rep={rep}")
    print()

    out_dir = os.environ.get("SWEEP_OUTPUT_DIR", "/tmp/claude")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"w8a8_sweep_{dtype_str}_results.csv")
    csv_file = open(csv_path, "w", newline="")  # noqa: SIM115
    writer = csv.writer(csv_file)
    writer.writerow(
        [
            "M",
            "K",
            "N",
            "label",
            "dtype",
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
        for N in batch_sizes:
            if not fits_lds(K, N, lds_cap):
                skipped += len(YTILES) * len(UNRLS) * len(ACHUNKS) * len(WVPRGRPS)
                continue

            # Generate data: int8 weights, bf16/fp16 activations (fused quant)
            import math

            xavier = math.sqrt(2 / K)
            W_fp = (
                torch.rand(M, K, dtype=torch.float32, device="cuda") * 2 - 1
            ) * xavier
            A_fp = (
                torch.rand(N, K, dtype=torch.float32, device="cuda") * 2 - 1
            ) * xavier

            W_int8, w_scale = quantize_symmetric(W_fp)
            _, a_scale = quantize_per_tensor(A_fp)
            w_scale_typed = w_scale.to(dtype)
            A_typed = A_fp.to(dtype)

            # Weight bytes: M * K * 1 (int8)
            weight_bytes = M * K

            shape_results = []
            for ytile, unrl, achunk, wvprgrp in itertools.product(
                YTILES, UNRLS, ACHUNKS, WVPRGRPS
            ):
                if K % achunk != 0:
                    skipped += 1
                    continue
                if M % ytile != 0:
                    skipped += 1
                    continue

                try:
                    fn = (
                        lambda yt=ytile,
                        ur=unrl,
                        ac=achunk,
                        wv=wvprgrp,
                        w=W_int8,
                        a=A_typed,
                        ws=w_scale_typed,
                        as_=a_scale: (
                            ops.wvSplitK_w8a8_sweep(
                                w,
                                a,
                                ws,
                                as_,
                                cu_count,
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
                        dtype_str,
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

            # Measure the production op too — it runs the actual C++
            # heuristic internally, so this is the timing the heuristic
            # would deliver in real use. No need to mirror the heuristic
            # in Python; the source of truth stays in C++.
            heur_time_us = float("inf")
            try:
                heur_fn = (
                    lambda w=W_int8,
                    a=A_typed,
                    ws=w_scale_typed,
                    as_=a_scale: ops.wvSplitK_w8a8(w, a, ws, as_, cu_count)
                )
                heur_ms = triton.testing.do_bench(
                    heur_fn, warmup=warmup, rep=rep, return_mode="median"
                )
                heur_time_us = heur_ms * 1000
            except Exception as e:
                print(f"  ERROR: N={N} {M}x{K} heuristic op: {e}")

            if shape_results:
                best = min(shape_results, key=lambda r: r["time_us"])
                best["heur_time_us"] = heur_time_us
                results.append(best)
                elapsed = time.time() - t0
                heur_marker = ""
                if heur_time_us > best["time_us"] * 1.05:
                    regret = (heur_time_us - best["time_us"]) / best["time_us"] * 100
                    heur_marker = f"  heur {heur_time_us:.1f} us ({regret:.0f}% slower)"
                else:
                    heur_marker = f"  heur {heur_time_us:.1f} us"
                print(
                    f"  N={N} {M:>6}x{K:<6} {label:<22} "
                    f"best: yt={best['ytile']} ur={best['unrl']} "
                    f"ac={best['achunk']} wv={best['wvprgrp']}  "
                    f"{best['time_us']:>8.1f} us  {best['bw_gibs']:>6.1f} GiB/s "
                    f"{heur_marker}  "
                    f"[{tested} tested, {elapsed:.0f}s elapsed]"
                )

            csv_file.flush()

    csv_file.close()
    elapsed = time.time() - t0
    print()
    print(f"Done: {tested} combos tested, {skipped} skipped, {elapsed:.0f}s total")
    print(f"Full CSV: {csv_path}")
    print()

    analyze_results(csv_path, results)


def analyze_results(csv_path, best_per_shape):
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

    print("=" * 100)
    print("BEST CONFIG PER SHAPE")
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

    # --- Per-parameter analysis ---
    shape_keys = set()
    for row in all_rows:
        shape_keys.add((row["M"], row["K"], row["N"], row["label"]))

    for param_name, param_values in [
        ("A_CHUNK", ACHUNKS),
        ("WvPrGrp", WVPRGRPS),
    ]:
        print()
        print("=" * 100)
        print(f"{param_name} ANALYSIS (is one value universally best?)")
        print("=" * 100)

        wins = collections.Counter()
        regret = {v: [] for v in param_values}

        for sk in sorted(shape_keys):
            M, K, N, label = sk
            rows_for_shape = [
                r for r in all_rows if r["M"] == M and r["K"] == K and r["N"] == N
            ]
            if not rows_for_shape:
                continue
            best_time = min(r["time_us"] for r in rows_for_shape)
            best_per_val = {}
            key = param_name.lower().replace("_", "")
            # map display name to csv column
            col = {"achunk": "achunk", "wvprgrp": "wvprgrp"}[key]
            for v in param_values:
                v_rows = [r for r in rows_for_shape if r[col] == v]
                if v_rows:
                    best_per_val[v] = min(r["time_us"] for r in v_rows)
            if not best_per_val:
                continue
            winner = min(best_per_val, key=best_per_val.get)
            wins[winner] += 1
            for v in param_values:
                if v in best_per_val:
                    r_pct = (best_per_val[v] - best_time) / best_time * 100
                    regret[v].append(r_pct)

        for v in param_values:
            regrets = regret[v]
            if regrets:
                avg_r = sum(regrets) / len(regrets)
                max_r = max(regrets)
                print(
                    f"  {param_name}={v:>2}: wins {wins[v]:>3} shapes, "
                    f"avg regret {avg_r:.1f}%, max regret {max_r:.1f}%"
                )

    # --- Production heuristic vs sweep best ---
    # Heuristic time was measured during the sweep loop by calling the
    # production op (ops.wvSplitK_w8a8) directly — no Python mirror of
    # the C++ heuristic. Look up by (M, K, N).
    heur_by_shape = {
        (r["M"], r["K"], r["N"]): r.get("heur_time_us", float("inf"))
        for r in best_per_shape
    }
    print()
    print("=" * 100)
    print("HEURISTIC ANALYSIS (best per shape vs measured production heuristic)")
    print("=" * 100)

    for sk in sorted(shape_keys):
        M, K, N, label = sk
        rows_for_shape = [
            r for r in all_rows if r["M"] == M and r["K"] == K and r["N"] == N
        ]
        if not rows_for_shape:
            continue
        best = min(rows_for_shape, key=lambda r: r["time_us"])
        heur_time = heur_by_shape.get((M, K, N), float("inf"))

        marker = ""
        heur_str = "N/A"
        if heur_time != float("inf"):
            heur_str = f"{heur_time:.1f} us"
            if heur_time > best["time_us"] * 1.05:
                regret = (heur_time - best["time_us"]) / best["time_us"] * 100
                marker = f"  <-- heuristic {regret:.0f}% slower"

        print(
            f"  N={N} {M:>6}x{K:<6} {label:<22} "
            f"best: yt={best['ytile']} ur={best['unrl']} "
            f"ac={best['achunk']} wv={best['wvprgrp']} "
            f"({best['time_us']:.1f} us)  heur: {heur_str}{marker}"
        )

    print()
    out_dir = os.environ.get("SWEEP_OUTPUT_DIR", "/tmp/claude")
    summary_path = os.path.join(out_dir, "w8a8_sweep_summary.txt")
    with open(summary_path, "w") as f:
        f.write("W8A8 skinny GEMM sweep summary\n")
        for r in best_per_shape:
            f.write(
                f"N={r['N']} {r['M']}x{r['K']} {r['label']}: "
                f"yt={r['ytile']} ur={r['unrl']} ac={r['achunk']} wv={r['wvprgrp']} "
                f"{r['time_us']:.1f} us {r['bw_gibs']:.1f} GiB/s\n"
            )
    print(f"Summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Sweep W8A8 skinny GEMM kernel tuning parameters"
    )
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
        "--dtype",
        choices=["float16", "bfloat16"],
        default="float16",
        help="Output dtype (default: float16)",
    )
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations")
    parser.add_argument("--rep", type=int, default=100, help="Benchmark repetitions")
    args = parser.parse_args()

    shapes = args.shapes if args.shapes else SHAPES
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    run_sweep(shapes, args.batch_sizes, args.warmup, args.rep, dtype)


if __name__ == "__main__":
    main()
