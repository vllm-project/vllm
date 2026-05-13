#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sweep tuning parameters for the bf16/fp16 wvSplitK skinny GEMM kernel.

Mirrors ``sweep_int4g_kernel.py`` / ``sweep_w8a8_kernel.py`` for the plain
(unquantized) wvSplitK path used by every Linear that hasn't been replaced
by an int4/int8/AWQ wrapper.

Limitation: the in-tree ``wvSplitK_sweep`` op (csrc/rocm/skinny_gemms.cu)
currently exposes only ``(ytile, unrl)`` as runtime args; ``WvPrGrp`` and
``A_CHUNK`` are macro-baked at 16 and 8 respectively. The script therefore
sweeps only YTILE x UNRL. For 4-axis sweeps (the (W, AC) corner of the
search space) use the out-of-tree HIP harness in
``rdna35-asm-expert/microbench/wv_splitk/`` until ``wvSplitK_sweep`` is
extended to match ``wvSplitK_int4g_sweep`` (which already takes all 4 axes).

For each (M, K, N) cell the script also calls the production ``wvSplitK``
op once to record the dispatcher's chosen wall-clock — useful both as a
baseline for the sweep and as the verification anchor any external rig
(microbench, custom harness) should match.

Usage:
    python sweep_bf16_kernel.py
    python sweep_bf16_kernel.py --batch-sizes 1 4
    python sweep_bf16_kernel.py --shapes 1024x2048      # Qwen3.5-A3B shared gate_up
    python sweep_bf16_kernel.py --shapes 1024x4096      # K=4096 corner
"""

import argparse
import csv
import itertools
import time

import torch

import vllm._custom_ops as ops
from vllm.triton_utils import triton
from vllm.utils.platform_utils import num_compute_units as get_cu_count

# Shape list: (M = output_features, K = input_features, label).  Mirrors the
# int4g/w8a8 sweep coverage plus the Qwen3.5-A3B-specific bf16 callers (its
# int4 build only hits the bf16 wvSplitK path on the shared-expert MLP and,
# when --dynamic-lm-head-quantization is off, the lm_head).
SHAPES = [
    # --- Qwen3.5-A3B bf16 wvSplitK callers (the model this PR was tuned for) ---
    (256, 2048, "Qwen3.5-A3B router gate"),
    (1024, 2048, "Qwen3.5-A3B shared gate_up"),
    (2048, 512, "Qwen3.5-A3B shared down"),
    (248320, 2048, "Qwen3.5-A3B lm_head (bf16 path)"),
    # --- Broader coverage (mirrors sweep_int4g_kernel.py) ---
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
    (4096, 4096, "LLaMA8B q/o"),
    (6144, 4096, "LLaMA8B qkv"),
    (28672, 4096, "LLaMA8B gate_up"),
    (4096, 14336, "LLaMA8B down"),
    (1024, 4096, "K=4096 small-M sanity"),
]

YTILES = [1, 2, 3, 4]
UNRLS = [1, 2, 4]

# Working-set guards used by the dispatcher's variant selection
# (wvSplitK_hf_sml_ vs wvSplitK_hf_ vs wvSplitK_hf_big_).  bf16 / fp16
# weights are 2 bytes; the kernel reserves half the LDS for activations.
LDS_CAPACITY_BYTES = 64 * 1024


def fits_sml(K: int, N: int) -> bool:
    """Match WVSPLIT_TILE_CFG's sml fit check: Kbp * N <= max_lds_len/2."""
    return K * N * 2 <= LDS_CAPACITY_BYTES


def parse_shape(s: str) -> tuple[int, int, str]:
    parts = s.split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Shape must be MxK, got '{s}'")
    return (int(parts[0]), int(parts[1]), s)


def time_us(fn, warmup: int, rep: int) -> float:
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")
    return ms * 1000.0


def run_sweep(shapes, batch_sizes, warmup, rep, dtype, csv_path):
    cu_count = get_cu_count()
    gpu_name = torch.cuda.get_device_name(0)

    print(f"GPU: {gpu_name}, CU count: {cu_count}")
    print(f"Dtype: {dtype}")
    print(f"Shapes: {len(shapes)}, Batch sizes: {batch_sizes}")
    print(
        f"Param grid: YTILE={YTILES} x UNRL={UNRLS}  "
        f"(WvPrGrp and A_CHUNK are macro-baked at 16/8 in wvSplitK_sweep)"
    )
    print(f"warmup={warmup}ms, rep={rep}ms")
    print()

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
            "time_us",
            "weight_bw_gibs",
            "is_dispatcher_pick",
        ]
    )

    best_per_shape = []
    tested = skipped = 0
    t0 = time.time()

    for M, K, label in shapes:
        for N in batch_sizes:
            if not fits_sml(K, N):
                # The big_ variant also exists; skip for now -- the dispatcher
                # picks it transparently and the in-tree sweep op covers all
                # three variants under the hood.  Just noting it for the user.
                pass

            A = torch.randn(M, K, dtype=dtype, device="cuda") * 0.01
            B = torch.randn(N, K, dtype=dtype, device="cuda") * 0.01
            weight_bytes = M * K * A.element_size()

            # Production dispatcher pick (single measurement, used as anchor).
            prod_us = time_us(
                lambda a=A, b=B: ops.wvSplitK(a, b, cu_count=cu_count),
                warmup=warmup,
                rep=rep,
            )
            prod_bw = weight_bytes / (prod_us * 1e-6) / (1 << 30)

            shape_rows = []
            for ytile, unrl in itertools.product(YTILES, UNRLS):
                if M % ytile != 0:
                    skipped += 1
                    continue
                try:
                    us = time_us(
                        lambda yt=ytile, ur=unrl, a=A, b=B: ops.wvSplitK_sweep(
                            a, b, cu_count, yt, ur
                        ),
                        warmup=warmup,
                        rep=rep,
                    )
                    bw = weight_bytes / (us * 1e-6) / (1 << 30)
                except Exception as e:
                    print(f"  ERROR: N={N} {M}x{K} yt={ytile} ur={unrl}: {e}")
                    us = float("inf")
                    bw = 0.0
                writer.writerow(
                    [M, K, N, label, ytile, unrl, f"{us:.1f}", f"{bw:.1f}", ""]
                )
                shape_rows.append((ytile, unrl, us, bw))
                tested += 1

            # Production-pick anchor row.
            writer.writerow(
                [M, K, N, label, "", "", f"{prod_us:.1f}", f"{prod_bw:.1f}", "True"]
            )

            if shape_rows:
                best = min(shape_rows, key=lambda r: r[2])
                speedup = prod_us / best[2] if best[2] > 0 else 0.0
                elapsed = time.time() - t0
                print(
                    f"  N={N} {M:>6}x{K:<6} {label:<38} "
                    f"prod={prod_us:>7.1f}us  "
                    f"best yt={best[0]} ur={best[1]} -> {best[2]:>7.1f}us "
                    f"({speedup:.2f}x)  [{tested} tested, {elapsed:.0f}s]"
                )
                best_per_shape.append(
                    {
                        "M": M,
                        "K": K,
                        "N": N,
                        "label": label,
                        "ytile": best[0],
                        "unrl": best[1],
                        "time_us": best[2],
                        "bw_gibs": best[3],
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

    print("=" * 110)
    print(
        "BEST CONFIG PER SHAPE  (wvSplitK_sweep best vs production wvSplitK dispatcher)"
    )
    print("=" * 110)
    print(
        f"{'N':>2} {'M':>6}x{'K':<6}  {'Label':<38}  "
        f"{'yt':>3} {'ur':>3}  {'best_us':>9}  {'prod_us':>9}  {'speedup':>8}"
    )
    print("-" * 110)
    for r in best_per_shape:
        speedup = r["prod_us"] / r["time_us"] if r["time_us"] > 0 else 0.0
        print(
            f"{r['N']:>2} {r['M']:>6}x{r['K']:<6}  {r['label']:<38}  "
            f"{r['ytile']:>3} {r['unrl']:>3}  "
            f"{r['time_us']:>9.1f}  {r['prod_us']:>9.1f}  {speedup:>7.2f}x"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1],
        help="N values to sweep (default: 1, the decode-token case)",
    )
    ap.add_argument(
        "--shapes",
        type=parse_shape,
        nargs="+",
        default=None,
        help="Override the default shape list. e.g. --shapes 1024x4096",
    )
    ap.add_argument(
        "--warmup", type=int, default=25, help="do_bench warmup time in ms (default 25)"
    )
    ap.add_argument(
        "--rep", type=int, default=100, help="do_bench rep time in ms (default 100)"
    )
    ap.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    ap.add_argument(
        "--csv", default="bf16_wvsplitk_sweep_results.csv", help="Output CSV path"
    )
    args = ap.parse_args()

    shapes = args.shapes if args.shapes else SHAPES
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    run_sweep(shapes, args.batch_sizes, args.warmup, args.rep, dtype, args.csv)


if __name__ == "__main__":
    main()
