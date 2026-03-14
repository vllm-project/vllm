# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark: Fused FP8 output quantization in merge_attn_states

Compares three approaches for producing FP8-quantized merged attention output:
  1. CUDA  fused  -- single CUDA kernel (merge + FP8 quant)
  2. Triton fused -- single Triton kernel (merge + FP8 quant)
  3. Unfused       -- CUDA merge (BF16 output) + separate scaled_fp8_quant

Also benchmarks the non-quantized (BF16/FP16/FP32) merge kernels for
reference.

Usage:
    python benchmarks/fused_kernels/merge_attn_states_benchmarks.py
"""

import argparse
import itertools
from dataclasses import dataclass

import torch

import vllm._custom_ops as ops
from vllm._custom_ops import merge_attn_states as merge_attn_states_cuda
from vllm.platforms import current_platform
from vllm.v1.attention.ops.triton_merge_attn_states import (
    merge_attn_states as merge_attn_states_triton,
)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

NUM_TOKENS_LIST = [1, 16, 64, 256, 1024, 4096]

# (label, num_heads, head_size)
HEAD_CONFIGS = [
    ("DeepSeek-V3 MLA", 128, 128),
    ("Llama-70B", 64, 128),
    ("Llama-8B", 32, 128),
]

INPUT_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

DEFAULT_WARMUP = 10
DEFAULT_ITERS = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    config_label: str
    num_tokens: int
    num_heads: int
    head_size: int
    input_dtype: torch.dtype
    mode: str  # "fp8_fused_cuda", "fp8_fused_triton", "fp8_unfused", etc.
    time_us: float


def short_dtype(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def make_inputs(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
):
    """Create random prefix/suffix outputs and LSEs."""
    prefix_output = torch.randn(
        (num_tokens, num_heads, head_size), dtype=dtype, device="cuda"
    )
    suffix_output = torch.randn(
        (num_tokens, num_heads, head_size), dtype=dtype, device="cuda"
    )
    prefix_lse = torch.randn(num_heads, num_tokens, dtype=torch.float32, device="cuda")
    suffix_lse = torch.randn(num_heads, num_tokens, dtype=torch.float32, device="cuda")
    # Sprinkle some inf values to exercise edge-case paths
    mask = torch.rand(num_heads, num_tokens, device="cuda") < 0.05
    prefix_lse[mask] = float("inf")
    mask2 = torch.rand(num_heads, num_tokens, device="cuda") < 0.05
    suffix_lse[mask2] = float("inf")
    return prefix_output, suffix_output, prefix_lse, suffix_lse


def bench_fn(fn, warmup: int, iters: int) -> float:
    """Time *fn* using CUDA events. Returns mean elapsed time in
    microseconds."""
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()

    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)
    times_ms: list[float] = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.accelerator.synchronize()
        times_ms.append(start.elapsed_time(end))

    mean_ms = sum(times_ms) / len(times_ms)
    return mean_ms * 1000.0  # convert to microseconds


# ---------------------------------------------------------------------------
# Benchmark routines
# ---------------------------------------------------------------------------


def bench_nonfp8(
    config_label: str,
    num_tokens: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> list[BenchResult]:
    """Benchmark non-quantized merge_attn_states (CUDA vs Triton)."""
    prefix_out, suffix_out, prefix_lse, suffix_lse = make_inputs(
        num_tokens, num_heads, head_size, dtype
    )
    output_cuda = torch.empty_like(prefix_out)
    output_triton = torch.empty_like(prefix_out)

    t_cuda = bench_fn(
        lambda: merge_attn_states_cuda(
            output_cuda, prefix_out, prefix_lse, suffix_out, suffix_lse
        ),
        warmup,
        iters,
    )
    t_triton = bench_fn(
        lambda: merge_attn_states_triton(
            output_triton, prefix_out, prefix_lse, suffix_out, suffix_lse
        ),
        warmup,
        iters,
    )

    return [
        BenchResult(
            config_label,
            num_tokens,
            num_heads,
            head_size,
            dtype,
            f"cuda_{short_dtype(dtype)}",
            t_cuda,
        ),
        BenchResult(
            config_label,
            num_tokens,
            num_heads,
            head_size,
            dtype,
            f"triton_{short_dtype(dtype)}",
            t_triton,
        ),
    ]


def bench_fp8_fused_vs_unfused(
    config_label: str,
    num_tokens: int,
    num_heads: int,
    head_size: int,
    input_dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> list[BenchResult]:
    """Benchmark fused FP8 merge (CUDA & Triton) vs unfused merge+quant."""
    fp8_dtype = current_platform.fp8_dtype()
    prefix_out, suffix_out, prefix_lse, suffix_lse = make_inputs(
        num_tokens, num_heads, head_size, input_dtype
    )
    output_scale = torch.tensor([0.1], dtype=torch.float32, device="cuda")

    # -- Fused CUDA --
    output_fused_cuda = torch.empty(
        (num_tokens, num_heads, head_size), dtype=fp8_dtype, device="cuda"
    )
    t_fused_cuda = bench_fn(
        lambda: merge_attn_states_cuda(
            output_fused_cuda,
            prefix_out,
            prefix_lse,
            suffix_out,
            suffix_lse,
            output_scale=output_scale,
        ),
        warmup,
        iters,
    )

    # -- Fused Triton --
    output_fused_triton = torch.empty(
        (num_tokens, num_heads, head_size), dtype=fp8_dtype, device="cuda"
    )
    t_fused_triton = bench_fn(
        lambda: merge_attn_states_triton(
            output_fused_triton,
            prefix_out,
            prefix_lse,
            suffix_out,
            suffix_lse,
            output_scale=output_scale,
        ),
        warmup,
        iters,
    )

    # -- Unfused CUDA: CUDA merge (input_dtype output) + scaled_fp8_quant --
    output_merge_buf_cuda = torch.empty(
        (num_tokens, num_heads, head_size), dtype=input_dtype, device="cuda"
    )
    merge_buf_cuda_2d = output_merge_buf_cuda.view(-1, head_size)
    quant_scale = output_scale.clone()

    def unfused_cuda_fn():
        merge_attn_states_cuda(
            output_merge_buf_cuda, prefix_out, prefix_lse, suffix_out, suffix_lse
        )
        ops.scaled_fp8_quant(merge_buf_cuda_2d, quant_scale)

    t_unfused_cuda = bench_fn(unfused_cuda_fn, warmup, iters)

    # -- Unfused Triton: Triton merge (input_dtype output) + scaled_fp8_quant --
    output_merge_buf_triton = torch.empty(
        (num_tokens, num_heads, head_size), dtype=input_dtype, device="cuda"
    )
    merge_buf_triton_2d = output_merge_buf_triton.view(-1, head_size)
    quant_scale_triton = output_scale.clone()

    def unfused_triton_fn():
        merge_attn_states_triton(
            output_merge_buf_triton, prefix_out, prefix_lse, suffix_out, suffix_lse
        )
        ops.scaled_fp8_quant(merge_buf_triton_2d, quant_scale_triton)

    t_unfused_triton = bench_fn(unfused_triton_fn, warmup, iters)

    return [
        BenchResult(
            config_label,
            num_tokens,
            num_heads,
            head_size,
            input_dtype,
            "fp8_fused_cuda",
            t_fused_cuda,
        ),
        BenchResult(
            config_label,
            num_tokens,
            num_heads,
            head_size,
            input_dtype,
            "fp8_fused_triton",
            t_fused_triton,
        ),
        BenchResult(
            config_label,
            num_tokens,
            num_heads,
            head_size,
            input_dtype,
            "fp8_unfused_cuda",
            t_unfused_cuda,
        ),
        BenchResult(
            config_label,
            num_tokens,
            num_heads,
            head_size,
            input_dtype,
            "fp8_unfused_triton",
            t_unfused_triton,
        ),
    ]


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------


def print_fp8_table(results: list[BenchResult]):
    """Print a comparison table: fused vs unfused for both CUDA and Triton."""
    from collections import defaultdict

    groups: dict[tuple, dict[str, float]] = defaultdict(dict)
    meta: dict[tuple, tuple] = {}
    for r in results:
        key = (r.config_label, r.num_tokens, r.input_dtype)
        groups[key][r.mode] = r.time_us
        meta[key] = (r.num_heads, r.head_size)

    header = (
        f"{'Config':<18} {'Tokens':>6} {'Heads':>5} {'Hdim':>5} "
        f"{'InDtype':<10} "
        f"{'FuCUDA':>9} {'UnfCUDA':>10} {'SpdCUDA':>8} "
        f"{'FuTriton':>11} {'UnfTriton':>12} {'SpdTriton':>10}"
    )
    sep = "-" * len(header)

    print("\n" + sep)
    print("FP8 Fused vs Unfused merge_attn_states")
    print(sep)
    print(header)
    print(sep)

    for key in sorted(groups.keys(), key=lambda k: (k[0], str(k[2]), k[1])):
        config_label, num_tokens, input_dtype = key
        num_heads, head_size = meta[key]
        g = groups[key]
        t_fused_cuda = g.get("fp8_fused_cuda", float("nan"))
        t_fused_triton = g.get("fp8_fused_triton", float("nan"))
        t_unfused_cuda = g.get("fp8_unfused_cuda", float("nan"))
        t_unfused_triton = g.get("fp8_unfused_triton", float("nan"))
        spd_cuda = t_unfused_cuda / t_fused_cuda if t_fused_cuda > 0 else float("nan")
        spd_triton = (
            t_unfused_triton / t_fused_triton if t_fused_triton > 0 else float("nan")
        )
        print(
            f"{config_label:<18} {num_tokens:>6} {num_heads:>5} "
            f"{head_size:>5} "
            f"{short_dtype(input_dtype):<10} "
            f"{t_fused_cuda:>7.1f}us {t_unfused_cuda:>8.1f}us "
            f"{spd_cuda:>7.2f}x "
            f"{t_fused_triton:>9.1f}us {t_unfused_triton:>10.1f}us "
            f"{spd_triton:>9.2f}x"
        )
    print(sep)


def print_nonfp8_table(results: list[BenchResult]):
    """Print a comparison table for non-quantized merge: CUDA vs Triton."""
    from collections import defaultdict

    groups: dict[tuple, dict[str, float]] = defaultdict(dict)
    meta: dict[tuple, tuple] = {}
    for r in results:
        key = (r.config_label, r.num_tokens, r.input_dtype)
        groups[key][r.mode] = r.time_us
        meta[key] = (r.num_heads, r.head_size)

    header = (
        f"{'Config':<18} {'Tokens':>6} {'Heads':>5} {'Hdim':>5} "
        f"{'Dtype':<10} "
        f"{'CUDA':>11} {'Triton':>11} {'Speedup':>8}"
    )
    sep = "-" * len(header)

    print("\n" + sep)
    print("Non-FP8 merge_attn_states: CUDA vs Triton")
    print(sep)
    print(header)
    print(sep)

    for key in sorted(groups.keys(), key=lambda k: (k[0], str(k[2]), k[1])):
        config_label, num_tokens, input_dtype = key
        num_heads, head_size = meta[key]
        g = groups[key]
        dtype_str = short_dtype(input_dtype)
        t_cuda = g.get(f"cuda_{dtype_str}", float("nan"))
        t_triton = g.get(f"triton_{dtype_str}", float("nan"))
        speedup = t_triton / t_cuda if t_cuda > 0 else float("nan")
        print(
            f"{config_label:<18} {num_tokens:>6} {num_heads:>5} "
            f"{head_size:>5} "
            f"{dtype_str:<10} "
            f"{t_cuda:>9.1f}us {t_triton:>9.1f}us {speedup:>7.2f}x"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark merge_attn_states fused FP8 quantization"
    )
    parser.add_argument(
        "--fp8-only",
        action="store_true",
        help="Only run FP8 fused-vs-unfused benchmarks (skip non-FP8)",
    )
    parser.add_argument(
        "--nonfp8-only",
        action="store_true",
        help="Only run non-FP8 CUDA-vs-Triton benchmarks (skip FP8)",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=None,
        help=f"Override token counts (default: {NUM_TOKENS_LIST})",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"Warmup iterations (default: {DEFAULT_WARMUP})",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=DEFAULT_ITERS,
        help=f"Benchmark iterations (default: {DEFAULT_ITERS})",
    )
    args = parser.parse_args()

    warmup = args.warmup
    iters = args.iters
    num_tokens_list = args.num_tokens if args.num_tokens else NUM_TOKENS_LIST

    device_name = current_platform.get_device_name()
    print(f"Device: {device_name}")
    print(f"Warmup: {warmup}, Bench iters: {iters}")
    print(f"Token counts: {num_tokens_list}")
    print(f"Head configs: {[(c[0], c[1], c[2]) for c in HEAD_CONFIGS]}")
    print(f"Input dtypes: {[short_dtype(d) for d in INPUT_DTYPES]}")

    run_fp8 = not args.nonfp8_only
    run_nonfp8 = not args.fp8_only

    # --- Non-FP8 benchmarks ---
    if run_nonfp8:
        nonfp8_results: list[BenchResult] = []
        configs = list(itertools.product(HEAD_CONFIGS, num_tokens_list, INPUT_DTYPES))
        for count, ((label, nh, hs), nt, dtype) in enumerate(configs, 1):
            print(
                f"\r  Non-FP8 [{count}/{len(configs)}] "
                f"{label} tokens={nt} dtype={short_dtype(dtype)}    ",
                end="",
                flush=True,
            )
            nonfp8_results.extend(bench_nonfp8(label, nt, nh, hs, dtype, warmup, iters))
        print()
        print_nonfp8_table(nonfp8_results)

    # --- FP8 fused vs unfused benchmarks ---
    if run_fp8:
        fp8_results: list[BenchResult] = []
        configs = list(itertools.product(HEAD_CONFIGS, num_tokens_list, INPUT_DTYPES))
        for count, ((label, nh, hs), nt, dtype) in enumerate(configs, 1):
            print(
                f"\r  FP8 [{count}/{len(configs)}] "
                f"{label} tokens={nt} dtype={short_dtype(dtype)}    ",
                end="",
                flush=True,
            )
            fp8_results.extend(
                bench_fp8_fused_vs_unfused(label, nt, nh, hs, dtype, warmup, iters)
            )
        print()
        print_fp8_table(fp8_results)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
