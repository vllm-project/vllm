# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the fused tanh-GELU + gated-mul + static FP8 quantization kernel.

Compares, per shape:

    unfused:  torch.ops._C.gelu_tanh_and_mul  ->  scaled_fp8_quant
    fused:    torch.ops._C.gelu_tanh_and_mul_quant

The fused path skips the bf16/fp16 intermediate activation tensor, so the
post-activation bytes are written once as FP8 instead of twice in the model
dtype.

Usage:
    python benchmarks/kernels/benchmark_gelu_tanh_fp8_quant.py --check
"""

import argparse
import math

import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.triton_utils import triton

FP8_DTYPE = current_platform.fp8_dtype()

# (num_tokens, hidden_size). 2048x14336 matches the down_proj input width of
# a Gemma-4-31B style MLP.
DEFAULT_SHAPES = [(1, 4096), (64, 4096), (512, 4096), (2048, 4096), (2048, 14336)]


def bench_shape(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    warmup_ms: float,
    rep_ms: float,
    check: bool,
) -> tuple[float, float]:
    """Return (unfused_ms, fused_ms) as the median of `do_bench`."""
    x = torch.randn(num_tokens, 2 * hidden_size, dtype=dtype, device="cuda")
    scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")
    tmp = torch.empty(num_tokens, hidden_size, dtype=dtype, device="cuda")
    out = torch.empty(num_tokens, hidden_size, dtype=FP8_DTYPE, device="cuda")

    def unfused() -> None:
        torch.ops._C.gelu_tanh_and_mul(tmp, x)
        ops.scaled_fp8_quant(tmp, scale, output=out)

    def fused() -> None:
        torch.ops._C.gelu_tanh_and_mul_quant(out, x, scale)

    if check:
        unfused()
        ref = out.clone()
        out.zero_()
        fused()
        if not torch.equal(out.view(torch.uint8), ref.view(torch.uint8)):
            raise AssertionError(
                f"fused kernel is not bit-identical at {num_tokens}x{hidden_size}"
            )

    t_unfused = triton.testing.do_bench(unfused, warmup=warmup_ms, rep=rep_ms)
    t_fused = triton.testing.do_bench(fused, warmup=warmup_ms, rep=rep_ms)
    return t_unfused, t_fused


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="assert the fused kernel is bit-identical to the unfused path",
    )
    parser.add_argument("--warmup-ms", type=float, default=25.0)
    parser.add_argument("--rep-ms", type=float, default=100.0)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument(
        "--shapes",
        nargs="*",
        default=None,
        help="override shapes as MxD, e.g. --shapes 64x4096 2048x14336",
    )
    args = parser.parse_args()

    shapes = DEFAULT_SHAPES
    if args.shapes:
        shapes = [tuple(int(v) for v in s.split("x")) for s in args.shapes]

    dtype = getattr(torch, args.dtype)
    print(f"dtype: {args.dtype} -> {FP8_DTYPE}")
    print(f"{'shape':>14} {'unfused(us)':>13} {'fused(us)':>12} {'speedup':>9}")

    speedups = []
    for num_tokens, hidden_size in shapes:
        t_unfused, t_fused = bench_shape(
            num_tokens,
            hidden_size,
            dtype,
            args.warmup_ms,
            args.rep_ms,
            args.check,
        )
        speedups.append(t_unfused / t_fused)
        print(
            f"{f'{num_tokens}x{hidden_size}':>14} "
            f"{t_unfused * 1000:13.1f} {t_fused * 1000:12.1f} "
            f"{t_unfused / t_fused:8.2f}x"
        )

    geomean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
    print(f"{'geomean':>14} {'':>13} {'':>12} {geomean:8.2f}x")


if __name__ == "__main__":
    main()
