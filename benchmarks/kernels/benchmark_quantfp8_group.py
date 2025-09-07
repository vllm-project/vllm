#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark for QuantFP8 Group Quantization implementation."""

import argparse

import torch

from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform


def _time_cuda(
    fn,
    warmup_iters: int,
    bench_iters: int,
) -> float:
    # warmup
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(bench_iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / bench_iters  # ms/iter


def run_benchmark(
    shape: tuple[int, int],
    group_size: int,
    column_major: bool,
    warmup_iters: int,
    bench_iters: int,
) -> None:
    """Benchmark QuantFP8 with group quantization using different backends."""
    num_tokens, hidden_dim = shape

    device = torch.device("cuda")
    torch.manual_seed(42)
    x = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.bfloat16) * 8

    group_shape = GroupShape(1, group_size)
    quant_op = QuantFP8(
        static=False, group_shape=group_shape, column_major_scales=column_major
    )

    def cuda_impl():
        return quant_op.forward_cuda(x.clone())

    def native_impl():
        return quant_op.forward_native(x.clone())

    cuda_ms = _time_cuda(cuda_impl, warmup_iters, bench_iters)
    native_ms = _time_cuda(native_impl, warmup_iters, bench_iters)

    speedup = cuda_ms / native_ms if native_ms else 0

    cfg_desc = f"shape={shape}  gs={group_size:<3}  col_major={column_major}"
    print(f"{cfg_desc:45} | {cuda_ms:7.3f} | {native_ms:7.3f} | {speedup:6.2f}x")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark QuantFP8 group quantization implementation"
    )
    parser.add_argument(
        "--warmup-iters", type=int, default=10, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--bench-iters", type=int, default=100, help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--shapes",
        type=str,
        default="32,128;64,256;16,512;128,1024;256,2048",
        help="Shapes to benchmark as 'tokens,hidden;...' (default: multiple shapes)",
    )
    parser.add_argument(
        "--group-sizes",
        type=str,
        default="64,128",
        help="Group sizes to benchmark (comma-separated)",
    )
    parser.add_argument(
        "--no-column-major",
        action="store_true",
        help="Skip column-major scale benchmarks",
    )
    return parser.parse_args()


def main():
    if not current_platform.is_cuda():
        raise RuntimeError("CUDA device is required to run this benchmark.")

    args = parse_args()

    shapes = []
    for shape_str in args.shapes.split(";"):
        tokens, hidden = map(int, shape_str.split(","))
        shapes.append((tokens, hidden))

    group_sizes = list(map(int, args.group_sizes.split(",")))

    print("\n" + "=" * 80)
    print("QuantFP8 Group Quantization Benchmark (CUDA kernel vs PyTorch native)")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Warmup iterations: {args.warmup_iters}")
    print(f"Benchmark iterations: {args.bench_iters}")
    print("=" * 80)

    print(f"{'Configuration':45} | {'CUDA':^9} | {'Native':^9} | {'Speedup':^8}")
    print("-" * 80)

    for shape in shapes:
        for gs in group_sizes:
            run_benchmark(
                shape,
                gs,
                column_major=False,
                warmup_iters=args.warmup_iters,
                bench_iters=args.bench_iters,
            )

            if not args.no_column_major:
                run_benchmark(
                    shape,
                    gs,
                    column_major=True,
                    warmup_iters=args.warmup_iters,
                    bench_iters=args.bench_iters,
                )

    print("=" * 80)


if __name__ == "__main__":
    main()
