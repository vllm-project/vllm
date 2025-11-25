#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CLI wrapper for benchmarking Helion kernels.

Usage:
    # Run quick benchmark for a specific kernel
    python benchmarks/benchmark_helion.py --benchmark silu_mul_fp8 --mode quick

    # Run full benchmark suite for RMS norm
    python benchmarks/benchmark_helion.py \\
        --benchmark rms_norm_fp8 --mode full

    # List available benchmarks
    python benchmarks/benchmark_helion.py --list-benchmarks

    # Custom output directory
    python benchmarks/benchmark_helion.py \\
        --benchmark silu_mul_fp8 --mode full --output-dir ./results
"""

import argparse
from datetime import datetime
from pathlib import Path

import torch

# Import helion package to auto-register all kernel benchmarks
import vllm.compilation.helion  # noqa: F401
from vllm.compilation.helion.benchmark import (
    KernelBenchmark,
    print_results,
    print_summary_statistics,
    save_results_csv,
    save_results_json,
)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Helion kernels vs CUDA reference"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="silu_mul_fp8",
        help="Registered name of the Helion kernel benchmark to run",
    )
    parser.add_argument(
        "--list-benchmarks",
        action="store_true",
        help="List all available Helion kernel benchmarks and exit",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["quick", "full"],
        default="quick",
        help=(
            "Benchmark mode: 'quick' for smoke testing, "
            "'full' for comprehensive benchmarking (default: quick)"
        ),
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=1000,
        help="Number of iterations for benchmarking (default: 1000)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--no-cudagraph",
        action="store_true",
        help="Disable CUDA graphs (enabled by default)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip correctness verification",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for correctness verification (default: 1e-5)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for correctness verification (default: 1e-3)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory to save reports "
            "(default: /tmp/helion_benchmark_{benchmark}_{timestamp})"
        ),
    )
    args = parser.parse_args()

    # List all benchmarks if requested
    if args.list_benchmarks:
        benchmarks = KernelBenchmark.list_benchmarks()
        print("Available Helion kernel benchmarks:\n")
        for bench_name in benchmarks:
            print(f"  {bench_name}")
        return

    # Get the benchmark class
    try:
        benchmark_cls = KernelBenchmark.get_benchmark_class(args.benchmark)
    except ValueError as e:
        print(f"ERROR: {e}")
        return

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available")
        return

    # Print device information
    device_name = torch.cuda.get_device_name()
    print(f"Running on: {device_name}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Mode: {args.mode}")
    print(f"CUDA graphs: {'DISABLED' if args.no_cudagraph else 'ENABLED'}")
    print(f"Correctness verification: {'DISABLED' if args.no_verify else 'ENABLED'}")
    print(f"Iterations: {args.num_iterations}")
    print(f"Warmup: {args.warmup}")
    print()

    # Set default device
    torch.set_default_device("cuda")

    # Create benchmark instance
    benchmark = benchmark_cls()

    # Run benchmark using the standardized API
    results = benchmark.run(
        mode=args.mode,
        num_iterations=args.num_iterations,
        warmup=args.warmup,
        use_cudagraph=not args.no_cudagraph,
        verify=not args.no_verify,
        atol=args.atol,
        rtol=args.rtol,
    )

    # Print results
    print_results(results)
    print_summary_statistics(results)

    # Save reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Use default /tmp directory if not specified
    output_dir = Path(
        args.output_dir or f"/tmp/helion_benchmark_{args.benchmark}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_file = output_dir / f"{args.benchmark}_{args.mode}_{timestamp}.csv"
    json_file = output_dir / f"{args.benchmark}_{args.mode}_{timestamp}.json"

    save_results_csv(results, str(csv_file))
    save_results_json(results, str(json_file))


if __name__ == "__main__":
    main()
