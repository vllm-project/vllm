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

# Import helion package to auto-register all kernel benchmarks and CustomOps
import vllm.compilation.helion  # noqa: F401
import vllm.compilation.helion.allreduce_add_rmsnorm  # noqa: F401
import vllm.compilation.helion.rms_norm_fp8  # noqa: F401
import vllm.compilation.helion.silu_mul_fp8  # noqa: F401
from vllm.compilation.helion.benchmark import (
    print_results,
    print_summary_statistics,
    save_results_csv,
    save_results_json,
)


# Mock classes for kernel benchmarking (must be at module level for multiprocessing)
class MockHFConfig:
    """Mock HuggingFace config for benchmarking purposes."""

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size


class MockModelConfig:
    """Mock vLLM ModelConfig that mimics the real one for benchmarking."""

    def __init__(self, hidden_size):
        self.hf_text_config = MockHFConfig(hidden_size)

    def get_hidden_size(self):
        return getattr(self.hf_text_config, "hidden_size", 0)


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
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=4096,
        help="Hidden size for model configuration (default: 4096)",
    )
    args = parser.parse_args()

    # List all benchmarks if requested
    if args.list_benchmarks:
        try:
            from vllm.compilation.helion.custom_op import get_registered_custom_ops

            custom_ops = get_registered_custom_ops()
            available_benchmarks = []
            for op_name, op_class in custom_ops.items():
                if op_class.get_benchmark() is not None:
                    available_benchmarks.append(op_name)

            print("Available Helion CustomOp benchmarks:\n")
            for bench_name in available_benchmarks:
                print(f"  {bench_name}")
        except ImportError:
            print("ERROR: Helion not available")
        return

    # Get the CustomOp by name and its associated benchmark class
    try:
        from vllm.compilation.helion.custom_op import get_registered_custom_ops

        custom_ops = get_registered_custom_ops()
        if args.benchmark not in custom_ops:
            raise ValueError(f"Unknown CustomOp '{args.benchmark}'")

        custom_op_class = custom_ops[args.benchmark]
        benchmark_cls = custom_op_class.get_benchmark()
        if benchmark_cls is None:
            raise ValueError(f"No benchmark registered for CustomOp '{args.benchmark}'")

    except (ImportError, ValueError) as e:
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
    print(f"Hidden size: {args.hidden_size}")
    print(f"CUDA graphs: {'DISABLED' if args.no_cudagraph else 'ENABLED'}")
    print(f"Correctness verification: {'DISABLED' if args.no_verify else 'ENABLED'}")
    print(f"Iterations: {args.num_iterations}")
    print(f"Warmup: {args.warmup}")
    print()

    # Set default device
    torch.set_default_device("cuda")

    # Configure the kernel with the specified hidden size
    try:
        from vllm.compilation.helion.config_manager import ConfigManager

        mock_model_config = MockModelConfig(args.hidden_size)

    except ImportError:
        print("ERROR: Helion not available")
        return
    except Exception as e:
        print(f"ERROR: Failed to configure kernel: {e}")
        return

    # Create benchmark instance with model config
    benchmark = benchmark_cls(model_config=mock_model_config)

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
