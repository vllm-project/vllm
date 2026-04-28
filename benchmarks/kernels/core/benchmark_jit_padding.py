# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Performance benchmark for JIT padding optimization.

Compares execution latency between:
- Python fallback (JIT disabled)
- JIT C++ extension (JIT enabled)

== Usage Examples ==

Benchmark mode (default):
  python3 benchmarks/kernels/core/benchmark_jit_padding.py

Custom parameters:
  python3 benchmarks/kernels/core/benchmark_jit_padding.py --token-length 100000 --batch-size 100

Profile mode (PyTorch profiler):
  python3 benchmarks/kernels/core/benchmark_jit_padding.py --profile
  python3 benchmarks/kernels/core/benchmark_jit_padding.py --profile --profile-output-dir ./profile_traces
"""

import os
import random
import statistics
import time

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function

from vllm.utils.argparse_utils import FlexibleArgumentParser


def generate_test_data(batch_size: int, token_length: int, seed: int = 42):
    """
    Generate test data: list of variable-length token sequences.

    Creates realistic scenario where each request has different token count,
    simulating padding requirement in actual inference.

    Note: Uses native Python int (not numpy scalar) for JIT compatibility.
    """
    random.seed(seed)
    data = []
    for i in range(batch_size):
        length = token_length + (i % 200 - 100)  # Vary length by ±100
        row = [random.randint(0, 50000) for _ in range(length)]
        data.append(row)
    return data


def reset_jit_state():
    """Reset JIT module state to force fresh load attempt."""
    import vllm.utils.torch_utils as tu
    tu._CUSTOM_PAD_OP_MODULE = None
    tu._JIT_LOAD_FAILED = False


def set_jit_enabled(enabled: bool):
    """Enable or disable JIT by modifying the envs module directly."""
    import vllm.envs as envs
    envs.VLLM_ENABLE_JIT_PADDING = enabled


def run_benchmark_iteration(test_data, iterations: int, warmup: int) -> dict:
    """
    Run benchmark iterations and return latency statistics.
    """
    import vllm.utils.torch_utils as tu

    # Warmup runs
    for _ in range(warmup):
        _ = tu.make_ndarray_with_pad(test_data, 0, np.int64)

    # Benchmark
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = tu.make_ndarray_with_pad(test_data, 0, np.int64)
        end = time.perf_counter()
        latencies.append((end - start) * 1_000_000)  # microseconds

    return {
        "mean": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "min": min(latencies),
        "max": max(latencies),
        "stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "p99": sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) >= 100 else max(latencies),
        "jit_loaded": tu._CUSTOM_PAD_OP_MODULE is not None,
    }


def run_benchmark(args):
    """Run benchmark comparing JIT vs Python fallback."""
    print("=" * 70)
    print("JIT Padding Performance Benchmark")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  - Token length: {args.token_length} (varying ±100 per request)")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Iterations: {args.iterations}")
    print(f"  - Warmup: {args.warmup}")
    print("=" * 70)
    print()

    # Generate test data
    print("Generating test data...")
    test_data = generate_test_data(args.batch_size, args.token_length)
    actual_lengths = [len(row) for row in test_data]

    print("Test data info:")
    print(f"  Min length: {min(actual_lengths)}")
    print(f"  Max length: {max(actual_lengths)}")
    print(f"  Total elements: {sum(actual_lengths):,}")
    print(f"  Output shape: ({args.batch_size}, {max(actual_lengths)})")
    print()

    results = {}

    # Test 1: Python fallback (JIT disabled)
    print("Benchmark 1: Python Fallback (JIT disabled)")
    print("-" * 40)

    set_jit_enabled(False)
    reset_jit_state()

    results["python"] = run_benchmark_iteration(test_data, args.iterations,
                                                   args.warmup)

    print(f"  JIT loaded: {results['python']['jit_loaded']}")
    print(f"  Mean latency: {results['python']['mean']:.2f} us "
          f"({results['python']['mean']/1000:.2f} ms)")
    print(f"  Median latency: {results['python']['median']:.2f} us "
          f"({results['python']['median']/1000:.2f} ms)")
    print(f"  Min latency: {results['python']['min']:.2f} us")
    print(f"  Max latency: {results['python']['max']:.2f} us "
          f"({results['python']['max']/1000:.2f} ms)")
    print(f"  P99 latency: {results['python']['p99']:.2f} us "
          f"({results['python']['p99']/1000:.2f} ms)")
    print(f"  Std dev: {results['python']['stdev']:.2f} us")
    print()

    # Test 2: JIT C++ extension (JIT enabled)
    print("Benchmark 2: JIT C++ Extension (JIT enabled)")
    print("-" * 40)

    set_jit_enabled(True)
    reset_jit_state()

    results["jit"] = run_benchmark_iteration(test_data, args.iterations,
                                               args.warmup)

    print(f"  JIT loaded: {results['jit']['jit_loaded']}")
    print(f"  Mean latency: {results['jit']['mean']:.2f} us "
          f"({results['jit']['mean']/1000:.2f} ms)")
    print(f"  Median latency: {results['jit']['median']:.2f} us "
          f"({results['jit']['median']/1000:.2f} ms)")
    print(f"  Min latency: {results['jit']['min']:.2f} us")
    print(f"  Max latency: {results['jit']['max']:.2f} us "
          f"({results['jit']['max']/1000:.2f} ms)")
    print(f"  P99 latency: {results['jit']['p99']:.2f} us "
          f"({results['jit']['p99']/1000:.2f} ms)")
    print(f"  Std dev: {results['jit']['stdev']:.2f} us")
    print()

    # Comparison
    if results["python"] and results["jit"]:
        print("=" * 70)
        print("Performance Comparison")
        print("=" * 70)

        speedup_mean = results["python"]["mean"] / results["jit"]["mean"]
        speedup_median = results["python"]["median"] / results["jit"]["median"]
        speedup_p99 = results["python"]["p99"] / results["jit"]["p99"]

        latency_reduction_mean = (1 - results["jit"]["mean"] /
                                  results["python"]["mean"]) * 100
        latency_reduction_p99 = (1 - results["jit"]["p99"] /
                                 results["python"]["p99"]) * 100

        print(f"  Mean latency speedup: {speedup_mean:.2f}x")
        print(f"  Median latency speedup: {speedup_median:.2f}x")
        print(f"  P99 latency speedup: {speedup_p99:.2f}x")
        print()
        print(f"  Mean latency reduction: {latency_reduction_mean:.1f}%")
        print(f"  P99 latency reduction: {latency_reduction_p99:.1f}%")
        print()

        # Table format
        print(f"  | Metric      | Python (us)  | JIT (us)    | Speedup |")
        print(f"  |-------------|--------------|------------|---------|")
        print(f"  | Mean        | {results['python']['mean']:>10.2f}   "
              f"| {results['jit']['mean']:>10.2f}   | {speedup_mean:>6.2f}x |")
        print(f"  | Median      | {results['python']['median']:>10.2f}   "
              f"| {results['jit']['median']:>10.2f}   | {speedup_median:>6.2f}x |")
        print(f"  | P99         | {results['python']['p99']:>10.2f}   "
              f"| {results['jit']['p99']:>10.2f}   | {speedup_p99:>6.2f}x |")
        print(f"  | Max         | {results['python']['max']:>10.2f}   "
              f"| {results['jit']['max']:>10.2f}   | "
              f"{results['python']['max']/results['jit']['max']:>6.2f}x |")
        print("=" * 70)

    return results


def run_profile(args):
    """Run PyTorch profiler instead of benchmark."""
    print("=" * 70)
    print("JIT Padding Profiling")
    print("=" * 70)
    print(f"Token length: {args.token_length}")
    print(f"Batch size: {args.batch_size}")
    print(f"Warmup: {args.warmup}")
    print(f"Output dir: {args.profile_output_dir}")
    print("=" * 70)

    test_data = generate_test_data(args.batch_size, args.token_length)

    # Enable JIT
    set_jit_enabled(True)
    reset_jit_state()

    # Warmup
    import vllm.utils.torch_utils as tu
    for _ in range(args.warmup):
        _ = tu.make_ndarray_with_pad(test_data, 0, np.int64)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True,
                 profile_memory=True,
                 with_stack=True) as prof:
        for _ in range(args.iterations):
            with record_function("make_ndarray_with_pad"):
                _ = tu.make_ndarray_with_pad(test_data, 0, np.int64)

    # Save trace
    output_path = os.path.join(args.profile_output_dir,
                               f"jit_padding_trace_{args.token_length}.json")
    prof.export_chrome_trace(output_path)
    print(f"\nTrace saved to: {output_path}")

    # Print summary
    print("\nTop 10 CPU operations:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


def main():
    parser = FlexibleArgumentParser(
        description="Benchmark JIT padding optimization")

    parser.add_argument(
        "--token-length",
        type=int,
        default=50000,
        help="Token length per request (default: 50000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of requests in batch (default: 50)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations (default: 100)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run PyTorch profiler instead of benchmark",
    )
    parser.add_argument(
        "--profile-output-dir",
        type=str,
        default="./profile_traces",
        help="Output directory for traces (default: ./profile_traces)",
    )

    args = parser.parse_args()

    os.makedirs(args.profile_output_dir, exist_ok=True)

    if args.profile:
        run_profile(args)
    else:
        run_benchmark(args)


if __name__ == "__main__":
    main()
