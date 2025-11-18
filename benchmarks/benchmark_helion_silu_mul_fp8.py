#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark script comparing Helion silu_mul_fp8 kernel vs vLLM's CUDA kernel.

Usage:
    python benchmarks/benchmark_helion_silu_mul_fp8.py
    python benchmarks/benchmark_helion_silu_mul_fp8.py \
        --num-tokens 1024 --hidden-size 4096
    python benchmarks/benchmark_helion_silu_mul_fp8.py --profile
    python benchmarks/benchmark_helion_silu_mul_fp8.py --preset llm-serving
    python benchmarks/benchmark_helion_silu_mul_fp8.py --preset all --save-report
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import torch

# Import both kernels
from vllm.compilation.activation_quant_fusion import silu_mul_fp8 as helion_silu_mul_fp8
from vllm.triton_utils import triton

# Common LLM model configurations (intermediate size for MLP)
LLM_CONFIGS = {
    "llama-7b": 11008 // 2,  # 5504 (hidden_size for silu_and_mul is intermediate/2)
    "llama-13b": 13824 // 2,  # 6912
    "llama-70b": 28672 // 2,  # 14336
    "mistral-7b": 14336 // 2,  # 7168
    "mixtral-8x7b": 14336 // 2,  # 7168 (per expert)
}

# Common batch sizes for LLM serving
BATCH_SIZES = {
    "online": [1, 4, 8, 16, 32],  # Online serving (low latency)
    "micro": [64, 128, 256],  # Micro-batching
    "offline": [512, 1024, 2048, 4096],  # Offline batching (high throughput)
}

PRESETS = {
    "quick": {
        "num_tokens": [1, 32, 256, 1024],
        "hidden_sizes": [2048, 4096, 8192],
        "dtypes": ["bf16"],
    },
    "llm-serving": {
        "num_tokens": BATCH_SIZES["online"] + BATCH_SIZES["micro"],
        "hidden_sizes": list(LLM_CONFIGS.values()),
        "dtypes": ["bf16"],
    },
    "offline-batch": {
        "num_tokens": BATCH_SIZES["offline"],
        "hidden_sizes": list(LLM_CONFIGS.values()),
        "dtypes": ["bf16"],
    },
    "all": {
        "num_tokens": BATCH_SIZES["online"]
        + BATCH_SIZES["micro"]
        + BATCH_SIZES["offline"],
        "hidden_sizes": list(LLM_CONFIGS.values()) + [2048, 4096, 8192],
        "dtypes": ["bf16", "fp16"],
    },
}


def benchmark_cuda_kernel(
    input: torch.Tensor,
    scale: torch.Tensor,
    num_iterations: int = 1000,
    warmup: int = 10,
    use_cudagraph: bool = False,
) -> tuple[float, float]:
    """Benchmark CUDA kernel (out-variant) and return (avg_time_ms, throughput_gbps)."""
    num_tokens, hidden_size_x2 = input.shape
    hidden_size = hidden_size_x2 // 2

    if use_cudagraph:
        # CUDAGraph mode using Triton's benchmark utility
        def cuda_kernel_fn():
            out = torch.empty(
                num_tokens, hidden_size, dtype=torch.float8_e4m3fn, device="cuda"
            )
            torch.ops._C.silu_and_mul_quant(out, input, scale)
            return out

        # do_bench_cudagraph returns time in milliseconds
        avg_time_ms = triton.testing.do_bench_cudagraph(
            cuda_kernel_fn, rep=num_iterations
        )
    else:
        # Standard mode
        # Warmup
        for _ in range(warmup):
            out = torch.empty(
                num_tokens, hidden_size, dtype=torch.float8_e4m3fn, device="cuda"
            )
            torch.ops._C.silu_and_mul_quant(out, input, scale)
        torch.cuda.synchronize()

        # Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_iterations):
            out = torch.empty(
                num_tokens, hidden_size, dtype=torch.float8_e4m3fn, device="cuda"
            )
            torch.ops._C.silu_and_mul_quant(out, input, scale)
        end_event.record()
        torch.cuda.synchronize()

        # Calculate metrics
        total_time_ms = start_event.elapsed_time(end_event)
        avg_time_ms = total_time_ms / num_iterations

    # Calculate throughput (GB/s)
    # Read: input (2*d elements), scale (1 element)
    # Write: out (d elements)
    out_numel = num_tokens * hidden_size
    bytes_read = input.numel() * input.element_size() + scale.element_size()
    bytes_write = out_numel * torch.finfo(torch.float8_e4m3fn).bits // 8
    bytes_total = bytes_read + bytes_write
    throughput_gbps = (bytes_total / 1e9) / (avg_time_ms / 1000)

    return avg_time_ms, throughput_gbps


def benchmark_helion_kernel(
    input: torch.Tensor,
    scale: torch.Tensor,
    num_iterations: int = 1000,
    warmup: int = 10,
    use_cudagraph: bool = False,
) -> tuple[float, float]:
    """Benchmark Helion kernel (functional variant)."""
    num_tokens, hidden_size_x2 = input.shape
    hidden_size = hidden_size_x2 // 2

    if use_cudagraph:
        # CUDAGraph mode using Triton's benchmark utility
        def helion_kernel_fn():
            return helion_silu_mul_fp8(input, scale)

        # do_bench_cudagraph returns time in milliseconds
        avg_time_ms = triton.testing.do_bench_cudagraph(
            helion_kernel_fn, rep=num_iterations
        )
    else:
        # Standard mode
        # Warmup
        for _ in range(warmup):
            _ = helion_silu_mul_fp8(input, scale)
        torch.cuda.synchronize()

        # Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_iterations):
            _ = helion_silu_mul_fp8(input, scale)
        end_event.record()
        torch.cuda.synchronize()

        # Calculate metrics
        total_time_ms = start_event.elapsed_time(end_event)
        avg_time_ms = total_time_ms / num_iterations

    # Calculate throughput (GB/s)
    # Read: input (2*d elements), scale (1 element)
    # Write: out (d elements)
    out_numel = num_tokens * hidden_size
    bytes_read = input.numel() * input.element_size() + scale.element_size()
    bytes_write = out_numel * torch.finfo(torch.float8_e4m3fn).bits // 8
    bytes_total = bytes_read + bytes_write
    throughput_gbps = (bytes_total / 1e9) / (avg_time_ms / 1000)

    return avg_time_ms, throughput_gbps


def verify_correctness(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> bool:
    """Verify that Helion kernel produces same results as CUDA kernel."""
    torch.manual_seed(42)

    # Prepare inputs
    input_tensor = torch.randn(num_tokens, hidden_size * 2, dtype=dtype, device="cuda")
    scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")

    # Run CUDA kernel
    cuda_out = torch.empty(
        num_tokens, hidden_size, dtype=torch.float8_e4m3fn, device="cuda"
    )
    torch.ops._C.silu_and_mul_quant(cuda_out, input_tensor, scale)

    # Run Helion kernel
    helion_out = helion_silu_mul_fp8(input_tensor, scale)

    # Compare
    try:
        torch.testing.assert_close(
            cuda_out.to(dtype=dtype),
            helion_out.to(dtype=dtype),
            atol=atol,
            rtol=rtol,
        )
        return True
    except AssertionError as e:
        print(f"Correctness check failed: {e}")
        return False


def run_benchmark(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    num_iterations: int = 1000,
    warmup: int = 10,
    verify: bool = True,
    use_cudagraph: bool = False,
) -> dict:
    """Run benchmark for given configuration."""
    if verify:
        print(
            f"Verifying correctness for {num_tokens}x{hidden_size} {dtype}... ",
            end="",
        )
        if verify_correctness(num_tokens, hidden_size, dtype):
            print("✓ PASSED")
        else:
            print("✗ FAILED")
            return None

    # Prepare inputs
    torch.manual_seed(42)
    input_tensor = torch.randn(num_tokens, hidden_size * 2, dtype=dtype, device="cuda")
    scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")

    # Benchmark CUDA kernel (includes allocation cost)
    cuda_time, cuda_throughput = benchmark_cuda_kernel(
        input_tensor,
        scale,
        num_iterations,
        warmup,
        use_cudagraph,
    )

    # Benchmark Helion kernel (includes allocation cost)
    helion_time, helion_throughput = benchmark_helion_kernel(
        input_tensor,
        scale,
        num_iterations,
        warmup,
        use_cudagraph,
    )

    peak_bandwidth_gbps = 3350  # H100 HBM3 bandwidth (approximate)

    cuda_efficiency = (cuda_throughput / peak_bandwidth_gbps) * 100
    helion_efficiency = (helion_throughput / peak_bandwidth_gbps) * 100

    # Determine model category
    model_name = get_model_name(hidden_size)

    return {
        "num_tokens": num_tokens,
        "hidden_size": hidden_size,
        "model": model_name,
        "dtype": str(dtype),
        "cuda_time_ms": cuda_time,
        "cuda_throughput_gbps": cuda_throughput,
        "cuda_efficiency_pct": cuda_efficiency,
        "helion_time_ms": helion_time,
        "helion_throughput_gbps": helion_throughput,
        "helion_efficiency_pct": helion_efficiency,
        "speedup": cuda_time / helion_time,
    }


def get_model_name(hidden_size: int) -> str:
    """Get model name from hidden size."""
    for name, hs in LLM_CONFIGS.items():
        if hs == hidden_size:
            return name
    return f"custom-{hidden_size}"


def save_results_csv(results: list[dict], filename: str):
    """Save results to CSV file."""
    if not results:
        return

    valid_results = [r for r in results if r is not None]
    if not valid_results:
        return

    filepath = Path(filename)
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=valid_results[0].keys())
        writer.writeheader()
        writer.writerows(valid_results)

    print(f"\n✓ Results saved to {filepath}")


def save_results_json(results: list[dict], filename: str):
    """Save results to JSON file."""
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        return

    filepath = Path(filename)
    with open(filepath, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "device": torch.cuda.get_device_name(),
                "cuda_version": torch.version.cuda,
                "pytorch_version": torch.__version__,
                "results": valid_results,
            },
            f,
            indent=2,
        )

    print(f"✓ Results saved to {filepath}")


def print_results(results: list[dict]):
    """Pretty print benchmark results."""
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        return

    print("\n" + "=" * 140)
    print(
        f"{'Tokens':<10} {'Hidden':<8} {'Model':<15} {'DType':<12} "
        f"{'CUDA ms':<10} {'Helion ms':<10} {'Speedup':<10} "
        f"{'CUDA GB/s':<10} {'Helion GB/s':<12} {'CUDA Eff%':<10} {'Helion Eff%':<12}"
    )
    print("=" * 140)

    for result in valid_results:
        print(
            f"{result['num_tokens']:<10} "
            f"{result['hidden_size']:<8} "
            f"{result['model']:<15} "
            f"{result['dtype']:<12} "
            f"{result['cuda_time_ms']:<10.4f} "
            f"{result['helion_time_ms']:<10.4f} "
            f"{result['speedup']:<10.2f}x "
            f"{result['cuda_throughput_gbps']:<10.2f} "
            f"{result['helion_throughput_gbps']:<12.2f} "
            f"{result['cuda_efficiency_pct']:<10.1f} "
            f"{result['helion_efficiency_pct']:<12.1f}"
        )
    print("=" * 140 + "\n")


def print_grouped_results(results: list[dict]):
    """Print results grouped by batch size category."""
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        return

    # Group by batch size category
    categories = {
        "Online Serving (1-32)": [],
        "Micro-Batching (64-256)": [],
        "Offline Batching (512+)": [],
    }

    for result in valid_results:
        tokens = result["num_tokens"]
        if tokens <= 32:
            categories["Online Serving (1-32)"].append(result)
        elif tokens <= 256:
            categories["Micro-Batching (64-256)"].append(result)
        else:
            categories["Offline Batching (512+)"].append(result)

    for category, cat_results in categories.items():
        if not cat_results:
            continue

        print(f"\n{'=' * 60}")
        print(f"{category}")
        print(f"{'=' * 60}")

        speedups = [r["speedup"] for r in cat_results]
        cuda_times = [r["cuda_time_ms"] for r in cat_results]
        helion_times = [r["helion_time_ms"] for r in cat_results]

        print(f"Configurations: {len(cat_results)}")
        print(f"Avg Speedup: {sum(speedups) / len(speedups):.2f}x")
        print(f"Avg CUDA Time: {sum(cuda_times) / len(cuda_times):.4f}ms")
        print(f"Avg Helion Time: {sum(helion_times) / len(helion_times):.4f}ms")
        best_speedup_tokens = [
            r["num_tokens"] for r in cat_results if r["speedup"] == max(speedups)
        ][0]
        print(f"Best Speedup: {max(speedups):.2f}x at {best_speedup_tokens} tokens")
        worst_speedup_tokens = [
            r["num_tokens"] for r in cat_results if r["speedup"] == min(speedups)
        ][0]
        print(f"Worst Speedup: {min(speedups):.2f}x at {worst_speedup_tokens} tokens")


def print_summary_statistics(results: list[dict]):
    """Print comprehensive summary statistics."""
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        return

    speedups = [r["speedup"] for r in valid_results]
    cuda_throughputs = [r["cuda_throughput_gbps"] for r in valid_results]
    helion_throughputs = [r["helion_throughput_gbps"] for r in valid_results]
    cuda_efficiencies = [r["cuda_efficiency_pct"] for r in valid_results]
    helion_efficiencies = [r["helion_efficiency_pct"] for r in valid_results]

    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"Total configurations tested: {len(valid_results)}")
    print()

    print("Speedup:")
    print(f"  Average: {sum(speedups) / len(speedups):.2f}x")
    print(f"  Median:  {sorted(speedups)[len(speedups) // 2]:.2f}x")
    print(f"  Min:     {min(speedups):.2f}x")
    print(f"  Max:     {max(speedups):.2f}x")
    print()

    print("Throughput (GB/s):")
    cuda_avg = sum(cuda_throughputs) / len(cuda_throughputs)
    cuda_peak = max(cuda_throughputs)
    print(f"  CUDA   - Avg: {cuda_avg:.2f}, Peak: {cuda_peak:.2f}")
    helion_avg = sum(helion_throughputs) / len(helion_throughputs)
    helion_peak = max(helion_throughputs)
    print(f"  Helion - Avg: {helion_avg:.2f}, Peak: {helion_peak:.2f}")
    print()

    print("Memory Efficiency (% of H100 peak 3.35 TB/s):")
    print(f"  CUDA   - Avg: {sum(cuda_efficiencies) / len(cuda_efficiencies):.1f}%")
    print(f"  Helion - Avg: {sum(helion_efficiencies) / len(helion_efficiencies):.1f}%")
    print("=" * 60 + "\n")


def profile_kernel(
    num_tokens: int, hidden_size: int, dtype: torch.dtype, num_iterations: int = 100
):
    """Profile kernels with CUDA profiler."""
    print(
        f"\nProfiling with {num_tokens} tokens, hidden_size={hidden_size}, "
        f"dtype={dtype}"
    )
    print(
        "Run with: nsys profile -o profile python "
        "benchmarks/benchmark_helion_silu_mul_fp8.py --profile"
    )

    # Prepare inputs
    input_tensor = torch.randn(num_tokens, hidden_size * 2, dtype=dtype, device="cuda")
    scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")

    # Warmup
    for _ in range(10):
        cuda_out = torch.empty(
            num_tokens, hidden_size, dtype=torch.float8_e4m3fn, device="cuda"
        )
        torch.ops._C.silu_and_mul_quant(cuda_out, input_tensor, scale)
        helion_silu_mul_fp8(input_tensor, scale)
    torch.cuda.synchronize()

    # Profile CUDA kernel
    torch.cuda.nvtx.range_push("CUDA kernel")
    for _ in range(num_iterations):
        cuda_out = torch.empty(
            num_tokens, hidden_size, dtype=torch.float8_e4m3fn, device="cuda"
        )
        torch.ops._C.silu_and_mul_quant(cuda_out, input_tensor, scale)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    # Profile Helion kernel
    torch.cuda.nvtx.range_push("Helion kernel")
    for _ in range(num_iterations):
        helion_silu_mul_fp8(input_tensor, scale)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    print("Profiling complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Helion silu_mul_fp8 vs CUDA kernel"
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=None,
        help="Number of tokens to test (default: sweep)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        nargs="+",
        default=None,
        help="Hidden size to test (default: sweep)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp16", "bf16", "both"],
        default="both",
        help="Data type to test (default: both)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=list(PRESETS.keys()),
        default=None,
        help=f"Use preset configuration: {', '.join(PRESETS.keys())}",
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
        "--no-verify",
        action="store_true",
        help="Skip correctness verification",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run profiling mode (use with nsys)",
    )
    parser.add_argument(
        "--cudagraph",
        action="store_true",
        help="Use CUDAGraph for benchmarking (via triton.testing.do_bench_cudagraph)",
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save results to CSV and JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save reports (default: benchmark_results)",
    )
    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available")
        return

    device_name = torch.cuda.get_device_name()
    print(f"Running on: {device_name}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    if args.cudagraph:
        print("CUDAGraph mode: ENABLED (using triton.testing.do_bench_cudagraph)")
    print()

    # Profile mode
    if args.profile:
        profile_kernel(
            num_tokens=args.num_tokens[0] if args.num_tokens else 1024,
            hidden_size=args.hidden_size[0] if args.hidden_size else 4096,
            dtype=torch.bfloat16,
            num_iterations=100,
        )
        return

    # Determine test configurations
    if args.preset:
        print(f"Using preset: {args.preset}")
        preset = PRESETS[args.preset]
        num_tokens_list = preset["num_tokens"]
        hidden_size_list = preset["hidden_sizes"]
        dtypes_list = preset["dtypes"]
    else:
        # Use explicit args or defaults
        if args.num_tokens is None:
            num_tokens_list = [32, 64, 128, 256, 512, 1024, 2048, 4096]
        else:
            num_tokens_list = args.num_tokens

        if args.hidden_size is None:
            hidden_size_list = [512, 1024, 2048, 4096, 8192]
        else:
            hidden_size_list = args.hidden_size

        dtypes_list = [args.dtype] if args.dtype != "both" else ["fp16", "bf16"]

    # Convert dtype strings to torch dtypes
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16}
    dtypes = [dtype_map[d] for d in dtypes_list]

    # Print configuration summary
    print("Test Configuration:")
    print(f"  Batch sizes: {num_tokens_list}")
    print(f"  Hidden sizes: {hidden_size_list}")
    print(f"  Data types: {dtypes_list}")
    print(f"  Iterations per config: {args.num_iterations}")
    print()

    # Run benchmarks
    results = []
    total_configs = len(num_tokens_list) * len(hidden_size_list) * len(dtypes)
    print(f"Running {total_configs} configurations...\n")

    for dtype in dtypes:
        for hidden_size in hidden_size_list:
            for num_tokens in num_tokens_list:
                result = run_benchmark(
                    num_tokens=num_tokens,
                    hidden_size=hidden_size,
                    dtype=dtype,
                    num_iterations=args.num_iterations,
                    warmup=args.warmup,
                    verify=not args.no_verify,
                    use_cudagraph=args.cudagraph,
                )
                results.append(result)

    # Print results
    print_results(results)
    print_grouped_results(results)
    print_summary_statistics(results)

    # Save reports if requested
    if args.save_report:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        preset_suffix = f"_{args.preset}" if args.preset else ""

        csv_file = output_dir / f"helion_benchmark{preset_suffix}_{timestamp}.csv"
        json_file = output_dir / f"helion_benchmark{preset_suffix}_{timestamp}.json"

        save_results_csv(results, str(csv_file))
        save_results_json(results, str(json_file))


if __name__ == "__main__":
    main()
