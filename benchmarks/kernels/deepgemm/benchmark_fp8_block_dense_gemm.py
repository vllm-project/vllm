# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
import time

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
    w8a8_triton_block_scaled_mm,
)
from vllm.triton_utils import triton
from vllm.utils.deep_gemm import (
    calc_diff,
    fp8_gemm_nt,
    per_block_cast_to_fp8,
)


def benchmark_shape(
    m: int,
    n: int,
    k: int,
    warmup: int = 100,
    repeat: int = 10000,
    verbose: bool = False,
) -> dict:
    """Benchmark all implementations for a specific (m, n, k) shape."""
    if verbose:
        print(f"\n=== Benchmarking shape: m={m}, n={n}, k={k} ===")

    # Create test tensors
    A = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    B = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)

    # Reference result in BF16
    torch.cuda.synchronize()
    C_ref = A @ B.t()

    # Pre-quantize B for all implementations
    # (weights can be pre-quantized offline)
    B_deepgemm, B_scale_deepgemm = per_block_cast_to_fp8(B, [128, 128], use_ue8m0=True)
    B_vllm, B_scale_vllm = per_block_cast_to_fp8(B, [128, 128], use_ue8m0=True)

    # Block size configuration
    block_size = [128, 128]

    # Pre-quantize A for all implementations
    A_deepgemm, A_scale_deepgemm = per_token_group_quant_fp8(
        A, block_size[1], column_major_scales=True, tma_aligned_scales=True
    )
    C_deepgemm = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    A_vllm, A_scale_vllm = per_token_group_quant_fp8(A, block_size[1])
    A_vllm_cutlass, A_scale_vllm_cutlass = per_token_group_quant_fp8(
        A, block_size[1], column_major_scales=True
    )

    # === DeepGEMM Implementation ===
    def deepgemm_gemm():
        fp8_gemm_nt(
            (A_deepgemm, A_scale_deepgemm), (B_deepgemm, B_scale_deepgemm), C_deepgemm
        )
        return C_deepgemm

    # === vLLM Triton Implementation ===
    def vllm_triton_gemm():
        return w8a8_triton_block_scaled_mm(
            A_vllm,
            B_vllm,
            A_scale_vllm,
            B_scale_vllm,
            block_size,
            output_dtype=torch.bfloat16,
        )

    # === vLLM CUTLASS Implementation ===
    def vllm_cutlass_gemm():
        return ops.cutlass_scaled_mm(
            A_vllm_cutlass,
            B_vllm.T,
            scale_a=A_scale_vllm_cutlass,
            scale_b=B_scale_vllm.T,
            out_dtype=torch.bfloat16,
        )

    # Run correctness check first
    if verbose:
        print("Running correctness check...")
    C_deepgemm = deepgemm_gemm()
    C_vllm_triton = vllm_triton_gemm()
    C_vllm_cutlass = vllm_cutlass_gemm()

    deepgemm_diff = calc_diff(C_deepgemm, C_ref)
    vllm_triton_diff = calc_diff(C_vllm_triton, C_ref)
    vllm_cutlass_diff = calc_diff(C_vllm_cutlass, C_ref)

    if verbose:
        print(f"DeepGEMM vs Reference difference: {deepgemm_diff:.6f}")
        print(f"vLLM Triton vs Reference difference: {vllm_triton_diff:.6f}")
        print(f"vLLM CUTLASS vs Reference difference: {vllm_cutlass_diff:.6f}")
        print(
            "vLLM Triton vs DeepGEMM difference: "
            f"{calc_diff(C_vllm_triton, C_deepgemm):.6f}"
        )
        print(
            "vLLM CUTLASS vs DeepGEMM difference: "
            f"{calc_diff(C_vllm_cutlass, C_deepgemm):.6f}"
        )

    # Benchmark implementations
    implementations = {
        "DeepGEMM": deepgemm_gemm,
        "vLLM Triton": vllm_triton_gemm,
        "vLLM CUTLASS": vllm_cutlass_gemm,
    }

    benchmark_results = {"shape": {"m": m, "n": n, "k": k}, "implementations": {}}

    for name, func in implementations.items():
        # Warmup
        for _ in range(warmup):
            func()
            torch.cuda.synchronize()

        # Timing loop
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(repeat):
            func()
        torch.cuda.synchronize()
        end = time.time()

        # Calculate timing and TFLOPS
        avg_time_ms = (end - start) / repeat * 1000
        avg_time_us = avg_time_ms * 1000
        tflops = 2 * m * n * k / (avg_time_ms * 1e-3) / 1e12
        gb_s = (m * k + k * n + m * n * 2) / 1e9 / (avg_time_ms * 1e-3)

        benchmark_results["implementations"][name] = {
            "time_ms": avg_time_ms,
            "time_us": avg_time_us,
            "tflops": tflops,
            "gb_s": gb_s,
            "diff": {
                "DeepGEMM": 0.0
                if name == "DeepGEMM"
                else calc_diff(func(), C_deepgemm),
                "Reference": deepgemm_diff
                if name == "DeepGEMM"
                else (vllm_triton_diff if name == "vLLM Triton" else vllm_cutlass_diff),
            },
        }

        if verbose:
            print(f"{name}: {avg_time_ms:.3f} ms, {tflops:.2f} TFLOPS, {gb_s:.2f} GB/s")

    # Calculate speedups
    baseline = benchmark_results["implementations"]["DeepGEMM"]["time_ms"]
    for name, data in benchmark_results["implementations"].items():
        if name != "DeepGEMM":
            speedup = baseline / data["time_ms"]
            benchmark_results["implementations"][name]["speedup_vs_deepgemm"] = speedup
            if verbose:
                print(
                    f"DeepGEMM is {1 / speedup:.2f}x "
                    f"{'faster' if 1 / speedup > 1 else 'slower'} than {name}"
                )

    vllm_triton_time = benchmark_results["implementations"]["vLLM Triton"]["time_ms"]
    vllm_cutlass_time = benchmark_results["implementations"]["vLLM CUTLASS"]["time_ms"]
    cutlass_vs_triton = vllm_triton_time / vllm_cutlass_time
    benchmark_results["implementations"]["vLLM CUTLASS"]["speedup_vs_triton"] = (
        cutlass_vs_triton
    )
    if verbose:
        print(
            f"vLLM CUTLASS is {cutlass_vs_triton:.2f}x "
            f"{'faster' if cutlass_vs_triton > 1 else 'slower'} than vLLM Triton"
        )

    return benchmark_results


def format_table_row(values, widths):
    """Format a row with specified column widths."""
    return "| " + " | ".join(f"{val:{w}}" for val, w in zip(values, widths)) + " |"


def print_table(headers, rows, title=None):
    """Print a table with headers and rows."""
    if title:
        print(f"\n{title}")

    # Calculate column widths based on headers and data
    widths = [
        max(len(str(h)), max(len(str(row[i])) for row in rows))
        for i, h in enumerate(headers)
    ]

    # Create separator line
    separator = "+-" + "-+-".join("-" * w for w in widths) + "-+"

    # Print table
    print(separator)
    print(format_table_row(headers, widths))
    print(separator)
    for row in rows:
        print(format_table_row(row, widths))
    print(separator)


def format_speedup(value):
    """Format speedup value with indicator if it's faster or slower."""
    return f"{value:.2f}x {'faster' if value > 1.0 else 'slower'}"


def run_benchmarks(verbose: bool = False):
    """Run benchmarks for a set of common shapes."""
    print("===== STARTING FP8 GEMM BENCHMARK =====")

    # Make sure we're using the GPU
    if not torch.cuda.is_available():
        print("CUDA not available! Tests require GPU.")
        return

    # Print system information
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Triton version: {triton.__version__}")
    print(f"Using device: {torch.cuda.get_device_name()}")

    # Enable TF32 for better performance
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Set seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Define benchmark shapes (m, n, k)
    shapes = [
        (8, 4096, 7168),
        (8, 7168, 18432),
        (8, 18432, 7168),
        (64, 4096, 7168),
        (64, 7168, 18432),
        (64, 18432, 7168),
        (64, 24576, 1536),
        (64, 32768, 512),
        (64, 7168, 16384),
        (128, 4096, 7168),
        (128, 7168, 18432),
        (128, 18432, 7168),
        (1024, 4096, 7168),
        (1024, 18432, 7168),
        (2048, 4096, 7168),
        (4096, 4096, 7168),
    ]
    shapes = [
        # (64, 2112, 7168),
        (64, 24576, 1536),
        (64, 32768, 512),
        (64, 7168, 16384),
        (64, 4096, 7168),
        (64, 7168, 2048),
        # (128, 2112, 7168),
        (128, 24576, 1536),
        (128, 32768, 512),
        (128, 7168, 16384),
        (128, 4096, 7168),
        (128, 7168, 2048),
        # (4096, 2112, 7168),
        (4096, 24576, 1536),
        (4096, 32768, 512),
        (4096, 7168, 16384),
        (4096, 4096, 7168),
        (4096, 7168, 2048),
    ]

    all_results = []
    for m, n, k in shapes:
        result = benchmark_shape(m, n, k, verbose=verbose)
        all_results.append(result)

    # Print results in a nicely formatted table
    print("\n===== PERFORMANCE COMPARISON =====")

    # Print DeepGEMM table
    deepgemm_headers = ["m", "n", "k", "Time (μs)", "TFLOPS", "GB/s"]
    deepgemm_rows = []
    for result in all_results:
        shape = result["shape"]
        impl_data = result["implementations"]["DeepGEMM"]
        deepgemm_rows.append(
            [
                shape["m"],
                shape["n"],
                shape["k"],
                f"{impl_data['time_us']:.1f}",
                f"{impl_data['tflops']:.1f}",
                f"{impl_data['gb_s']:.1f}",
            ]
        )

    print_table(deepgemm_headers, deepgemm_rows, title="DeepGEMM Implementation:")

    # Print vLLM Triton table
    triton_headers = ["m", "n", "k", "Time (μs)", "TFLOPS", "GB/s", "vs DeepGEMM"]
    triton_rows = []
    for result in all_results:
        shape = result["shape"]
        impl_data = result["implementations"]["vLLM Triton"]
        speedup = impl_data.get("speedup_vs_deepgemm", 1.0)
        triton_rows.append(
            [
                shape["m"],
                shape["n"],
                shape["k"],
                f"{impl_data['time_us']:.1f}",
                f"{impl_data['tflops']:.1f}",
                f"{impl_data['gb_s']:.1f}",
                format_speedup(speedup),
            ]
        )

    print_table(triton_headers, triton_rows, title="vLLM Triton Implementation:")

    # Print vLLM CUTLASS table
    cutlass_headers = [
        "m",
        "n",
        "k",
        "Time (μs)",
        "TFLOPS",
        "GB/s",
        "vs DeepGEMM",
        "vs Triton",
    ]
    cutlass_rows = []
    for result in all_results:
        shape = result["shape"]
        impl_data = result["implementations"]["vLLM CUTLASS"]
        vs_deepgemm = impl_data.get("speedup_vs_deepgemm", 1.0)
        vs_triton = impl_data.get("speedup_vs_triton", 1.0)
        cutlass_rows.append(
            [
                shape["m"],
                shape["n"],
                shape["k"],
                f"{impl_data['time_us']:.1f}",
                f"{impl_data['tflops']:.1f}",
                f"{impl_data['gb_s']:.1f}",
                format_speedup(vs_deepgemm),
                format_speedup(vs_triton),
            ]
        )

    print_table(cutlass_headers, cutlass_rows, title="vLLM CUTLASS Implementation:")

    # Calculate and print averages
    print("\n===== AVERAGE PERFORMANCE =====")

    implementations = ["DeepGEMM", "vLLM Triton", "vLLM CUTLASS"]
    avg_metrics = {
        impl: {"tflops": 0, "gb_s": 0, "time_ms": 0} for impl in implementations
    }

    for result in all_results:
        for impl in implementations:
            impl_data = result["implementations"][impl]
            avg_metrics[impl]["tflops"] += impl_data["tflops"]
            avg_metrics[impl]["gb_s"] += impl_data["gb_s"]
            avg_metrics[impl]["time_ms"] += impl_data["time_ms"]

    num_shapes = len(all_results)
    avg_headers = ["Implementation", "Avg TFLOPS", "Avg GB/s", "Avg Time (ms)"]
    avg_rows = []

    for impl in implementations:
        avg_tflops = avg_metrics[impl]["tflops"] / num_shapes
        avg_mem_bw = avg_metrics[impl]["gb_s"] / num_shapes
        avg_time = avg_metrics[impl]["time_ms"] / num_shapes
        avg_rows.append(
            [impl, f"{avg_tflops:.2f}", f"{avg_mem_bw:.2f}", f"{avg_time:.2f}"]
        )

    print_table(avg_headers, avg_rows)

    # Calculate average speedups
    avg_speedups = {
        "DeepGEMM vs vLLM Triton": 0,
        "DeepGEMM vs vLLM CUTLASS": 0,
        "vLLM CUTLASS vs vLLM Triton": 0,
    }

    for result in all_results:
        deepgemm_time = result["implementations"]["DeepGEMM"]["time_ms"]
        vllm_triton_time = result["implementations"]["vLLM Triton"]["time_ms"]
        vllm_cutlass_time = result["implementations"]["vLLM CUTLASS"]["time_ms"]

        avg_speedups["DeepGEMM vs vLLM Triton"] += vllm_triton_time / deepgemm_time
        avg_speedups["DeepGEMM vs vLLM CUTLASS"] += vllm_cutlass_time / deepgemm_time
        avg_speedups["vLLM CUTLASS vs vLLM Triton"] += (
            vllm_triton_time / vllm_cutlass_time
        )

    print("\n===== AVERAGE SPEEDUPS =====")
    speedup_headers = ["Comparison", "Speedup"]
    speedup_rows = []
    for comparison, total in avg_speedups.items():
        avg_speedup = total / num_shapes
        status = "faster" if avg_speedup > 1 else "slower"
        speedup_rows.append([comparison, f"{avg_speedup:.2f}x {status}"])

    print_table(speedup_headers, speedup_rows)

    # Average accuracy comparison
    print("\n===== ACCURACY COMPARISON =====")
    avg_diff = {impl: 0 for impl in implementations}

    for result in all_results:
        for impl in implementations:
            avg_diff[impl] += result["implementations"][impl]["diff"]["Reference"]

    diff_headers = ["Implementation", "Avg Diff vs Reference"]
    diff_rows = []
    for impl in implementations:
        diff_rows.append([impl, f"{avg_diff[impl] / num_shapes:.6f}"])

    print_table(diff_headers, diff_rows)


if __name__ == "__main__":
    run_benchmarks(verbose=False)
