#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark script for GELU kernel optimization.
Profiles and compares:
1. Standard GELU (with erf)
2. GELU tanh approximation
3. Accuracy vs PyTorch reference
"""

import time
import torch
import numpy as np
from typing import Tuple, Dict, Any
import argparse


def gelu_cpu_ref(x: torch.Tensor) -> torch.Tensor:
    """PyTorch CPU reference GELU implementation."""
    return torch.nn.functional.gelu(x, approximate='none')


def gelu_tanh_cpu_ref(x: torch.Tensor) -> torch.Tensor:
    """PyTorch CPU reference GELU tanh approximation."""
    return torch.nn.functional.gelu(x, approximate='tanh')


def benchmark_kernel(
    kernel_fn,
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    num_warmup: int = 5,
    num_iterations: int = 100,
) -> Dict[str, float]:
    """
    Benchmark a kernel function.
    
    Args:
        kernel_fn: The kernel function to benchmark
        input_tensor: Input tensor
        output_tensor: Output tensor (preallocated)
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations
    
    Returns:
        Dictionary with timing statistics
    """
    # Warmup
    for _ in range(num_warmup):
        kernel_fn(output_tensor, input_tensor)
    
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iterations):
        kernel_fn(output_tensor, input_tensor)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    elapsed_ms = (end - start) * 1000.0
    total_elements = input_tensor.numel()
    
    return {
        "elapsed_ms": elapsed_ms,
        "iterations": num_iterations,
        "time_per_iter_ms": elapsed_ms / num_iterations,
        "throughput_gbps": (
            total_elements * input_tensor.element_size() * num_iterations 
            / (elapsed_ms * 1e6)
        ),
        "gflops": (
            total_elements * num_iterations * 1.5  # Rough estimate
            / (elapsed_ms * 1e6)
        ),
    }


def compute_accuracy_metrics(
    gpu_output: torch.Tensor,
    ref_output: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute accuracy metrics comparing GPU vs CPU reference.
    
    Args:
        gpu_output: GPU kernel output
        ref_output: CPU reference output
    
    Returns:
        Dictionary with accuracy metrics
    """
    # Convert to float32 for comparison
    gpu_f32 = gpu_output.float()
    ref_f32 = ref_output.float()
    
    diff = torch.abs(gpu_f32 - ref_f32)
    rel_diff = diff / (torch.abs(ref_f32) + 1e-8)
    
    return {
        "max_abs_error": diff.max().item(),
        "mean_abs_error": diff.mean().item(),
        "max_rel_error": rel_diff.max().item(),
        "mean_rel_error": rel_diff.mean().item(),
    }


def run_benchmark_suite(
    num_tokens_list: list,
    d_list: list,
    dtype_list: list,
    num_iterations: int = 100,
) -> None:
    """
    Run comprehensive benchmark suite comparing GELU variants.
    
    Args:
        num_tokens_list: List of token counts to test
        d_list: List of embedding dimensions to test
        dtype_list: List of data types to test
        num_iterations: Number of benchmark iterations
    """
    
    print("=" * 100)
    print("vLLM GELU Kernel Benchmark Suite")
    print("=" * 100)
    print()
    
    # Import vLLM activation layers
    from vllm.model_executor.layers.activation import GeluAndMul
    
    results = {}
    
    for dtype in dtype_list:
        print(f"\n{'='*100}")
        print(f"Data Type: {dtype}")
        print(f"{'='*100}\n")
        
        for num_tokens in num_tokens_list:
            for d in d_list:
                print(f"Shape: ({num_tokens}, {2*d}) -> ({num_tokens}, {d})")
                print("-" * 80)
                
                # Create test tensors
                # Input shape: [num_tokens, 2*d] (split between x and y)
                x = torch.randn(num_tokens, 2 * d, dtype=dtype, device='cuda')
                out_standard = torch.empty(num_tokens, d, dtype=dtype, device='cuda')
                out_tanh = torch.empty(num_tokens, d, dtype=dtype, device='cuda')
                
                # Create activation layers
                gelu_layer = GeluAndMul(approximate='none')
                gelu_tanh_layer = GeluAndMul(approximate='tanh')
                
                # Benchmark standard GELU
                print("  Standard GELU (erf-based):")
                stats_standard = benchmark_kernel(
                    gelu_layer,
                    x,
                    out_standard,
                    num_iterations=num_iterations,
                )
                print(f"    Time: {stats_standard['time_per_iter_ms']:.4f} ms")
                print(f"    Throughput: {stats_standard['throughput_gbps']:.2f} GB/s")
                print(f"    GFLOPs: {stats_standard['gflops']:.2f}")
                
                # Benchmark tanh GELU
                print("  GELU tanh approximation:")
                stats_tanh = benchmark_kernel(
                    gelu_tanh_layer,
                    x,
                    out_tanh,
                    num_iterations=num_iterations,
                )
                print(f"    Time: {stats_tanh['time_per_iter_ms']:.4f} ms")
                print(f"    Throughput: {stats_tanh['throughput_gbps']:.2f} GB/s")
                print(f"    GFLOPs: {stats_tanh['gflops']:.2f}")
                
                # Compute speedup
                speedup = stats_standard['time_per_iter_ms'] / stats_tanh['time_per_iter_ms']
                print(f"  Speedup (tanh vs std): {speedup:.3f}x")
                
                # Accuracy comparison
                print("\n  Accuracy vs PyTorch reference:")
                
                # Get CPU reference for standard GELU
                x_cpu = x.cpu()
                x_split = x_cpu.chunk(2, dim=-1)
                ref_standard = gelu_cpu_ref(x_split[0]) * x_split[1]
                
                acc_standard = compute_accuracy_metrics(
                    out_standard.cpu(),
                    ref_standard.to(dtype),
                )
                print(f"    Standard GELU:")
                print(f"      Max absolute error: {acc_standard['max_abs_error']:.2e}")
                print(f"      Mean absolute error: {acc_standard['mean_abs_error']:.2e}")
                print(f"      Max relative error: {acc_standard['max_rel_error']:.2e}")
                print(f"      Mean relative error: {acc_standard['mean_rel_error']:.2e}")
                
                # Get CPU reference for tanh GELU
                ref_tanh = gelu_tanh_cpu_ref(x_split[0]) * x_split[1]
                
                acc_tanh = compute_accuracy_metrics(
                    out_tanh.cpu(),
                    ref_tanh.to(dtype),
                )
                print(f"    GELU tanh approximation:")
                print(f"      Max absolute error: {acc_tanh['max_abs_error']:.2e}")
                print(f"      Mean absolute error: {acc_tanh['mean_abs_error']:.2e}")
                print(f"      Max relative error: {acc_tanh['max_rel_error']:.2e}")
                print(f"      Mean relative error: {acc_tanh['mean_rel_error']:.2e}")
                
                # Error between standard and tanh
                diff = torch.abs(out_standard - out_tanh)
                print(f"\n  Difference (standard vs tanh approximation):")
                print(f"    Max: {diff.max().item():.2e}")
                print(f"    Mean: {diff.mean().item():.2e}")
                
                # Store results
                key = (num_tokens, d, dtype)
                results[key] = {
                    'standard': stats_standard,
                    'tanh': stats_tanh,
                    'acc_standard': acc_standard,
                    'acc_tanh': acc_tanh,
                    'speedup': speedup,
                }
                
                print()
    
    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print()
    print("Speedup (tanh vs standard GELU):")
    print("-" * 50)
    for (num_tokens, d, dtype), res in results.items():
        speedup = res['speedup']
        print(f"  {num_tokens} tokens, dim={d}, {dtype}: {speedup:.3f}x")
    
    # Find fastest configuration
    fastest_key = max(results.keys(), 
                      key=lambda k: results[k]['speedup'])
    fastest_speedup = results[fastest_key]['speedup']
    print(f"\nFastest tanh approximation: {fastest_speedup:.3f}x at {fastest_key}")
    
    # Average accuracy loss
    print("\nAverage accuracy impact of tanh approximation:")
    print("-" * 50)
    avg_abs_error_diff = np.mean([
        res['acc_tanh']['max_abs_error'] - res['acc_standard']['max_abs_error']
        for res in results.values()
    ])
    print(f"  Avg max abs error increase: {avg_abs_error_diff:.2e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM GELU kernels"
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=[7, 128, 2048],
        help="Token counts to benchmark"
    )
    parser.add_argument(
        "--d",
        type=int,
        nargs="+",
        default=[512, 4096],
        help="Embedding dimensions to benchmark"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        nargs="+",
        choices=["float", "float16", "bfloat16"],
        default=["float", "float16", "bfloat16"],
        help="Data types to benchmark"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations"
    )
    
    args = parser.parse_args()
    
    # Convert dtype strings to torch dtypes
    dtype_map = {
        "float": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype_list = [dtype_map[d] for d in args.dtype]
    
    run_benchmark_suite(
        num_tokens_list=args.num_tokens,
        d_list=args.d,
        dtype_list=dtype_list,
        num_iterations=args.iterations,
    )
