#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark script for ILP-optimized GELU kernel.
Compares original vs ILP-optimized versions to measure speedup
from instruction-level parallelism and loop unrolling.
"""

import time
import torch
import numpy as np
from typing import Dict, Tuple, List
import argparse


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
    }


def run_ilp_benchmark(
    num_tokens_list: List[int] = [32, 128, 2048],
    d_list: List[int] = [512, 4096],
    num_iterations: int = 100,
) -> None:
    """
    Run comprehensive ILP optimization benchmark.
    
    Args:
        num_tokens_list: List of token counts to test
        d_list: List of embedding dimensions to test
        num_iterations: Number of benchmark iterations per test
    """
    
    print("=" * 100)
    print("GELU ILP Optimization Benchmark")
    print("=" * 100)
    print("\nComparing original kernel vs ILP-optimized kernel")
    print("ILP kernel uses 4-element loop unrolling to hide transcendental function latency\n")
    
    try:
        import torch.ops._C as ops_C
    except ImportError:
        print("ERROR: vLLM CUDA ops not available. Ensure vLLM is compiled with CUDA.")
        return
    
    results = {}
    
    for d in d_list:
        for num_tokens in num_tokens_list:
            print(f"\nShape: ({num_tokens}, {2*d}) -> ({num_tokens}, {d})")
            print("-" * 80)
            
            # Create test tensors
            x = torch.randn(num_tokens, 2 * d, dtype=torch.float32, device='cuda')
            out_orig = torch.empty(num_tokens, d, dtype=torch.float32, device='cuda')
            out_ilp = torch.empty(num_tokens, d, dtype=torch.float32, device='cuda')
            
            try:
                # Benchmark standard GELU
                print("  Standard GELU (original kernel):")
                stats_orig = benchmark_kernel(
                    torch.ops._C.gelu_and_mul,
                    x,
                    out_orig,
                    num_iterations=num_iterations,
                )
                print(f"    Time: {stats_orig['time_per_iter_ms']:.4f} ms")
                print(f"    Throughput: {stats_orig['throughput_gbps']:.2f} GB/s")
                
                # Benchmark ILP GELU
                print("  Standard GELU (ILP kernel):")
                stats_ilp = benchmark_kernel(
                    torch.ops._C.gelu_and_mul_ilp,
                    x,
                    out_ilp,
                    num_iterations=num_iterations,
                )
                print(f"    Time: {stats_ilp['time_per_iter_ms']:.4f} ms")
                print(f"    Throughput: {stats_ilp['throughput_gbps']:.2f} GB/s")
                
                # Compute speedup
                speedup = stats_orig['time_per_iter_ms'] / stats_ilp['time_per_iter_ms']
                print(f"  ILP Speedup: {speedup:.3f}x")
                
                # Verify correctness
                max_diff = torch.abs(out_orig - out_ilp).max().item()
                rel_error = max_diff / (torch.abs(out_orig).max().item() + 1e-8)
                print(f"  Max difference: {max_diff:.2e} (rel: {rel_error:.2e})")
                
                # Benchmark GELU tanh
                print("\n  GELU tanh (original kernel):")
                out_tanh_orig = torch.empty(num_tokens, d, dtype=torch.float32, device='cuda')
                stats_tanh_orig = benchmark_kernel(
                    torch.ops._C.gelu_tanh_and_mul,
                    x,
                    out_tanh_orig,
                    num_iterations=num_iterations,
                )
                print(f"    Time: {stats_tanh_orig['time_per_iter_ms']:.4f} ms")
                print(f"    Throughput: {stats_tanh_orig['throughput_gbps']:.2f} GB/s")
                
                print("  GELU tanh (ILP kernel):")
                out_tanh_ilp = torch.empty(num_tokens, d, dtype=torch.float32, device='cuda')
                stats_tanh_ilp = benchmark_kernel(
                    torch.ops._C.gelu_tanh_and_mul_ilp,
                    x,
                    out_tanh_ilp,
                    num_iterations=num_iterations,
                )
                print(f"    Time: {stats_tanh_ilp['time_per_iter_ms']:.4f} ms")
                print(f"    Throughput: {stats_tanh_ilp['throughput_gbps']:.2f} GB/s")
                
                speedup_tanh = stats_tanh_orig['time_per_iter_ms'] / stats_tanh_ilp['time_per_iter_ms']
                print(f"  ILP Speedup (tanh): {speedup_tanh:.3f}x")
                
                max_diff_tanh = torch.abs(out_tanh_orig - out_tanh_ilp).max().item()
                print(f"  Max difference: {max_diff_tanh:.2e}")
                
                # Benchmark SiLU
                print("\n  SiLU (original kernel):")
                out_silu_orig = torch.empty(num_tokens, d, dtype=torch.float32, device='cuda')
                stats_silu_orig = benchmark_kernel(
                    torch.ops._C.silu_and_mul,
                    x,
                    out_silu_orig,
                    num_iterations=num_iterations,
                )
                print(f"    Time: {stats_silu_orig['time_per_iter_ms']:.4f} ms")
                print(f"    Throughput: {stats_silu_orig['throughput_gbps']:.2f} GB/s")
                
                print("  SiLU (ILP kernel):")
                out_silu_ilp = torch.empty(num_tokens, d, dtype=torch.float32, device='cuda')
                stats_silu_ilp = benchmark_kernel(
                    torch.ops._C.silu_and_mul_ilp,
                    x,
                    out_silu_ilp,
                    num_iterations=num_iterations,
                )
                print(f"    Time: {stats_silu_ilp['time_per_iter_ms']:.4f} ms")
                print(f"    Throughput: {stats_silu_ilp['throughput_gbps']:.2f} GB/s")
                
                speedup_silu = stats_silu_orig['time_per_iter_ms'] / stats_silu_ilp['time_per_iter_ms']
                print(f"  ILP Speedup (SiLU): {speedup_silu:.3f}x")
                
                max_diff_silu = torch.abs(out_silu_orig - out_silu_ilp).max().item()
                print(f"  Max difference: {max_diff_silu:.2e}")
                
                # Store results
                key = (num_tokens, d)
                results[key] = {
                    'gelu_speedup': speedup,
                    'gelu_tanh_speedup': speedup_tanh,
                    'silu_speedup': speedup_silu,
                    'gelu_stats_orig': stats_orig,
                    'gelu_stats_ilp': stats_ilp,
                }
                
            except AttributeError as e:
                print(f"  ❌ ILP kernels not available (recompile vLLM): {e}")
                break
    
    # Summary
    if results:
        print("\n" + "=" * 100)
        print("SUMMARY")
        print("=" * 100)
        
        print("\nGELU (erf) - ILP Speedup:")
        print("-" * 50)
        for (num_tokens, d), res in results.items():
            speedup = res['gelu_speedup']
            status = "✓ Good" if speedup > 1.1 else "✗ No improvement" if speedup < 0.99 else "~ Minimal"
            print(f"  {num_tokens:4d} tokens, dim={d:5d}: {speedup:.3f}x {status}")
        
        print("\nGELU (tanh) - ILP Speedup:")
        print("-" * 50)
        for (num_tokens, d), res in results.items():
            speedup = res['gelu_tanh_speedup']
            status = "✓ Good" if speedup > 1.1 else "✗ No improvement" if speedup < 0.99 else "~ Minimal"
            print(f"  {num_tokens:4d} tokens, dim={d:5d}: {speedup:.3f}x {status}")
        
        print("\nSiLU - ILP Speedup:")
        print("-" * 50)
        for (num_tokens, d), res in results.items():
            speedup = res['silu_speedup']
            status = "✓ Good" if speedup > 1.1 else "✗ No improvement" if speedup < 0.99 else "~ Minimal"
            print(f"  {num_tokens:4d} tokens, dim={d:5d}: {speedup:.3f}x {status}")
        
        # Average speedup
        avg_speedup_gelu = np.mean([r['gelu_speedup'] for r in results.values()])
        avg_speedup_tanh = np.mean([r['gelu_tanh_speedup'] for r in results.values()])
        avg_speedup_silu = np.mean([r['silu_speedup'] for r in results.values()])
        
        print("\n" + "=" * 100)
        print(f"Average ILP Speedup - GELU(erf): {avg_speedup_gelu:.3f}x")
        print(f"Average ILP Speedup - GELU(tanh): {avg_speedup_tanh:.3f}x")
        print(f"Average ILP Speedup - SiLU: {avg_speedup_silu:.3f}x")
        print("=" * 100 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM ILP-optimized activation kernels"
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=[32, 128, 2048],
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
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations"
    )
    
    args = parser.parse_args()
    
    run_ilp_benchmark(
        num_tokens_list=args.num_tokens,
        d_list=args.d,
        num_iterations=args.iterations,
    )
