#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Simple profiling script to understand GELU kernel behavior.
This script helps identify bottlenecks before optimization.
"""

import torch
import time
from vllm.model_executor.layers.activation import GeluAndMul


def profile_gelu_kernels(num_tokens: int = 128, d: int = 4096, num_runs: int = 50):
    """Profile GELU kernel variants with detailed timing."""
    
    print("\n" + "="*80)
    print("GELU Kernel Profiling")
    print("="*80)
    print(f"Config: num_tokens={num_tokens}, d={d}, dtype=float32")
    print(f"Input shape: ({num_tokens}, {2*d}), Output shape: ({num_tokens}, {d})")
    print(f"Element count: {num_tokens * 2 * d:,}")
    print(f"Memory: Input={num_tokens * 2 * d * 4 / 1e6:.1f}MB, Output={num_tokens * d * 4 / 1e6:.1f}MB")
    print("="*80 + "\n")
    
    # Create test data
    x = torch.randn(num_tokens, 2 * d, dtype=torch.float32, device='cuda')
    
    # Create GELU layers
    gelu_std = GeluAndMul(approximate='none')
    gelu_tanh = GeluAndMul(approximate='tanh')
    
    # Test 1: Standard GELU with erf
    print("Test 1: Standard GELU (erf-based)")
    print("-" * 80)
    
    out_std = torch.empty(num_tokens, d, dtype=torch.float32, device='cuda')
    
    # Warmup
    for _ in range(5):
        gelu_std(x)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(num_runs):
        out_std = gelu_std(x)
    
    torch.cuda.synchronize()
    elapsed_std = (time.perf_counter() - start) * 1000
    
    time_per_iter = elapsed_std / num_runs
    throughput = (num_tokens * 2 * d * 4) / (time_per_iter * 1e6)  # GB/s
    
    print(f"  Total time: {elapsed_std:.2f} ms ({num_runs} runs)")
    print(f"  Time per iteration: {time_per_iter:.4f} ms")
    print(f"  Throughput: {throughput:.2f} GB/s")
    print(f"  Output dtype: {out_std.dtype}")
    print(f"  Output sample (first 10 elements): {out_std[0, :10]}\n")
    
    # Test 2: GELU tanh approximation
    print("Test 2: GELU tanh approximation")
    print("-" * 80)
    
    out_tanh = torch.empty(num_tokens, d, dtype=torch.float32, device='cuda')
    
    # Warmup
    for _ in range(5):
        gelu_tanh(x)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(num_runs):
        out_tanh = gelu_tanh(x)
    
    torch.cuda.synchronize()
    elapsed_tanh = (time.perf_counter() - start) * 1000
    
    time_per_iter = elapsed_tanh / num_runs
    throughput = (num_tokens * 2 * d * 4) / (time_per_iter * 1e6)  # GB/s
    
    print(f"  Total time: {elapsed_tanh:.2f} ms ({num_runs} runs)")
    print(f"  Time per iteration: {time_per_iter:.4f} ms")
    print(f"  Throughput: {throughput:.2f} GB/s")
    print(f"  Output dtype: {out_tanh.dtype}")
    print(f"  Output sample (first 10 elements): {out_tanh[0, :10]}\n")
    
    # Test 3: Comparison
    print("Test 3: Performance Comparison")
    print("-" * 80)
    
    speedup = elapsed_std / elapsed_tanh
    print(f"  Speedup (tanh vs std): {speedup:.3f}x")
    print(f"  Time difference: {elapsed_std - elapsed_tanh:.2f} ms ({(speedup - 1) * 100:.1f}% faster)")
    
    # Test 4: Accuracy comparison
    print("\nTest 4: Accuracy Comparison")
    print("-" * 80)
    
    # Compute reference using PyTorch
    x_split = x.chunk(2, dim=-1)
    x_val = x_split[0]
    y_val = x_split[1]
    
    ref_std = torch.nn.functional.gelu(x_val, approximate='none') * y_val
    ref_tanh = torch.nn.functional.gelu(x_val, approximate='tanh') * y_val
    
    # Compute errors
    err_std = torch.abs(out_std - ref_std).max().item()
    err_tanh = torch.abs(out_tanh - ref_tanh).max().item()
    
    print(f"  Max error vs PyTorch (standard GELU): {err_std:.2e}")
    print(f"  Max error vs PyTorch (tanh GELU): {err_tanh:.2e}")
    
    # Difference between the two implementations
    diff = torch.abs(out_std - out_tanh).max().item()
    print(f"  Max difference (std vs tanh): {diff:.2e}")
    print(f"  Relative error (tanh vs std): {diff / (torch.abs(out_std).max().item() + 1e-8):.2e}")
    
    # Test 5: Memory efficiency
    print("\nTest 5: Memory Efficiency")
    print("-" * 80)
    
    input_bytes = num_tokens * 2 * d * 4
    output_bytes = num_tokens * d * 4
    total_bytes = input_bytes + output_bytes
    
    print(f"  Input memory: {input_bytes / 1e6:.1f} MB")
    print(f"  Output memory: {output_bytes / 1e6:.1f} MB")
    print(f"  Total memory: {total_bytes / 1e6:.1f} MB")
    print(f"  Arithmetic intensity (ops/byte): ~1.5 / element")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # Profile with different configurations
    configs = [
        (32, 4096),    # Small batch
        (128, 4096),   # Medium batch
        (2048, 4096),  # Large batch
    ]
    
    for num_tokens, d in configs:
        try:
            profile_gelu_kernels(num_tokens=num_tokens, d=d, num_runs=50)
        except Exception as e:
            print(f"Error profiling {num_tokens} tokens, {d} dim: {e}")
            continue
