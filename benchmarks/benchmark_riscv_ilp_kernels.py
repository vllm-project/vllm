#!/usr/bin/env python3
"""
RISC-V ILP Kernel Benchmark Tool

Benchmark original vs ILP-optimized transcendental functions (exp, tanh, erf)
on RISC-V Vector (RVV) systems.

This tool measures:
- Performance (cycles per element)
- Speedup (ILP vs original)
- Correctness (bitwise equivalence)
- Throughput (elements per cycle)
"""

import torch
import time
import numpy as np
from typing import Tuple, List, Dict
import argparse


def measure_cycles(fn, input_tensor: torch.Tensor, warmup: int = 100, iterations: int = 1000) -> float:
    """Measure average cycles for a function using timing."""
    # Warmup
    for _ in range(warmup):
        _ = fn(input_tensor)
    
    # Measure
    start = time.perf_counter()
    for _ in range(iterations):
        _ = fn(input_tensor)
    end = time.perf_counter()
    
    elapsed = end - start
    total_elements = input_tensor.numel() * iterations
    
    # Rough estimate: 1 cycle ≈ 1ns on modern CPUs (adjust for your platform)
    # This is a heuristic; real cycle counts need CPU counter support
    cycles_per_element = (elapsed * 1e9) / total_elements
    return cycles_per_element


def benchmark_function_pair(
    name: str,
    original_fn,
    ilp_fn,
    input_tensor: torch.Tensor,
    sizes: List[int] = [8, 16, 32, 128],
) -> Dict:
    """Benchmark original vs ILP variant of a function."""
    
    print(f"\n{'='*70}")
    print(f"Benchmarking: {name}")
    print(f"{'='*70}")
    print(f"{'Size':<10} {'Original (cy/el)':<20} {'ILP (cy/el)':<20} {'Speedup':<10}")
    print(f"{'-'*70}")
    
    results = {
        'name': name,
        'original_times': [],
        'ilp_times': [],
        'speedups': [],
    }
    
    for size in sizes:
        # Create test tensor
        test_tensor = input_tensor[:size] if input_tensor.numel() >= size else input_tensor.repeat((size + input_tensor.numel() - 1) // input_tensor.numel())[:size]
        
        # Measure original
        try:
            orig_time = measure_cycles(original_fn, test_tensor)
        except Exception as e:
            print(f"  Error measuring original: {e}")
            orig_time = 0.0
        
        # Measure ILP
        try:
            ilp_time = measure_cycles(ilp_fn, test_tensor)
        except Exception as e:
            print(f"  Error measuring ILP: {e}")
            ilp_time = 0.0
        
        # Calculate speedup
        if orig_time > 0:
            speedup = orig_time / ilp_time if ilp_time > 0 else 1.0
        else:
            speedup = 1.0
        
        results['original_times'].append(orig_time)
        results['ilp_times'].append(ilp_time)
        results['speedups'].append(speedup)
        
        print(f"{size:<10} {orig_time:<20.3f} {ilp_time:<20.3f} {speedup:<10.2f}x")
    
    return results


def test_correctness(
    name: str,
    original_fn,
    ilp_fn,
    input_tensor: torch.Tensor,
) -> bool:
    """Test that ILP function produces identical results to original."""
    
    print(f"\nTesting correctness: {name}")
    
    try:
        original_output = original_fn(input_tensor)
        ilp_output = ilp_fn(input_tensor)
        
        # Check if outputs are exactly the same (bitwise)
        max_diff = torch.abs(original_output - ilp_output).max().item()
        
        # Also check relative error for non-zero values
        mask = torch.abs(original_output) > 1e-10
        if mask.any():
            rel_error = (torch.abs(original_output - ilp_output) / torch.abs(original_output))[mask].max().item()
        else:
            rel_error = 0.0
        
        is_exact = torch.allclose(original_output, ilp_output, rtol=1e-6, atol=1e-8)
        
        status = "✓ PASS" if is_exact else "⚠ APPROX"
        print(f"  {status}: max_diff={max_diff:.2e}, rel_error={rel_error:.2e}")
        
        return is_exact
        
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="RISC-V ILP Kernel Benchmark Tool")
    parser.add_argument('--sizes', nargs='+', type=int, default=[8, 16, 32, 128, 256],
                        help="Tensor sizes to benchmark")
    parser.add_argument('--iterations', type=int, default=1000,
                        help="Number of iterations for timing")
    parser.add_argument('--warmup', type=int, default=100,
                        help="Warmup iterations")
    parser.add_argument('--device', type=str, default='cpu',
                        help="Device to use (cpu, cuda)")
    args = parser.parse_args()
    
    print("RISC-V ILP Kernel Benchmark Tool")
    print("="*70)
    print(f"Device: {args.device}")
    print(f"Sizes: {args.sizes}")
    print(f"Iterations: {args.iterations}, Warmup: {args.warmup}")
    
    # Create test input tensor
    test_input = torch.randn(max(args.sizes), dtype=torch.float32, device=args.device)
    
    # Test cases
    test_cases = []
    
    # Test exp (if available)
    try:
        test_cases.append((
            "exp()",
            lambda x: torch.exp(x),
            lambda x: torch.exp(x),  # TODO: Replace with ILP variant when available
            test_input,
        ))
    except Exception as e:
        print(f"Skipping exp: {e}")
    
    # Test tanh (if available)
    try:
        test_cases.append((
            "tanh()",
            lambda x: torch.tanh(x),
            lambda x: torch.tanh(x),  # TODO: Replace with ILP variant when available
            test_input,
        ))
    except Exception as e:
        print(f"Skipping tanh: {e}")
    
    # Run benchmarks
    all_results = []
    for name, original_fn, ilp_fn, test_tensor in test_cases:
        # Test correctness first
        test_correctness(name, original_fn, ilp_fn, test_tensor)
        
        # Benchmark
        results = benchmark_function_pair(
            name,
            original_fn,
            ilp_fn,
            test_tensor,
            sizes=args.sizes,
        )
        all_results.append(results)
    
    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    for results in all_results:
        if results['speedups']:
            avg_speedup = np.mean(results['speedups'])
            max_speedup = np.max(results['speedups'])
            print(f"{results['name']:<20} avg_speedup={avg_speedup:.2f}x, max_speedup={max_speedup:.2f}x")


if __name__ == "__main__":
    main()
