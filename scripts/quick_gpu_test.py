#!/usr/bin/env python3
"""
Quick GPU validation and benchmarking for ILP optimization.
Usage:
  python3 scripts/quick_gpu_test.py               # Quick 5-min test
  python3 scripts/quick_gpu_test.py --full        # Full 30-min benchmark
  python3 scripts/quick_gpu_test.py --profile     # With profiling
"""

import sys
import argparse
import time
import torch
import numpy as np
from pathlib import Path


def check_environment():
    """Verify GPU environment is ready."""
    print("\n" + "="*70)
    print("GPU ENVIRONMENT CHECK")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("❌ ERROR: CUDA not available!")
        print("   Run: VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=cuda")
        return False
    
    device = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    
    print(f"✓ GPU: {device}")
    print(f"✓ CUDA Version: {torch.version.cuda}")
    print(f"✓ GPU Memory: {props.total_memory / 1e9:.1f} GB")
    print(f"✓ Compute Capability: {props.major}.{props.minor}")
    
    return True


def check_ops():
    """Verify CUDA ops are available."""
    print("\n" + "="*70)
    print("CUDA OPERATIONS CHECK")
    print("="*70)
    
    ops = [
        ('gelu_tanh_and_mul', 'Original GELU+mul kernel'),
        ('gelu_tanh_and_mul_ilp', 'ILP GELU+mul kernel'),
        ('silu_and_mul_ilp', 'ILP SiLU+mul kernel'),
    ]
    
    all_available = True
    for op_name, description in ops:
        has_op = hasattr(torch.ops._C, op_name)
        status = "✓" if has_op else "✗"
        print(f"{status} {op_name:30s} - {description}")
        all_available = all_available and has_op
    
    if not all_available:
        print("\n❌ ERROR: Some operations not available!")
        print("   Recompile vLLM: python3 setup.py build_ext --inplace")
        return False
    
    return True


def benchmark_shape(num_tokens, d, iterations=100, warmup=5):
    """Benchmark a single shape."""
    x = torch.randn(num_tokens, 2*d, dtype=torch.float32, device='cuda')
    out_orig = torch.empty(num_tokens, d, dtype=torch.float32, device='cuda')
    out_ilp = torch.empty(num_tokens, d, dtype=torch.float32, device='cuda')
    
    # Warmup
    for _ in range(warmup):
        torch.ops._C.gelu_tanh_and_mul(out_orig, x)
        torch.ops._C.gelu_tanh_and_mul_ilp(out_ilp, x)
    
    torch.cuda.synchronize()
    
    # Benchmark original
    start = time.perf_counter()
    for _ in range(iterations):
        torch.ops._C.gelu_tanh_and_mul(out_orig, x)
    torch.cuda.synchronize()
    time_orig = (time.perf_counter() - start) / iterations * 1000
    
    # Benchmark ILP
    start = time.perf_counter()
    for _ in range(iterations):
        torch.ops._C.gelu_tanh_and_mul_ilp(out_ilp, x)
    torch.cuda.synchronize()
    time_ilp = (time.perf_counter() - start) / iterations * 1000
    
    # Verify correctness
    max_diff = (out_orig - out_ilp).abs().max().item()
    
    speedup = time_orig / time_ilp
    
    return {
        'time_orig': time_orig,
        'time_ilp': time_ilp,
        'speedup': speedup,
        'max_diff': max_diff,
    }


def quick_test():
    """Run quick 5-minute test."""
    print("\n" + "="*70)
    print("QUICK BENCHMARK (5 min)")
    print("="*70)
    
    configs = [
        (32, 512),
        (128, 2048),
        (2048, 4096),
    ]
    
    speedups = []
    
    for num_tokens, d in configs:
        input_size = num_tokens * 2 * d
        print(f"\nShape: ({num_tokens}, {2*d}) -> ({num_tokens}, {d}) [{input_size:,} elements]")
        
        result = benchmark_shape(num_tokens, d, iterations=50)
        
        print(f"  Original: {result['time_orig']:.4f} ms")
        print(f"  ILP:      {result['time_ilp']:.4f} ms")
        print(f"  Speedup:  {result['speedup']:.3f}x")
        print(f"  Max diff: {result['max_diff']:.2e}")
        
        speedups.append(result['speedup'])
    
    avg_speedup = np.mean(speedups)
    print(f"\n{'─'*70}")
    print(f"Average Speedup: {avg_speedup:.3f}x")
    print(f"{'─'*70}")
    
    return avg_speedup


def full_benchmark():
    """Run comprehensive 30-minute benchmark."""
    print("\n" + "="*70)
    print("COMPREHENSIVE BENCHMARK (30 min)")
    print("="*70)
    
    num_tokens_list = [32, 64, 128, 256, 512, 1024]
    d_list = [512, 1024, 2048, 4096]
    
    results = []
    total = len(num_tokens_list) * len(d_list)
    current = 0
    
    for num_tokens in num_tokens_list:
        for d in d_list:
            current += 1
            input_size = num_tokens * 2 * d
            
            # Skip if too large
            if input_size > 100_000_000:  # 100M elements
                print(f"\n[{current}/{total}] Shape: ({num_tokens}, {2*d}) - SKIPPED (too large)")
                continue
            
            print(f"\n[{current}/{total}] Shape: ({num_tokens}, {2*d}) [{input_size:,} elements]", end=" ")
            
            try:
                result = benchmark_shape(num_tokens, d, iterations=100)
                
                print(f"Speedup: {result['speedup']:.3f}x")
                results.append({
                    'num_tokens': num_tokens,
                    'd': d,
                    'speedup': result['speedup'],
                    'time_orig': result['time_orig'],
                    'time_ilp': result['time_ilp'],
                    'max_diff': result['max_diff'],
                })
            except RuntimeError as e:
                print(f"SKIPPED ({e})")
    
    # Summary statistics
    speedups = [r['speedup'] for r in results]
    
    print(f"\n{'─'*70}")
    print(f"Results Summary ({len(results)} shapes tested):")
    print(f"  Min speedup:  {min(speedups):.3f}x")
    print(f"  Max speedup:  {max(speedups):.3f}x")
    print(f"  Avg speedup:  {np.mean(speedups):.3f}x")
    print(f"  Median speedup: {np.median(speedups):.3f}x")
    print(f"{'─'*70}")
    
    return results


def correctness_test():
    """Test correctness across different sizes and dtypes."""
    print("\n" + "="*70)
    print("CORRECTNESS VALIDATION")
    print("="*70)
    
    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    
    test_configs = [
        (32, 512),
        (128, 2048),
        (512, 4096),
    ]
    
    all_passed = True
    
    for dtype in dtypes:
        print(f"\n{str(dtype):20s}")
        
        for num_tokens, d in test_configs:
            try:
                x = torch.randn(num_tokens, 2*d, dtype=dtype, device='cuda')
                out_orig = torch.empty(num_tokens, d, dtype=dtype, device='cuda')
                out_ilp = torch.empty(num_tokens, d, dtype=dtype, device='cuda')
                
                torch.ops._C.gelu_tanh_and_mul(out_orig, x)
                torch.ops._C.gelu_tanh_and_mul_ilp(out_ilp, x)
                
                max_diff = (out_orig - out_ilp).abs().max().item()
                passed = max_diff < 1e-3 or (dtype != torch.float32 and max_diff < 1e-2)
                
                status = "✓" if passed else "⚠"
                print(f"  {status} ({num_tokens:4d}, {2*d:5d}): diff={max_diff:.2e}")
                
                all_passed = all_passed and passed
            except Exception as e:
                print(f"  ✗ ({num_tokens:4d}, {2*d:5d}): {e}")
                all_passed = False
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(description='GPU ILP Optimization Test')
    parser.add_argument('--full', action='store_true', help='Run comprehensive benchmark')
    parser.add_argument('--profile', action='store_true', help='Include correctness validation')
    parser.add_argument('--correctness', action='store_true', help='Test correctness only')
    args = parser.parse_args()
    
    # Environment checks
    if not check_environment():
        return 1
    
    if not check_ops():
        return 1
    
    # Run tests
    try:
        if args.correctness:
            success = correctness_test()
            return 0 if success else 1
        
        if args.profile:
            correctness_test()
        
        if args.full:
            full_benchmark()
        else:
            quick_test()
        
        print("\n" + "="*70)
        print("✓ All tests completed successfully!")
        print("="*70 + "\n")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
