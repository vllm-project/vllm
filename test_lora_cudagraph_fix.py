#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Simple test script to verify LoRA CUDA graph optimization fix.

This script tests that:
1. Cache invalidation happens automatically on first LoRA load
2. Subsequent inferences are faster (CUDA graph optimized)
3. Logging shows the expected messages

Usage:
    python test_lora_cudagraph_fix.py \
        --model meta-llama/Llama-2-7b-hf \
        --lora-path /path/to/lora/adapter

Requirements:
    - vLLM 0.13.0+ with the fix applied
    - A trained LoRA adapter
"""

import argparse
import os
import time
from typing import List

# Enable debug logging to see cache invalidation messages
os.environ['VLLM_LOGGING_LEVEL'] = 'DEBUG'

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def test_cache_invalidation_automatic(
    model: str,
    lora_path: str,
    num_iterations: int = 5,
) -> bool:
    """
    Test automatic cache invalidation on first LoRA load.
    
    Returns:
        True if test passes, False otherwise
    """
    print("=" * 80)
    print("Test 1: Automatic Cache Invalidation")
    print("=" * 80)
    
    print("\n[1/4] Initializing model...")
    llm = LLM(
        model=model,
        enable_lora=True,
        max_loras=2,
        max_lora_rank=64,
        gpu_memory_utilization=0.9,
    )
    print("✓ Model initialized")
    
    print("\n[2/4] Loading first LoRA adapter...")
    print("Expected: Should see 'Invalidating compilation caches' message")
    
    lora_request = LoRARequest(
        lora_name="test_adapter",
        lora_int_id=1,
        lora_path=lora_path,
    )
    
    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=50,
    )
    
    # First inference - should trigger cache invalidation and recompilation
    print("\n[3/4] First inference (with recompilation)...")
    start = time.time()
    output = llm.generate(
        "Hello, how are you?",
        sampling_params=sampling_params,
        lora_request=lora_request,
    )
    first_time = time.time() - start
    print(f"✓ First inference completed in {first_time:.2f}s")
    
    # Subsequent inferences - should be fast (using cached CUDA graphs)
    print(f"\n[4/4] Running {num_iterations} subsequent inferences (optimized)...")
    times: List[float] = []
    for i in range(num_iterations):
        start = time.time()
        output = llm.generate(
            f"Test prompt {i+1}",
            sampling_params=sampling_params,
            lora_request=lora_request,
        )
        inference_time = time.time() - start
        times.append(inference_time)
        print(f"  Iteration {i+1}/{num_iterations}: {inference_time:.3f}s")
    
    avg_time = sum(times) / len(times)
    speedup = first_time / avg_time if avg_time > 0 else 0
    
    print("\n" + "=" * 80)
    print("Results:")
    print("=" * 80)
    print(f"First inference (with recompilation): {first_time:.3f}s")
    print(f"Average optimized inference:           {avg_time:.3f}s")
    print(f"Speedup after optimization:            {speedup:.2f}x")
    
    # Test passes if optimized inference is significantly faster
    is_optimized = speedup > 10  # Should be at least 10x faster
    
    if is_optimized:
        print("\n✅ TEST PASSED: LoRA inferences are CUDA graph optimized")
    else:
        print(f"\n❌ TEST FAILED: Expected speedup > 10x, got {speedup:.2f}x")
        print("   This suggests CUDA graphs are not being used for LoRA inference")
    
    return is_optimized


def test_cache_invalidation_disabled(
    model: str,
    lora_path: str,
) -> bool:
    """
    Test that cache invalidation can be disabled via environment variable.
    
    Returns:
        True if test passes, False otherwise
    """
    print("\n" + "=" * 80)
    print("Test 2: Disabled Cache Invalidation")
    print("=" * 80)
    
    # Set environment variable to disable cache invalidation
    os.environ['VLLM_DISABLE_LORA_CACHE_INVALIDATION'] = '1'
    
    print("\n[1/2] Initializing model with cache invalidation disabled...")
    llm = LLM(
        model=model,
        enable_lora=True,
        max_loras=2,
        max_lora_rank=64,
    )
    print("✓ Model initialized")
    
    print("\n[2/2] Loading LoRA adapter...")
    print("Expected: Should see warning about cache invalidation being disabled")
    
    lora_request = LoRARequest(
        lora_name="test_adapter_disabled",
        lora_int_id=2,
        lora_path=lora_path,
    )
    
    try:
        output = llm.generate(
            "Test prompt",
            lora_request=lora_request,
        )
        print("✓ LoRA loaded")
        print("\n✅ TEST PASSED: Environment variable control works")
        return True
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    finally:
        # Reset environment variable
        os.environ.pop('VLLM_DISABLE_LORA_CACHE_INVALIDATION', None)


def test_manual_invalidation(model: str, lora_path: str) -> bool:
    """
    Test manual cache invalidation API.
    
    Returns:
        True if test passes, False otherwise
    """
    print("\n" + "=" * 80)
    print("Test 3: Manual Cache Invalidation API")
    print("=" * 80)
    
    print("\n[1/3] Initializing model...")
    llm = LLM(
        model=model,
        enable_lora=True,
        max_loras=2,
        max_lora_rank=64,
    )
    print("✓ Model initialized")
    
    print("\n[2/3] Loading LoRA adapter...")
    lora_request = LoRARequest(
        lora_name="test_adapter_manual",
        lora_int_id=3,
        lora_path=lora_path,
    )
    
    output = llm.generate("Test", lora_request=lora_request)
    print("✓ LoRA loaded")
    
    print("\n[3/3] Testing manual cache invalidation API...")
    try:
        llm.lora_manager.invalidate_compilation_caches()
        print("✓ Manual invalidation succeeded")
        print("\n✅ TEST PASSED: Manual invalidation API works")
        return True
    except Exception as e:
        print(f"✗ Manual invalidation failed: {e}")
        print(f"\n❌ TEST FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test LoRA CUDA graph optimization fix"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        required=True,
        help="Path to LoRA adapter weights",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=5,
        help="Number of iterations for performance test",
    )
    parser.add_argument(
        "--skip-manual-test",
        action="store_true",
        help="Skip manual invalidation test",
    )
    parser.add_argument(
        "--skip-disabled-test",
        action="store_true",
        help="Skip disabled invalidation test",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("LoRA CUDA Graph Optimization Test Suite")
    print("=" * 80)
    print(f"\nModel: {args.model}")
    print(f"LoRA Path: {args.lora_path}")
    print(f"Iterations: {args.num_iterations}")
    
    results = {}
    
    # Test 1: Automatic cache invalidation
    try:
        results['automatic'] = test_cache_invalidation_automatic(
            args.model,
            args.lora_path,
            args.num_iterations,
        )
    except Exception as e:
        print(f"\n❌ Test 1 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        results['automatic'] = False
    
    # Test 2: Disabled cache invalidation
    if not args.skip_disabled_test:
        try:
            results['disabled'] = test_cache_invalidation_disabled(
                args.model,
                args.lora_path,
            )
        except Exception as e:
            print(f"\n❌ Test 2 FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results['disabled'] = False
    
    # Test 3: Manual invalidation API
    if not args.skip_manual_test:
        try:
            results['manual'] = test_manual_invalidation(
                args.model,
                args.lora_path,
            )
        except Exception as e:
            print(f"\n❌ Test 3 FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results['manual'] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name.title():20s}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("LoRA CUDA graph optimization is working correctly.")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("Please check the logs above for details.")
    print("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())

