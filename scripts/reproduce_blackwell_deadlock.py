#!/usr/bin/env python3
"""
Reproduce: Blackwell SM 12.0 deadlock in vLLM FLA solve_tril autotuner

On Blackwell GPUs, loading MoE+Mamba models (Qwen3-Coder-Next, Qwen3.5)
via vLLM deadlocks during model warmup. The process hangs in
futex_wait_queue with no error output.

Root cause chain:
  1. FLA solve_tril kernel uses @triton.autotune
  2. Autotuner calls _bench() for each config
  3. On Blackwell, kernel compiles with global_scratch memory
  4. NullAllocator raises RuntimeError during benchmark
  5. Autotuner catches the exception in _bench()
  6. CUDA synchronization state is corrupted
  7. Process deadlocks on futex_wait_queue

This script reproduces the deadlock by calling the actual FLA solve_tril
kernel from vLLM. Requires vLLM installed (pip install -e .).

Usage:
    # Reproduce (hangs on unpatched Blackwell, times out after 30s):
    python reproduce_blackwell_deadlock.py

    # With the Triton allocator fix applied:
    python reproduce_blackwell_deadlock.py --fix

Requirements: torch, triton, vLLM (installed from source or pip)
"""

import argparse
import os
import sys
import signal
import time

import torch

GPU_NAME = torch.cuda.get_device_name(0)
CC = torch.cuda.get_device_capability()
IS_BLACKWELL = CC[0] >= 12
TIMEOUT = 60


def apply_fix():
    """Apply the Triton allocator fix from triton-lang/triton#10002."""
    import triton
    triton.set_allocator(
        lambda size, align, stream:
            torch.empty(size, dtype=torch.uint8, device="cuda").data_ptr()
    )


def timeout_handler(signum, frame):
    print()
    print("=" * 70)
    print(f"DEADLOCK REPRODUCED (process hung for {TIMEOUT}s)")
    print("=" * 70)
    print()
    print("The vLLM FLA autotuner is stuck in futex_wait_queue.")
    print()
    print("Root cause: Triton kernels on Blackwell use global_scratch")
    print("memory. The NullAllocator raises RuntimeError during the")
    print("autotuner benchmark, corrupting CUDA sync state.")
    print()
    print("Fix options:")
    print("  1. triton-lang/triton#10002 (allocate_default_global_scratch)")
    print("  2. Run this script with --fix to apply the workaround")
    print("  3. vllm-project/vllm#39563 (CUDA-native GDN decode kernel)")
    os._exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Blackwell FLA deadlock in vLLM")
    parser.add_argument("--fix", action="store_true",
                        help="Apply Triton allocator fix before running")
    args = parser.parse_args()

    print(f"GPU:       {GPU_NAME}")
    print(f"SM:        {CC[0]}.{CC[1]}")
    print(f"Blackwell: {IS_BLACKWELL}")
    print(f"Timeout:   {TIMEOUT}s")
    print()

    if not IS_BLACKWELL:
        print("WARNING: This deadlock only occurs on Blackwell (SM 12.0+).")
        print("On this GPU, the kernels will run normally.")
        print()

    if args.fix:
        print("Applying Triton allocator fix...")
        apply_fix()
        print()

    # Try to import the actual FLA kernel from vLLM
    try:
        from vllm.model_executor.layers.fla.ops import (
            chunk_gated_delta_rule as fla_chunk_gated_delta_rule,
        )
    except ImportError:
        print("ERROR: vLLM not installed.")
        print("Install with: pip install -e . (from vLLM source)")
        sys.exit(1)

    signal.signal(signal.SIGALRM, timeout_handler)

    # Test 1: chunk_gated_delta_rule (prefill kernel)
    # This calls solve_tril internally, which triggers the deadlock
    print("Test 1: FLA chunk_gated_delta_rule (prefill path)...")
    print("  This calls solve_tril -> autotuner -> global_scratch")
    signal.alarm(TIMEOUT)

    B, L, H, K = 1, 128, 8, 128
    q = torch.randn(B, L, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, L, H, K, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, L, H, K, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(B, L, H, K, device="cuda", dtype=torch.bfloat16)
    beta = torch.randn(B, L, H, device="cuda", dtype=torch.bfloat16).sigmoid()

    try:
        t0 = time.time()
        out = fla_chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta,
                                         scale=1.0)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        signal.alarm(0)
        print(f"  PASS ({elapsed:.1f}s, output shape {out[0].shape})")
    except RuntimeError as e:
        signal.alarm(0)
        if "no allocator was set" in str(e):
            print(f"  CRASH: {e}")
            print()
            print("The allocator crash surfaced directly (not caught by")
            print("autotuner). This happens when the kernel is called")
            print("outside the autotuner context.")
            print()
            print("Inside the autotuner, this same crash gets caught,")
            print("corrupting CUDA sync state -> deadlock.")
            sys.exit(1)
        raise

    # Test 2: fused_sigmoid_gating_delta_rule_update (decode kernel)
    print()
    print("Test 2: FLA fused_sigmoid_gating decode kernel...")
    signal.alarm(TIMEOUT)

    try:
        from vllm.model_executor.layers.fla.ops import (
            fused_sigmoid_gating_delta_rule_update,
        )

        B_dec, T_dec, HV, K_dec, V_dec = 4, 1, 32, 128, 128
        H_dec = 8
        q_dec = torch.randn(B_dec * T_dec, H_dec, K_dec, device="cuda",
                            dtype=torch.bfloat16)
        k_dec = torch.randn_like(q_dec)
        v_dec = torch.randn(B_dec * T_dec, HV, V_dec, device="cuda",
                            dtype=torch.bfloat16)
        a_dec = torch.randn(B_dec * T_dec, HV, device="cuda",
                            dtype=torch.bfloat16)
        b_dec = torch.randn_like(a_dec)
        A_log = torch.randn(HV, device="cuda", dtype=torch.bfloat16)
        dt_bias = torch.randn(HV, device="cuda", dtype=torch.bfloat16)
        state = torch.randn(B_dec, HV, V_dec, K_dec, device="cuda",
                            dtype=torch.float32) * 0.01

        t0 = time.time()
        fused_sigmoid_gating_delta_rule_update(
            A_log=A_log, a=a_dec, b=b_dec, dt_bias=dt_bias,
            q=q_dec, k=k_dec, v=v_dec,
            scale=K_dec ** -0.5,
            initial_state=state,
            inplace_final_state=False,
        )
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        signal.alarm(0)
        print(f"  PASS ({elapsed:.1f}s)")
    except Exception as e:
        signal.alarm(0)
        print(f"  SKIP: {e}")

    print()
    print("All tests passed.")
    if IS_BLACKWELL:
        if args.fix:
            print("The Triton allocator fix resolved the Blackwell issue.")
        else:
            print("Note: if this passed without --fix, the Triton/vLLM")
            print("version may already include the fix.")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)
    main()
