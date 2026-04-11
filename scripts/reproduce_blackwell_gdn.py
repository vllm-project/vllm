#!/usr/bin/env python3
"""
Reproduce the Blackwell SM 12.0 GDN kernel issue and verify the fix.

This script demonstrates the Triton global_scratch allocator crash that
affects MoE+Mamba models (Qwen3-Coder-Next, Qwen3.5) on Blackwell GPUs
(RTX PRO 6000, RTX 5090, RTX 5080).

Usage:
    # 1. Show the bug (deadlocks on unpatched Blackwell):
    python scripts/reproduce_blackwell_gdn.py --show-bug

    # 2. Show the allocator crash directly (Blackwell only):
    python scripts/reproduce_blackwell_gdn.py --show-allocator-crash

    # 3. Verify the fix (should pass on any GPU):
    python scripts/reproduce_blackwell_gdn.py --verify-fix

    # 4. Benchmark CUDA decode kernel:
    python scripts/reproduce_blackwell_gdn.py --benchmark

Requirements:
    - NVIDIA GPU (Blackwell SM 12.0 to reproduce the bug, any GPU for fix)
    - PyTorch with CUDA
    - Triton
"""

import argparse
import sys
import time

import torch

GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
CC = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
IS_BLACKWELL = CC[0] >= 12


def show_bug():
    """
    Reproduce the deadlock that occurs when loading MoE+Mamba models
    (Qwen3-Coder-Next, Qwen3.5) on Blackwell via vLLM.

    The deadlock chain:
    1. vLLM loads model weights into GPU memory
    2. During warmup, it calls the FLA solve_tril Triton kernel
    3. Triton autotuner benchmarks the kernel configurations
    4. On Blackwell, the kernel uses global_scratch memory
    5. NullAllocator raises RuntimeError during benchmark
    6. The autotuner catches the exception but CUDA sync is corrupted
    7. Process hangs on futex_wait_queue (deadlock)

    Without the fix, this script will hang indefinitely (timeout after 30s).
    With the fix (triton-lang/triton#10002), the kernel compiles and runs.
    """
    print(f"GPU: {GPU_NAME} (SM {CC[0]}.{CC[1]})")
    print(f"Blackwell: {IS_BLACKWELL}")
    print()

    if not IS_BLACKWELL:
        print("This deadlock only manifests on Blackwell (SM 12.0+) GPUs.")
        print("On this GPU, Triton kernels don't use global_scratch,")
        print("so the deadlock path is never triggered.")
        print()
        print("To see the fix in action, use --verify-fix instead.")
        return

    print("Reproducing the deadlock scenario...")
    print("On unpatched vLLM+Triton, this hangs forever.")
    print("With the fix applied, it completes in ~30s (first JIT compile).")
    print()

    try:
        from vllm.model_executor.layers.fla.ops import (
            chunk_gated_delta_rule as fla_chunk_gated_delta_rule,
        )
    except ImportError:
        print("ERROR: vLLM not installed. Install with: pip install -e .")
        sys.exit(1)

    import signal

    def timeout_handler(signum, frame):
        print()
        print("DEADLOCK REPRODUCED: Process hung for 30 seconds.")
        print("The Triton autotuner is stuck on futex_wait_queue.")
        print()
        print("Root cause: solve_tril kernel uses global_scratch on")
        print("Blackwell, NullAllocator crashes during autotuner benchmark,")
        print("corrupting CUDA synchronization state.")
        print()
        print("Fix: Apply triton-lang/triton#10002 (allocator fallback)")
        print("  + vllm-project/vllm#36325 or #37700 (TMA detection)")
        sys.exit(1)

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)

    B, L, H, D = 1, 64, 8, 128
    q = torch.randn(B, L, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, L, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, L, H, D, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(B, L, H, D, device="cuda", dtype=torch.bfloat16)
    beta = torch.randn(B, L, H, device="cuda", dtype=torch.bfloat16).sigmoid()

    try:
        t0 = time.time()
        out = fla_chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta,
                                         scale=1.0)
        torch.cuda.synchronize()
        signal.alarm(0)
        elapsed = time.time() - t0
        print(f"SUCCESS: FLA kernel ran in {elapsed:.1f}s (no deadlock)")
        print(f"Output shape: {out[0].shape}")
        print()
        print("The fix is applied — Triton allocator fallback works.")
    except RuntimeError as e:
        signal.alarm(0)
        if "no allocator was set" in str(e):
            print(f"ALLOCATOR CRASH: {e}")
            print()
            print("The Triton global_scratch allocator fix is NOT applied.")
            print("On a real model load, this crash causes a deadlock")
            print("because the autotuner catches it internally.")
            print()
            print("Fix: Apply triton-lang/triton#10002")
        else:
            print(f"UNEXPECTED ERROR: {e}")
            raise


def show_allocator_crash():
    """
    Directly demonstrate the Triton global_scratch allocator crash.

    This calls the FLA solve_tril kernel outside the autotuner context,
    so the RuntimeError surfaces directly instead of being caught and
    causing a deadlock.
    """
    print(f"GPU: {GPU_NAME} (SM {CC[0]}.{CC[1]})")
    print()

    if not IS_BLACKWELL:
        print("This crash only occurs on Blackwell (SM 12.0+).")
        return

    print("Calling FLA solve_tril directly on Blackwell...")
    print()

    try:
        from vllm.model_executor.layers.fla.ops.solve_tril import solve_tril
    except ImportError:
        print("ERROR: vLLM not installed.")
        sys.exit(1)

    # Create a small test input for solve_tril
    B, H, L, BT = 1, 8, 64, 64
    A = torch.randn(B, H, L, BT, device="cuda", dtype=torch.bfloat16)

    try:
        result = solve_tril(A=A)
        torch.cuda.synchronize()
        print(f"SUCCESS: solve_tril ran, output shape {result.shape}")
        print("The allocator fix is applied.")
    except RuntimeError as e:
        if "no allocator was set" in str(e):
            print(f"REPRODUCED: {e}")
            print()
            print("Fix: triton-lang/triton#10002")
        else:
            print(f"ERROR: {e}")
            raise


def verify_fix():
    """
    Verify the CUDA-native GDN decode kernel produces correct output.

    Compares the CUDA kernel output against a pure PyTorch reference
    implementation for the GDN recurrence:
        h *= exp(g)
        v -= sum(h * k, dim=k)
        v *= sigmoid(b)
        h += outer(v, k)
        o = sum(h * q, dim=k)
    """
    print(f"GPU: {GPU_NAME} (SM {CC[0]}.{CC[1]})")
    print()

    # Build standalone CUDA kernel
    print("Building CUDA GDN decode kernel...")
    try:
        from torch.utils.cpp_extension import load
        import os
        kernel_path = os.path.join(os.path.dirname(__file__),
                                    "../csrc/mamba/gdn_decode_kernels.cu")
        if not os.path.exists(kernel_path):
            print(f"ERROR: Kernel source not found at {kernel_path}")
            sys.exit(1)

        ext = load(name="gdn_decode_test",
                   sources=[kernel_path],
                   extra_cuda_cflags=["-O2"],
                   verbose=False)
        print("Build OK")
    except Exception as e:
        print(f"Build failed: {e}")
        sys.exit(1)

    # Test configurations
    configs = [
        # (H, HV, K, V, B, description)
        (16, 64, 128, 128, 1, "Qwen3.5-397B single"),
        (16, 64, 128, 128, 4, "Qwen3.5-397B batch=4"),
        (8, 32, 128, 128, 1, "Medium model"),
        (4, 16, 64, 64, 8, "Small model batch=8"),
    ]

    all_passed = True
    for H, HV, K, V, B, desc in configs:
        torch.manual_seed(42)
        num_slots = B + 2
        scale = K ** -0.5

        q = torch.randn(B, H, K, device="cuda", dtype=torch.float32)
        k = torch.randn(B, H, K, device="cuda", dtype=torch.float32)
        v = torch.randn(B, HV, V, device="cuda", dtype=torch.float32)
        g_decay = torch.rand(B, HV, device="cuda") * 0.5 + 0.5
        beta = torch.rand(B, HV, device="cuda")
        state = torch.randn(num_slots, HV, V, K,
                            dtype=torch.float32, device="cuda") * 0.01
        state_ref = state.clone()
        state_indices = torch.arange(B, dtype=torch.int32, device="cuda")

        # CUDA kernel
        out_cuda = torch.zeros(B, HV, V, dtype=torch.float32, device="cuda")
        ext.gdn_decode_step(q, k, v, g_decay, beta,
                            state, out_cuda, state_indices,
                            scale, False)
        torch.cuda.synchronize()

        # PyTorch reference
        out_ref = torch.zeros_like(out_cuda)
        hpg = HV // H
        for b in range(B):
            slot = state_indices[b].item()
            for hv in range(HV):
                hi = hv // hpg
                h = state_ref[slot, hv].clone()
                h *= g_decay[b, hv]
                for vi in range(V):
                    kvm = (h[vi] * k[b, hi]).sum()
                    d = (v[b, hv, vi] - kvm) * beta[b, hv]
                    h[vi] += k[b, hi] * d
                    out_ref[b, hv, vi] = (h[vi] * q[b, hi] * scale).sum()
                state_ref[slot, hv] = h

        out_diff = (out_cuda - out_ref).abs().max().item()
        state_diff = (state - state_ref).abs().max().item()
        passed = out_diff < 0.01 and state_diff < 0.01
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {desc:30s} out_diff={out_diff:.6f} "
              f"state_diff={state_diff:.6f}")
        if not passed:
            all_passed = False

    # PAD_SLOT_ID test
    state_before = state.clone()
    out_pad = torch.zeros(1, HV, V, dtype=torch.float32, device="cuda")
    pad_idx = torch.tensor([-1], dtype=torch.int32, device="cuda")
    ext.gdn_decode_step(q[:1], k[:1], v[:1], g_decay[:1], beta[:1],
                        state, out_pad, pad_idx, scale, False)
    pad_ok = (out_pad == 0).all() and (state == state_before).all()
    print(f"  {'PASS' if pad_ok else 'FAIL'}: PAD_SLOT_ID (-1) handling")
    if not pad_ok:
        all_passed = False

    print()
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


def benchmark():
    """Benchmark CUDA decode kernel throughput."""
    print(f"GPU: {GPU_NAME} (SM {CC[0]}.{CC[1]})")
    print()

    from torch.utils.cpp_extension import load
    import os
    kernel_path = os.path.join(os.path.dirname(__file__),
                                "../csrc/mamba/gdn_decode_kernels.cu")
    ext = load(name="gdn_decode_bench", sources=[kernel_path],
               extra_cuda_cflags=["-O2"], verbose=False)

    for B in [1, 4, 8]:
        H, HV, K, V = 16, 64, 128, 128  # Qwen3.5-397B dims
        scale = K ** -0.5

        q = torch.randn(B, H, K, device="cuda", dtype=torch.float32)
        k = torch.randn(B, H, K, device="cuda", dtype=torch.float32)
        v = torch.randn(B, HV, V, device="cuda", dtype=torch.float32)
        g = torch.rand(B, HV, device="cuda") * 0.5 + 0.5
        beta = torch.rand(B, HV, device="cuda")
        state = torch.randn(B + 2, HV, V, K,
                            dtype=torch.float32, device="cuda") * 0.01
        idx = torch.arange(B, dtype=torch.int32, device="cuda")
        out = torch.zeros(B, HV, V, dtype=torch.float32, device="cuda")

        # Warmup
        for _ in range(3):
            ext.gdn_decode_step(q, k, v, g, beta, state, out, idx,
                                scale, False)
        torch.cuda.synchronize()

        # Benchmark
        iters = 100
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            ext.gdn_decode_step(q, k, v, g, beta, state, out, idx,
                                scale, False)
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        us_per_call = (elapsed / iters) * 1e6
        print(f"  batch={B:2d}: {us_per_call:8.1f} µs/step "
              f"({iters/elapsed:.0f} steps/s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reproduce and verify Blackwell GDN kernel fix")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--show-bug", action="store_true",
                       help="Reproduce the deadlock (hangs on unpatched Blackwell)")
    group.add_argument("--show-allocator-crash", action="store_true",
                       help="Show the Triton allocator RuntimeError directly")
    group.add_argument("--verify-fix", action="store_true",
                       help="Verify CUDA decode kernel correctness")
    group.add_argument("--benchmark", action="store_true",
                       help="Benchmark CUDA decode kernel")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    if args.show_bug:
        show_bug()
    elif args.show_allocator_crash:
        show_allocator_crash()
    elif args.verify_fix:
        verify_fix()
    elif args.benchmark:
        benchmark()
