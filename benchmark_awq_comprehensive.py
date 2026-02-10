#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Comprehensive AWQ GEMV and GEMM benchmark and correctness test."""

import os

os.environ["VLLM_USE_TRITON_AWQ_GEMV"] = "1"

import argparse

import torch

from vllm.triton_utils import triton


def check_gpu_idle(threshold_pct=5.0):
    """Check if GPU is idle before running benchmarks.

    Returns (is_idle, utilization, error_msg) tuple:
    - is_idle: True if GPU utilization is below threshold
    - utilization: GPU utilization percentage (None if couldn't check)
    - error_msg: Error message if couldn't check (None otherwise)
    """
    import json
    import subprocess

    try:
        # Use rocm-smi to get GPU utilization
        result = subprocess.run(
            ["rocm-smi", "--showuse", "--json"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            # Parse GPU utilization from rocm-smi output
            for card_key, card_data in data.items():
                if card_key.startswith("card"):
                    gpu_use = card_data.get("GPU use (%)", "0")
                    # Handle "N/A" or other non-numeric values
                    try:
                        utilization = float(gpu_use.replace("%", "").strip())
                    except (ValueError, AttributeError):
                        utilization = 0.0

                    return (utilization <= threshold_pct, utilization, None)
        else:
            # Try alternative method using rocm-smi without JSON
            result = subprocess.run(
                ["rocm-smi", "-u"], capture_output=True, text=True, timeout=5
            )
            if "GPU use" in result.stdout:
                for line in result.stdout.split("\n"):
                    if "GPU use" in line and "%" in line:
                        # Parse "GPU use (%):                 0"
                        parts = line.split(":")
                        if len(parts) >= 2:
                            try:
                                utilization = float(parts[-1].strip())
                                return (utilization <= threshold_pct, utilization, None)
                            except ValueError:
                                pass
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
        return (True, None, str(e))

    return (True, None, "Could not parse GPU utilization")


def load_autoawq_kernel():
    """Load the AutoAWQ GEMV kernel compiled with HIP."""
    from torch.utils.cpp_extension import load

    kernel_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "autoawq_kernels"
    )

    # Find thrust headers (needed by PyTorch headers)
    site_packages = os.path.dirname(torch.__file__)
    thrust_include = os.path.join(site_packages, "..", "cupy/_core/include/cupy/_cccl")
    thrust_include = os.path.abspath(thrust_include)

    extra_cuda_flags = ["-O3"]
    if os.path.exists(thrust_include):
        extra_cuda_flags.insert(0, f"-isystem{thrust_include}")

    return load(
        name="autoawq_gemv",
        sources=[
            os.path.join(kernel_dir, "binding.cpp"),
            os.path.join(kernel_dir, "gemv_cuda.cu"),
        ],
        extra_include_paths=[thrust_include] if os.path.exists(thrust_include) else [],
        extra_cuda_cflags=extra_cuda_flags,
        verbose=False,
    )


def convert_vllm_to_autoawq_tensors(qweight, qzeros, scales, group_size):
    """
    Convert vLLM AWQ tensors to AutoAWQ format.

    vLLM layout:
      qweight: [K, N//8] int32 with AWQ packing order [0,4,1,5,2,6,3,7]
      qzeros: [K//G, N//8] int32 with AWQ packing order
      scales: [K//G, N] fp16

    AutoAWQ layout:
      kernel: [N, K//8] int32 with sequential packing [0,1,2,3,4,5,6,7]
      zeros: [N, K//G//8] int32 with sequential packing
      scales: [N, K//G] fp16

    The key differences:
    1. Transposed dimensions (K,N -> N,K)
    2. Different int4 packing order within int32
    """
    K = qweight.shape[0]
    N = qweight.shape[1] * 8
    num_groups = K // group_size

    # Step 1: Unpack vLLM weights with AWQ order
    # AWQ order: [0, 4, 1, 5, 2, 6, 3, 7] - meaning bit position i*4 contains element at AWQ_order[i]
    awq_order = [0, 4, 1, 5, 2, 6, 3, 7]
    reverse_awq_order = [
        awq_order.index(i) for i in range(8)
    ]  # [0, 2, 4, 6, 1, 3, 5, 7]

    # Unpack qweight [K, N//8] -> [K, N] int4 values
    weights_unpacked = torch.zeros(K, N, dtype=torch.int32, device=qweight.device)
    for i in range(8):
        shift = reverse_awq_order[i] * 4
        weights_unpacked[:, i::8] = (qweight >> shift) & 0xF

    # Step 2: Transpose to [N, K]
    weights_transposed = weights_unpacked.T.contiguous()  # [N, K]

    # Step 3: Repack with sequential order for AutoAWQ [N, K//8]
    kernel = torch.zeros(N, K // 8, dtype=torch.int32, device=qweight.device)
    for i in range(8):
        kernel |= (weights_transposed[:, i::8] & 0xF) << (i * 4)

    # Step 4: Convert zeros similarly
    # Unpack qzeros [K//G, N//8] -> [K//G, N]
    zeros_unpacked = torch.zeros(num_groups, N, dtype=torch.int32, device=qzeros.device)
    for i in range(8):
        shift = reverse_awq_order[i] * 4
        zeros_unpacked[:, i::8] = (qzeros >> shift) & 0xF

    # Transpose to [N, K//G]
    zeros_transposed = zeros_unpacked.T.contiguous()  # [N, K//G]

    # AutoAWQ zeros_w = make_divisible(num_groups, 8) for g128
    zeros_packed_dim = (num_groups + 7) // 8
    zeros = torch.zeros(N, zeros_packed_dim, dtype=torch.int32, device=qzeros.device)
    for i in range(num_groups):
        pack_idx = i // 8
        bit_pos = (i % 8) * 4
        zeros[:, pack_idx] |= (zeros_transposed[:, i] & 0xF) << bit_pos

    # Step 5: Transpose scales [K//G, N] -> [N, sf_w]
    # AutoAWQ sf_w = make_divisible(num_groups, 8) * 8 for g128
    sf_w = ((num_groups + 7) // 8) * 8
    scales_awq = torch.zeros(N, sf_w, dtype=scales.dtype, device=scales.device)
    scales_transposed = scales.T.contiguous()  # [N, num_groups]
    scales_awq[:, :num_groups] = scales_transposed

    return kernel, zeros, scales_awq


def test_autoawq_correctness(autoawq_module, shapes, awq_dequantize_triton):
    """Run correctness tests for AutoAWQ kernel using vLLM's dequantize as reference."""
    print("\n" + "=" * 90)
    print("AUTOAWQ CORRECTNESS TEST")
    print("=" * 90)
    print(
        f"{'N':>6} x {'K':<6} | {'G':>4} | {'Max Diff':>12} | {'Mean Diff':>12} | {'Status':>8}"
    )
    print("-" * 90)

    all_pass = True
    for N, K, group_size in shapes:
        if K % group_size != 0:
            print(f"{N:>6} x {K:<6} | {group_size:>4} | K not divisible by group_size")
            continue

        # AutoAWQ only supports group_size 64 and 128
        if group_size not in [64, 128]:
            print(
                f"{N:>6} x {K:<6} | {group_size:>4} | {'N/A':>12} | {'N/A':>12} | {'SKIP':>8} (g!=64,128)"
            )
            continue

        # N must be divisible by 4 for the kernel grid
        if N % 4 != 0:
            print(
                f"{N:>6} x {K:<6} | {group_size:>4} | {'N/A':>12} | {'N/A':>12} | {'SKIP':>8} (N%4!=0)"
            )
            continue

        # AutoAWQ kernel requires K >= 1024 for g128, K >= 512 for g64
        # (each warp iteration processes 1024 or 512 elements)
        min_k = 1024 if group_size == 128 else 512
        if min_k > K:
            print(
                f"{N:>6} x {K:<6} | {group_size:>4} | {'N/A':>12} | {'N/A':>12} | {'SKIP':>8} (K<{min_k})"
            )
            continue

        try:
            torch.manual_seed(42)
            num_groups = K // group_size

            # Create vLLM format tensors
            input_tensor = torch.randn(1, K, dtype=torch.float16, device="cuda") * 0.01
            qweight = torch.randint(
                0, 2**31, (K, N // 8), dtype=torch.int32, device="cuda"
            )
            qzeros = torch.randint(
                0, 2**31, (num_groups, N // 8), dtype=torch.int32, device="cuda"
            )
            scales = (
                torch.randn(num_groups, N, dtype=torch.float16, device="cuda") * 0.01
            )

            # Compute reference using vLLM's dequantize
            dequantized = awq_dequantize_triton(qweight, scales, qzeros)
            output_ref = torch.matmul(input_tensor, dequantized)

            # Convert to AutoAWQ format
            kernel_awq, zeros_awq, scales_awq = convert_vllm_to_autoawq_tensors(
                qweight, qzeros, scales, group_size
            )

            # Run AutoAWQ kernel
            output_awq = autoawq_module.gemv_forward(
                input_tensor, kernel_awq, scales_awq, zeros_awq, group_size
            )

            # Synchronize to catch any async errors
            torch.cuda.synchronize()

            # Compare
            diff = (output_awq - output_ref).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            # Check for NaN (indicates kernel crash or invalid memory access)
            has_nan = (
                torch.isnan(output_awq).any().item()
                or torch.isnan(output_ref).any().item()
            )

            # Use tolerance for fp16 precision (0.5 is reasonable for accumulated fp16 errors)
            passed = not has_nan and max_diff < 0.5
            all_pass = all_pass and passed

            if has_nan:
                status = "NaN!"
            elif passed:
                status = "PASS"
            else:
                status = "FAIL"

            print(
                f"{N:>6} x {K:<6} | {group_size:>4} | {max_diff:>12.6f} | {mean_diff:>12.6f} | {status:>8}"
            )
        except Exception:
            print(
                f"{N:>6} x {K:<6} | {group_size:>4} | {'CRASH':>12} | {'CRASH':>12} | {'CRASH':>8}"
            )
            all_pass = False
            # Reset GPU state
            torch.cuda.synchronize()

    print("-" * 90)
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass


def convert_vllm_to_autoawq_format(qweight, qzeros, scales, group_size):
    """
    Create AutoAWQ format tensors with same shapes for bandwidth comparison.

    Note: For fair bandwidth comparison, we use random data in the correct layout
    since the actual values don't affect memory throughput.
    """
    K = qweight.shape[0]
    N = qweight.shape[1] * 8
    num_groups = K // group_size

    # Create tensors in AutoAWQ layout with proper shapes
    # kernel: [OC, IC // 8] = [N, K // 8]
    kernel_awq = torch.randint(0, 2**31, (N, K // 8), dtype=torch.int32, device="cuda")

    # zeros: [OC, IC // G // 8] = [N, num_groups // 8]
    zeros_packed = (num_groups + 7) // 8  # Round up
    zeros_awq = torch.randint(
        0, 2**31, (N, zeros_packed), dtype=torch.int32, device="cuda"
    )

    # scales: [OC, IC // G] = [N, num_groups]
    scales_awq = torch.randn(N, num_groups, dtype=torch.float16, device="cuda") * 0.01

    return kernel_awq, zeros_awq, scales_awq


def main():
    parser = argparse.ArgumentParser(description="AWQ GEMV/GEMM Benchmark")
    parser.add_argument(
        "--correctness-only", action="store_true", help="Only run correctness tests"
    )
    parser.add_argument(
        "--benchmark-only", action="store_true", help="Only run benchmarks"
    )
    parser.add_argument(
        "--exhaustive",
        action="store_true",
        help="Run exhaustive search for optimal GEMV configs",
    )
    parser.add_argument(
        "--gemm-tuning", action="store_true", help="Run exhaustive GEMM tuning for M>1"
    )
    parser.add_argument(
        "--gemm-quick",
        action="store_true",
        help="Quick GEMM tuning with subset of shapes/configs",
    )
    parser.add_argument(
        "--autoawq", action="store_true", help="Compare against AutoAWQ CUDA kernel"
    )
    parser.add_argument(
        "-M",
        type=int,
        default=1,
        help="Batch size / number of tokens (default: 1). "
        "M=1 uses GEMV kernels; M>1 uses GEMM path.",
    )
    parser.add_argument(
        "--peak-bw",
        type=float,
        default=None,
        help="Peak read bandwidth in GiB/s (auto-detected if not set)",
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=200,
        help="Number of repetitions for each benchmark (default: 200)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Number of warmup iterations (default: 50)",
    )
    args = parser.parse_args()

    # Check GPU utilization early, before any GPU work (result printed in summary)
    gpu_idle, gpu_utilization, gpu_error = check_gpu_idle()

    # Import after setting env var
    from vllm.model_executor.layers.quantization.awq_triton import (
        _awq_gemm_triton,
        _choose_optimal_config,
        _get_valid_split_k_values,
        awq_dequantize_triton,
        awq_gemm_kernel,
        awq_gemv_kernel_split_k,
        reduce_split_k_kernel,
    )

    # All shapes to test (N, K, group_size)
    # Format: (N, K, group_size) where input is [1, K] and output is [1, N]
    SHAPES = [
        # Qwen2.5-0.5B-Instruct (hidden=896, intermediate=4864)
        # (896, 896) and (1152, 896) removed: hit GEMM kernel (K<=1024, N<=2000)
        # (896, 4864) removed: hits Triton GEMV (N<1500)
        (9728, 896, 128),  # gate_up_proj (4864 * 2)
        # Gemma-2B (hidden=2048, intermediate=16384, vocab=256000)
        (2048, 2048, 128),  # q/o proj
        (2560, 2048, 128),  # qkv fused (2048 + 256 + 256 for MQA)
        (32768, 2048, 128),  # gate_up_proj (16384 * 2)
        (2048, 16384, 128),  # down_proj
        # Qwen3-1.7B / Qwen3-VL-2B / Cosmos-Reason2-2B (hidden=2048, intermediate=6144)
        (4096, 2048, 128),  # qkv fused (2048 + 1024 + 1024)
        (12288, 2048, 128),  # gate_up_proj (6144 * 2)
        (2048, 6144, 128),  # down_proj
        # Qwen3-4B (hidden=2560, intermediate=9728, 32 heads, 8 kv heads)
        (2560, 2560, 128),  # q/o proj
        (6144, 2560, 128),  # qkv fused (actual from profile)
        (19456, 2560, 128),  # gate_up_proj (9728 * 2)
        (2560, 9728, 128),  # down_proj
        # Qwen2.5-VL-3B (hidden=2048, intermediate=11008)
        (22016, 2048, 128),  # gate_up_proj (11008 * 2)
        (2048, 11008, 128),  # down_proj
        # Qwen2.5-7B / Qwen2.5-VL-7B (hidden=3584, intermediate=18944)
        (3584, 3584, 128),  # q/o proj
        (4608, 3584, 128),  # qkv fused (3584 + 512 + 512)
        (37888, 3584, 128),  # gate_up_proj (18944 * 2)
        (3584, 18944, 128),  # down_proj
        (152064, 3584, 128),  # lm_head (Qwen2.5-VL-7B)
        # LLaMA-3.1-8B (hidden=4096, intermediate=14336)
        (4096, 4096, 128),  # q/o proj
        (6144, 4096, 128),  # qkv fused (4096 + 1024 + 1024 for GQA)
        (28672, 4096, 128),  # gate_up_proj (14336 * 2)
        (4096, 14336, 128),  # down_proj
        # LLaMA2-7B shapes (hidden=4096, intermediate=11008)
        (11008, 4096, 128),  # up_proj or gate_proj
        (12288, 4096, 128),  # qkv alternative
        (22016, 4096, 128),  # gate_up_proj (11008 * 2)
        (4096, 11008, 128),  # down_proj
        # Additional test shapes
        (6144, 2560, 128),
        (2560, 4096, 128),
        (19456, 9728, 128),
    ]

    # =========================================================================
    # Per-GPU reference performance (GiB/s) and peak bandwidth
    # =========================================================================
    # Best known performance (GiB/s) for each shape (N, K, group_size).
    # Used to detect performance regressions.
    # Tolerance: 5% below reference triggers a warning.
    PERF_TOLERANCE = 0.05  # 5% tolerance for measurement noise

    # Radeon 8060S (gfx1151): Strix Halo, 16 CUs, LPDDR5X-8000
    # Updated: 2026-02-10, triton.testing.do_bench median, max of 2 runs
    BEST_KNOWN_PERF_8060S = {
        # (N, K, group_size): best_known_gibs
        #
        # Qwen2.5-0.5B-Instruct
        (9728, 896, 128): 142.9,
        # Gemma-2B
        (2048, 2048, 128): 111.5,
        (2560, 2048, 128): 132.6,
        (32768, 2048, 128): 178.6,
        (2048, 16384, 128): 159.4,
        # Qwen3-1.7B / Qwen3-VL-2B / Cosmos-Reason2-2B
        (4096, 2048, 128): 148.7,
        (12288, 2048, 128): 188.3,
        (2048, 6144, 128): 158.6,
        # Qwen3-4B
        (2560, 2560, 128): 143.8,
        (6144, 2560, 128): 175.4,
        (19456, 2560, 128): 188.7,
        (2560, 9728, 128): 159.4,
        # Qwen2.5-VL-3B
        (22016, 2048, 128): 186.0,
        (2048, 11008, 128): 144.1,
        # Qwen2.5-7B / Qwen2.5-VL-7B
        (3584, 3584, 128): 145.4,
        (4608, 3584, 128): 152.5,
        (37888, 3584, 128): 195.8,
        (3584, 18944, 128): 185.4,
        (152064, 3584, 128): 199.3,
        # LLaMA-3.1-8B
        (4096, 4096, 128): 176.2,
        (6144, 4096, 128): 184.2,
        (28672, 4096, 128): 200.5,
        (4096, 14336, 128): 188.4,
        # LLaMA2-7B
        (11008, 4096, 128): 170.5,
        (12288, 4096, 128): 183.3,
        (22016, 4096, 128): 194.7,
        (4096, 11008, 128): 165.5,
        # Additional
        (2560, 4096, 128): 157.3,
        (19456, 9728, 128): 199.0,
    }
    PEAK_BW_8060S = 215.1  # GiB/s measured via copy benchmark (231 GB/s)

    # Radeon 890M (gfx1150): Strix Point, 8 CUs, LPDDR5X-5600
    # Updated: 2026-02-09, triton.testing.do_bench timing, min of 2 runs
    BEST_KNOWN_PERF_890M = {
        # (N, K, group_size): best_known_gibs
        #
        # Qwen2.5-0.5B-Instruct
        (9728, 896, 128): 44.3,
        # Gemma-2B
        (2048, 2048, 128): 45.9,
        (2560, 2048, 128): 41.5,
        (32768, 2048, 128): 73.5,
        (2048, 16384, 128): 58.5,
        # Qwen3-1.7B / Qwen3-VL-2B / Cosmos-Reason2-2B
        (4096, 2048, 128): 53.1,
        (12288, 2048, 128): 70.2,
        (2048, 6144, 128): 61.5,
        # Qwen3-4B
        (2560, 2560, 128): 46.5,
        (6144, 2560, 128): 63.3,
        (19456, 2560, 128): 73.7,
        (2560, 9728, 128): 63.4,
        # Qwen2.5-VL-3B
        (22016, 2048, 128): 73.7,
        (2048, 11008, 128): 59.9,
        # Qwen2.5-7B / Qwen2.5-VL-7B
        (3584, 3584, 128): 62.0,
        (4608, 3584, 128): 66.6,
        (37888, 3584, 128): 77.3,
        (3584, 18944, 128): 76.1,
        (152064, 3584, 128): 79.1,
        # LLaMA-3.1-8B
        (4096, 4096, 128): 66.3,
        (6144, 4096, 128): 62.0,
        (28672, 4096, 128): 77.2,
        (4096, 14336, 128): 76.7,
        # LLaMA2-7B
        (11008, 4096, 128): 74.5,
        (12288, 4096, 128): 75.0,
        (22016, 4096, 128): 75.5,
        (4096, 11008, 128): 71.4,
        # Additional
        (2560, 4096, 128): 54.8,
        (19456, 9728, 128): 74.3,
    }
    PEAK_BW_890M = 80.1  # GiB/s measured via copy benchmark (86 GB/s)

    # --- Runtime GPU detection ---
    gpu_name = torch.cuda.get_device_name(0)
    if "8060" in gpu_name:
        BEST_KNOWN_PERF = BEST_KNOWN_PERF_8060S
        DEFAULT_PEAK_BW = PEAK_BW_8060S
        gpu_profile = "Radeon 8060S"
    elif "890M" in gpu_name:
        BEST_KNOWN_PERF = BEST_KNOWN_PERF_890M
        DEFAULT_PEAK_BW = PEAK_BW_890M
        gpu_profile = "Radeon 890M"
    else:
        # Unknown GPU: no reference values, use conservative peak BW
        BEST_KNOWN_PERF = {}
        DEFAULT_PEAK_BW = 93.1  # ~100 GB/s in GiB/s
        gpu_profile = f"Unknown ({gpu_name})"

    PEAK_BW = args.peak_bw if args.peak_bw is not None else DEFAULT_PEAK_BW
    M = args.M
    print(f"GPU: {gpu_name}")
    print(f"Performance profile: {gpu_profile}")
    print(
        f"Peak bandwidth: {PEAK_BW} GiB/s"
        f"{' (auto)' if args.peak_bw is None else ' (override)'}"
    )
    if M > 1:
        print(f"M={M} (GEMM path)")
    else:
        print(f"M={M} (GEMV path)")

    # Bandwidth is reported in GiB/s (1 GiB = 1024^3 bytes)
    # Given time in ms: GiB/s = bytes / (ms * 1e-3) / 1024^3
    GIB_DIVISOR = 1024**3 / 1000  # multiply ms by this to get GiB denominator

    def calculate_bytes(K, N, group_size, M=1):
        """Calculate bytes moved for bandwidth calculation."""
        num_groups = K // group_size
        # input[M,K] + qweight[K,N//8] + qzeros[K//G,N//8] + scales[K//G,N] + output[M,N]
        return (
            M * K * 2
            + K * (N // 8) * 4
            + num_groups * (N // 8) * 4
            + num_groups * N * 2
            + M * N * 2
        )

    def run_correctness_test(shapes):
        """Run correctness tests for all shapes with multiple input data sets.

        Uses multiple test data patterns to catch different types of bugs:
        1. Random data: catches general numerical issues
        2. All ones: catches missing contributions (e.g., loop bounds bugs)
        3. Increasing: catches ordering/indexing issues
        """
        print("\n" + "=" * 100)
        print(f"CORRECTNESS TEST (multiple input patterns, M={M})")
        print("=" * 100)
        print(
            f"{'N':>6} x {'K':<6} | {'G':>4} | {'Pattern':<12} | {'Config':<20} | {'Max Diff':>10} | {'Rel Err':>10} | {'Status':>8}"
        )
        print("-" * 100)

        def pack_awq_weights(values_per_col, K, N):
            """Pack weight values into AWQ format.
            values_per_col: function(k, n) -> int4 value for weight[k, n]
            """
            qweight = torch.zeros((K, N // 8), dtype=torch.int32, device="cuda")
            for k in range(K):
                for n_pack in range(N // 8):
                    packed = 0
                    for i in range(8):
                        n = n_pack * 8 + i
                        val = values_per_col(k, n) & 0xF
                        # AWQ packing: shift = (i // 2) * 4 + (i % 2) * 16
                        shift = (i // 2) * 4 + (i % 2) * 16
                        packed |= val << shift
                    qweight[k, n_pack] = packed
            return qweight

        # Test data patterns
        test_patterns = [
            ("random", None),  # Random weights and activations
            ("ones", None),  # All ones - catches missing row contributions
            ("increasing", None),  # Increasing values - catches ordering bugs
        ]

        all_pass = True
        for N, K, group_size in shapes:
            if K % group_size != 0:
                continue

            num_groups = K // group_size
            split_k, block_n, num_warps = _choose_optimal_config(K, N, group_size)
            config = f"sk={split_k},N={block_n}"

            for pattern_name, _ in test_patterns:
                torch.manual_seed(42)

                if pattern_name == "random":
                    # Random data - existing test
                    input_tensor = (
                        torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.1
                    )
                    qweight = torch.randint(
                        0, 2**31, (K, N // 8), dtype=torch.int32, device="cuda"
                    )
                    qzeros = torch.randint(
                        0, 2**31, (num_groups, N // 8), dtype=torch.int32, device="cuda"
                    )
                    scales = (
                        torch.randn(num_groups, N, dtype=torch.float16, device="cuda")
                        * 0.01
                    )

                elif pattern_name == "ones":
                    # All ones - catches missing contributions
                    import struct

                    input_tensor = torch.ones(M, K, dtype=torch.float16, device="cuda")
                    # Pack all weights as 1 (simple value)
                    packed_val = 0
                    for i in range(8):
                        shift = (i // 2) * 4 + (i % 2) * 16
                        packed_val |= 1 << shift
                    # Convert to signed int32 for PyTorch
                    packed_val = struct.unpack(
                        "i", struct.pack("I", packed_val & 0xFFFFFFFF)
                    )[0]
                    qweight = torch.full(
                        (K, N // 8), packed_val, dtype=torch.int32, device="cuda"
                    )
                    # Zeros = 0, so dequant = (1 - 0) * scale = 1 * scale
                    qzeros = torch.zeros(
                        (num_groups, N // 8), dtype=torch.int32, device="cuda"
                    )
                    scales = (
                        torch.ones((num_groups, N), dtype=torch.float16, device="cuda")
                        * 0.01
                    )
                    # Expected output per column: K * 1 * 0.01 = K * 0.01

                elif pattern_name == "increasing":
                    # Increasing activation values - catches ordering issues
                    import struct

                    row = torch.arange(K, dtype=torch.float16, device="cuda") / K
                    input_tensor = row.unsqueeze(0).expand(M, -1).contiguous()
                    # Weights = 1 for all
                    packed_val = 0
                    for i in range(8):
                        shift = (i // 2) * 4 + (i % 2) * 16
                        packed_val |= 1 << shift
                    packed_val = struct.unpack(
                        "i", struct.pack("I", packed_val & 0xFFFFFFFF)
                    )[0]
                    qweight = torch.full(
                        (K, N // 8), packed_val, dtype=torch.int32, device="cuda"
                    )
                    qzeros = torch.zeros(
                        (num_groups, N // 8), dtype=torch.int32, device="cuda"
                    )
                    scales = torch.ones(
                        (num_groups, N), dtype=torch.float16, device="cuda"
                    )

                # Reference using dequantize + matmul
                dequantized = awq_dequantize_triton(qweight, scales, qzeros)
                output_ref = torch.matmul(input_tensor, dequantized)

                # Test via production path
                output = _awq_gemm_triton(
                    input_tensor, qweight, scales, qzeros, split_k_iters=8
                )

                diff = (output[:, :N] - output_ref[:, :N]).abs()
                max_diff = diff.max().item()
                ref_max = output_ref[:, :N].abs().max().item()
                rel_err = max_diff / (ref_max + 1e-6)

                # Use relative error for pass/fail
                # 5% tolerance to allow for fp16 precision loss while catching real bugs
                # (Real bugs like loop issues cause 10-60%+ error)
                passed = rel_err < 0.05
                all_pass = all_pass and passed
                status = "PASS" if passed else "FAIL"

                print(
                    f"{N:>6} x {K:<6} | {group_size:>4} | {pattern_name:<12} | {config:<20} | {max_diff:>10.4f} | {rel_err:>10.2%} | {status:>8}"
                )

        print("-" * 100)
        print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
        return all_pass

    def run_hip_kernel_correctness_test(shapes):
        """Run correctness tests specifically for the HIP GEMV kernel."""
        if M > 1:
            print(
                "\nSkipping HIP kernel correctness test (M>1, HIP GEMV only supports M=1)"
            )
            return True

        from vllm.platforms import current_platform

        if not current_platform.is_rocm():
            print("\nSkipping HIP kernel correctness test (not on ROCm)")
            return True

        try:
            from vllm._custom_ops import awq_gemv_hip
        except (ImportError, AttributeError):
            print("\nSkipping HIP kernel correctness test (awq_gemv_hip not available)")
            return True

        print("\n" + "=" * 90)
        print("HIP GEMV KERNEL CORRECTNESS TEST")
        print("=" * 90)
        print(
            f"{'N':>6} x {'K':<6} | {'G':>4} | {'Groups':>6} | {'Kernel':<12} | {'Max Diff':>10} | {'Rel Err':>10} | {'Status':>8}"
        )
        print("-" * 90)

        all_pass = True
        for N, K, group_size in shapes:
            if K % group_size != 0:
                continue
            if group_size != 128:  # HIP kernel only supports group_size=128
                continue
            if N % 8 != 0:  # HIP kernel requires N divisible by 8
                continue

            num_groups = K // group_size

            # Determine which kernel variant will be used
            can_sk8 = num_groups % 8 == 0
            can_sk4 = num_groups % 4 == 0
            can_sk2 = num_groups % 2 == 0

            if can_sk8 and N <= 8192:
                kernel_type = "split-k=8"
            elif can_sk4 and N <= 12288:
                kernel_type = "split-k=4"
            elif can_sk2 and N <= 16384:
                kernel_type = "split-k=2"
            else:
                kernel_type = "no-split-k"

            torch.manual_seed(42)
            activation = torch.randn(K, dtype=torch.float16, device="cuda")
            qweight = torch.randint(
                0, 2**31, (K, N // 8), dtype=torch.int32, device="cuda"
            )
            qzeros = torch.randint(
                0, 2**31, (num_groups, N // 8), dtype=torch.int32, device="cuda"
            )
            scales = (
                torch.randn(num_groups, N, dtype=torch.float16, device="cuda") * 0.01
            )

            # Reference: Triton dequantize + manual dot product
            dequantized = awq_dequantize_triton(qweight, scales, qzeros)
            output_ref = torch.matmul(activation.unsqueeze(0), dequantized).squeeze(0)

            # HIP kernel
            try:
                output_hip = awq_gemv_hip(activation, qweight, scales, qzeros)
            except Exception as e:
                print(
                    f"{N:>6} x {K:<6} | {group_size:>4} | {num_groups:>6} | {kernel_type:<12} | ERROR: {e}"
                )
                all_pass = False
                continue

            diff = (output_hip - output_ref).abs()
            max_diff = diff.max().item()
            # Relative error: diff / max(|ref|, 1e-6)
            rel_err = (diff / (output_ref.abs().max() + 1e-6)).max().item()

            # Stricter tolerance for correctness: max 1% relative error
            passed = rel_err < 0.01
            all_pass = all_pass and passed
            status = "PASS" if passed else "FAIL"

            print(
                f"{N:>6} x {K:<6} | {group_size:>4} | {num_groups:>6} | {kernel_type:<12} | {max_diff:>10.4f} | {rel_err:>10.4%} | {status:>8}"
            )

        print("-" * 90)
        print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
        return all_pass

    def apply_hip_preprocessing(qweight, qzeros, scales, group_size, N):
        """Apply the same preprocessing as AWQLinearMethod.process_weights_after_loading.

        Uses the shared compute_awq_padding_for_rocm function from awq.py to ensure
        padding logic stays in sync.
        """
        from vllm.model_executor.layers.quantization.awq import (
            compute_awq_padding_for_rocm,
        )
        from vllm.platforms import current_platform

        if not current_platform.is_rocm():
            return qweight, qzeros, scales, qweight.shape[0]

        K = qweight.shape[0]
        num_groups = qzeros.shape[0]

        if group_size != 128:
            return qweight, qzeros, scales, K

        should_pad, padded_groups = compute_awq_padding_for_rocm(
            num_groups, N, group_size
        )

        if not should_pad or padded_groups <= num_groups:
            return qweight, qzeros, scales, K

        pad_groups = padded_groups - num_groups
        padded_K = K + pad_groups * group_size

        # Pad tensors
        qweight_padded = torch.zeros(
            (padded_K, qweight.shape[1]), dtype=qweight.dtype, device=qweight.device
        )
        qweight_padded[:K] = qweight

        qzeros_padded = torch.zeros(
            (padded_groups, qzeros.shape[1]), dtype=qzeros.dtype, device=qzeros.device
        )
        qzeros_padded[:num_groups] = qzeros

        scales_padded = torch.zeros(
            (padded_groups, scales.shape[1]), dtype=scales.dtype, device=scales.device
        )
        scales_padded[:num_groups] = scales

        return qweight_padded, qzeros_padded, scales_padded, padded_K

    def bench(fn, warmup=50, rep=200):
        """Benchmark using triton.testing.do_bench with median return mode."""
        return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")

    def run_benchmark(shapes, include_autoawq=False, autoawq_module=None):
        """Run benchmarks for all shapes."""
        # GPU idle check moved to start of main() to avoid false positives
        # from our own correctness tests

        print("\n" + "=" * 130)
        print(f"BENCHMARK RESULTS (with HIP preprocessing, M={M})")
        print("=" * 130)

        if include_autoawq:
            print(
                f"{'N':>6} x {'K':<6} | {'G':>4} | {'vLLM Triton':<20} | {'AutoAWQ CUDA':<20} | {'Speedup':>8}"
            )
            print(
                f"{'':>6}   {'':>6} | {'':>4} | {'GiB/s':>8} {'Time':>10} | {'GiB/s':>8} {'Time':>10} | {'':>8}"
            )
        else:
            print(
                f"{'N':>6} x {'K':<6} | {'G':>4} | {'Padded':>14} | {'Config':<22} | {'GiB/s':>10} | {'Time':>10} | {'% Peak':>10} | {'vs Ref':>10}"
            )
        print("-" * 130)

        regressions = []

        results = []
        for N, K, group_size in shapes:
            if K % group_size != 0:
                continue

            # AutoAWQ only supports group_size 64 and 128, and requires minimum K
            min_k = 1024 if group_size == 128 else 512
            can_run_autoawq = (
                include_autoawq
                and group_size in [64, 128]
                and N % 4 == 0
                and min_k <= K
            )

            num_groups = K // group_size
            bytes_moved = calculate_bytes(K, N, group_size, M)
            split_k, block_n, num_warps = _choose_optimal_config(K, N, group_size)

            qweight_orig = torch.randint(
                0, 2**31, (K, N // 8), dtype=torch.int32, device="cuda"
            )
            qzeros_orig = torch.randint(
                0, 2**31, (num_groups, N // 8), dtype=torch.int32, device="cuda"
            )
            scales_orig = (
                torch.randn(num_groups, N, dtype=torch.float16, device="cuda") * 0.01
            )

            # Apply HIP preprocessing (simulates process_weights_after_loading)
            qweight, qzeros, scales, padded_K = apply_hip_preprocessing(
                qweight_orig, qzeros_orig, scales_orig, group_size, N
            )
            is_padded = padded_K > K

            # Create input tensor (potentially padded)
            input_tensor = torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.01
            if is_padded:
                input_padded = torch.zeros(
                    M, padded_K, dtype=torch.float16, device="cuda"
                )
                input_padded[:, :K] = input_tensor
                input_for_bench = input_padded
            else:
                input_for_bench = input_tensor

            # vLLM Triton benchmark
            def run_triton():
                return _awq_gemm_triton(
                    input_for_bench, qweight, scales, qzeros, split_k_iters=8
                )

            ms_triton = bench(run_triton, warmup=args.warmup, rep=args.rep)
            bw_triton = bytes_moved / (ms_triton * GIB_DIVISOR)
            pct_peak = bw_triton / PEAK_BW * 100

            if can_run_autoawq:
                # Create AutoAWQ format tensors
                kernel_awq, zeros_awq, scales_awq = convert_vllm_to_autoawq_format(
                    qweight_orig, qzeros_orig, scales_orig, group_size
                )

                def run_autoawq():
                    return autoawq_module.gemv_forward(
                        input_tensor, kernel_awq, scales_awq, zeros_awq, group_size
                    )

                ms_autoawq = bench(run_autoawq, warmup=args.warmup, rep=args.rep)
                bw_autoawq = bytes_moved / (ms_autoawq * GIB_DIVISOR)
                speedup = ms_autoawq / ms_triton  # >1 means Triton is faster

                print(
                    f"{N:>6} x {K:<6} | {group_size:>4} | {bw_triton:>8.1f} {ms_triton * 1000:>8.0f} us | {bw_autoawq:>8.1f} {ms_autoawq * 1000:>8.0f} us | {speedup:>7.2f}x"
                )
                results.append(
                    (K, N, group_size, bw_triton, ms_triton, bw_autoawq, ms_autoawq)
                )
            elif include_autoawq:
                min_k = 1024 if group_size == 128 else 512
                if group_size not in [64, 128]:
                    reason = "g!=64,128"
                elif min_k > K:
                    reason = f"K<{min_k}"
                else:
                    reason = "N%4!=0"
                print(
                    f"{N:>6} x {K:<6} | {group_size:>4} | {bw_triton:>8.1f} {ms_triton * 1000:>8.0f} us | {'N/A (' + reason + ')':>20} | {'N/A':>8}"
                )
                results.append((K, N, group_size, bw_triton, ms_triton, None, None))
            else:
                config = f"N={block_n},w={num_warps},sk={split_k}"
                padded_str = f"{K}->{padded_K}" if is_padded else "-"

                # Compare against reference performance
                shape_key = (N, K, group_size)
                ref_perf = BEST_KNOWN_PERF.get(shape_key)
                if ref_perf is not None:
                    ratio = bw_triton / ref_perf
                    threshold = 1.0 - PERF_TOLERANCE
                    if ratio < threshold:
                        # Regression detected
                        vs_ref = f"SLOW {ratio * 100:.0f}%"
                        regressions.append(
                            (N, K, group_size, bw_triton, ref_perf, ratio)
                        )
                    elif ratio > 1.0 + PERF_TOLERANCE:
                        # Improvement - update reference!
                        vs_ref = f"NEW! {ratio * 100:.0f}%"
                    else:
                        vs_ref = "OK"
                else:
                    vs_ref = "no ref"

                print(
                    f"{N:>6} x {K:<6} | {group_size:>4} | {padded_str:>14} | {config:<22} | {bw_triton:>10.1f} | {ms_triton * 1000:>8.0f} us | {pct_peak:>9.1f}% | {vs_ref:>10}"
                )
                results.append((K, N, group_size, bw_triton, ms_triton, config))

        print("-" * 130)

        # Print regression summary if any
        if regressions:
            print("\n⚠️  PERFORMANCE REGRESSIONS DETECTED:")
            print("-" * 70)
            for N, K, G, actual, expected, ratio in regressions:
                print(
                    f"  {N}x{K} G={G}: {actual:.1f} GiB/s vs {expected:.1f} GiB/s expected ({ratio * 100:.0f}%)"
                )
            print("-" * 70)
            print(
                f"  Tolerance: {PERF_TOLERANCE * 100:.0f}% (values below {100 - PERF_TOLERANCE * 100:.0f}% of reference trigger warning)"
            )

        return results

    def run_exhaustive_search(shapes):
        """Run exhaustive search for optimal configurations."""
        if M > 1:
            print(
                "\nExhaustive GEMV search only supports M=1. Use --gemm-tuning for M>1."
            )
            return

        print("\n" + "=" * 110)
        print("EXHAUSTIVE SEARCH FOR OPTIMAL CONFIGURATIONS")
        print("=" * 110)

        for N, K, group_size in shapes:
            if K % group_size != 0:
                continue

            num_groups = K // group_size
            valid_sk = _get_valid_split_k_values(K, group_size)
            bytes_moved = calculate_bytes(K, N, group_size, M)

            print(f"\nShape N={N}, K={K}, group_size={group_size}")
            print(f"  num_groups={num_groups}, valid_split_k={valid_sk}")

            input_tensor = torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.01
            qweight = torch.randint(
                0, 2**31, (K, N // 8), dtype=torch.int32, device="cuda"
            )
            qzeros = torch.randint(
                0, 2**31, (num_groups, N // 8), dtype=torch.int32, device="cuda"
            )
            scales = (
                torch.randn(num_groups, N, dtype=torch.float16, device="cuda") * 0.01
            )

            dequantized = awq_dequantize_triton(qweight, scales, qzeros)
            output_ref = torch.matmul(input_tensor, dequantized)

            best_bw, best_config = 0, ""

            for split_k in valid_sk:
                for block_n in [64, 128, 256]:
                    if N % block_n != 0:
                        continue
                    for num_warps in [1, 2, 4, 8]:
                        partial = torch.zeros(
                            split_k, N, dtype=torch.float16, device="cuda"
                        )
                        result = torch.zeros(1, N, dtype=torch.float16, device="cuda")

                        def run():
                            awq_gemv_kernel_split_k[(triton.cdiv(N, block_n), split_k)](
                                input_tensor,
                                qweight,
                                qzeros,
                                scales,
                                partial,
                                K=K,
                                N=N,
                                GROUP_SIZE=group_size,
                                BLOCK_N=block_n,
                                SPLIT_K=split_k,
                                use_fp32_accumulator=False,
                                num_warps=num_warps,
                            )
                            reduce_split_k_kernel[(triton.cdiv(N, block_n),)](
                                partial,
                                result,
                                N=N,
                                SPLIT_K=split_k,
                                BLOCK_N=block_n,
                                num_warps=1,
                            )
                            return result

                        try:
                            out = run()
                            diff = (out[0, :N] - output_ref[0, :N]).abs().max().item()
                            if diff < 0.1:
                                ms = bench(run, warmup=args.warmup, rep=args.rep)
                                bw = bytes_moved / (ms * GIB_DIVISOR)
                                if bw > best_bw:
                                    best_bw = bw
                                    best_config = f"block_n={block_n}, warps={num_warps}, split_k={split_k}"
                        except Exception:
                            pass

            print(
                f"  BEST: {best_bw:.1f} GiB/s ({best_bw / PEAK_BW * 100:.1f}%), {best_config}"
            )

    def calculate_gemm_bytes(M, K, N, group_size):
        """Calculate bytes moved for GEMM bandwidth calculation."""
        num_groups = K // group_size
        # input[M,K] + qweight[K,N//8] + qzeros[K//G,N//8] + scales[K//G,N] + output[M,N]
        return (
            M * K * 2  # input fp16
            + K * (N // 8) * 4  # qweight int32
            + num_groups * (N // 8) * 4  # qzeros int32
            + num_groups * N * 2  # scales fp16
            + M * N * 2
        )  # output fp16

    def run_gemm_tuning(shapes, quick=False):
        """Run exhaustive GEMM tuning for M=128 and M=256."""
        print("\n" + "=" * 130)
        print(f"GEMM TUNING FOR STRIX HALO {'(QUICK MODE)' if quick else '(FULL)'}")
        print("=" * 130)
        print("""
Tuning parameters from MI300 commit 1178c7f18:
  - block_size_m=16, block_size_n=64, block_size_k=64
  - num_stages=1, num_warps=4
  - split_k_iters from caller (typically 8)

Strix Halo differences:
  - 16 CUs (vs 304 on MI300)
  - ~238 GiB/s LPDDR5X (vs ~4.9 TiB/s HBM)
  - 64 MB L3 cache shared CPU/GPU
  - Higher memory latency -> may benefit from more pipeline stages
""")

        if quick:
            # Quick mode: fewer shapes, fewer configs
            M_VALUES = [128]
            BLOCK_M_VALUES = [16, 32]
            BLOCK_N_VALUES = [64, 128]
            BLOCK_K_VALUES = [64, 128]
            SPLIT_K_VALUES = [1, 4]
            NUM_STAGES_VALUES = [1, 2]
            NUM_WARPS_VALUES = [4]
            # Use subset of shapes
            shapes = [s for s in shapes if s[1] >= 4096][
                :4
            ]  # Only larger N, max 4 shapes
        else:
            M_VALUES = [128, 256]
            # Tuning space for Strix Halo
            BLOCK_M_VALUES = [8, 16, 32, 64]
            BLOCK_N_VALUES = [32, 64, 128]
            BLOCK_K_VALUES = [32, 64, 128]
            SPLIT_K_VALUES = [1, 2, 4, 8]
            NUM_STAGES_VALUES = [1, 2, 3]
            NUM_WARPS_VALUES = [2, 4, 8]

        for M in M_VALUES:
            print(f"\n{'=' * 130}")
            print(f"M = {M}")
            print(f"{'=' * 130}")
            print(
                f"{'N':>6} x {'K':<6} | {'Config':<50} | {'Time':>10} | {'TFLOPS':>8} | {'GiB/s':>10} | {'% Peak':>8}"
            )
            print("-" * 130)

            for N, K, group_size in shapes:
                if K % group_size != 0:
                    continue

                num_groups = K // group_size
                bytes_moved = calculate_gemm_bytes(M, K, N, group_size)
                # 2*M*N*K FLOPs for matmul
                flops = 2 * M * N * K

                input_tensor = (
                    torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.01
                )
                qweight = torch.randint(
                    0, 2**31, (K, N // 8), dtype=torch.int32, device="cuda"
                )
                qzeros = torch.randint(
                    0, 2**31, (num_groups, N // 8), dtype=torch.int32, device="cuda"
                )
                scales = (
                    torch.randn(num_groups, N, dtype=torch.float16, device="cuda")
                    * 0.01
                )

                # Reference for correctness
                dequantized = awq_dequantize_triton(qweight, scales, qzeros)
                output_ref = torch.matmul(input_tensor, dequantized)

                best_time = float("inf")
                best_config = ""
                best_correct = False

                for block_m in BLOCK_M_VALUES:
                    for block_n in BLOCK_N_VALUES:
                        if N % block_n != 0:
                            continue
                        for block_k in BLOCK_K_VALUES:
                            for split_k in SPLIT_K_VALUES:
                                # Check split_k validity
                                if K % (block_k * split_k) != 0:
                                    continue
                                num_k_tiles = K // (block_k * split_k)
                                if num_k_tiles < 1:
                                    continue
                                # Check group alignment
                                if (
                                    block_k * split_k
                                ) % group_size != 0 and group_size % (
                                    block_k * split_k
                                ) != 0:
                                    continue

                                for num_stages in NUM_STAGES_VALUES:
                                    for num_warps in NUM_WARPS_VALUES:
                                        try:
                                            result = torch.zeros(
                                                (split_k, M, N),
                                                dtype=scales.dtype,
                                                device="cuda",
                                            )
                                            k_group = K // group_size

                                            grid = (
                                                triton.cdiv(M, block_m)
                                                * triton.cdiv(N, block_n),
                                                split_k,
                                            )

                                            def run():
                                                result.zero_()
                                                awq_gemm_kernel[grid](
                                                    input_tensor,
                                                    qweight,
                                                    result,
                                                    qzeros,
                                                    scales,
                                                    M,
                                                    N,
                                                    K,
                                                    group_size,
                                                    BLOCK_SIZE_M=block_m,
                                                    BLOCK_SIZE_N=block_n,
                                                    BLOCK_SIZE_K=block_k,
                                                    SPLIT_K=split_k,
                                                    NUM_K_TILES=num_k_tiles,
                                                    K_GROUPS=k_group,
                                                    num_stages=num_stages,
                                                    num_warps=num_warps,
                                                )
                                                return result.sum(0)

                                            # Quick correctness check
                                            out = run()
                                            diff = (out - output_ref).abs().max().item()
                                            if (
                                                diff > 0.5
                                            ):  # Allow slightly larger tolerance for GEMM
                                                continue

                                            # Benchmark
                                            ms = bench(
                                                run,
                                                warmup=args.warmup,
                                                rep=args.rep,
                                            )
                                            if ms < best_time:
                                                best_time = ms
                                                best_config = f"m={block_m},n={block_n},k={block_k},sk={split_k},st={num_stages},w={num_warps}"
                                                best_correct = True
                                        except Exception:
                                            # Skip invalid configs
                                            pass

                if best_correct:
                    bw = bytes_moved / (best_time * GIB_DIVISOR)
                    tflops = flops / (best_time * 1e9)
                    pct_peak = bw / PEAK_BW * 100
                    print(
                        f"{N:>6} x {K:<6} | {best_config:<50} | {best_time * 1000:>8.0f} us | {tflops:>7.2f} | {bw:>10.1f} | {pct_peak:>7.1f}%"
                    )
                else:
                    print(f"{N:>6} x {K:<6} | NO VALID CONFIG FOUND")

        # Print summary of best configs
        print("\n" + "=" * 130)
        print("RECOMMENDED STRIX HALO TUNING PARAMETERS")
        print("=" * 130)
        print("""
Based on benchmarking, consider these changes for Strix Halo:

1. If smaller block sizes (e.g., m=8, n=32, k=32) are consistently faster:
   - Strix has fewer CUs, so smaller tiles reduce wasted work
   
2. If num_stages > 1 helps:
   - LPDDR5X has higher latency than HBM, so software pipelining helps
   
3. If split_k=1 or 2 is better than 8:
   - Fewer CUs means less benefit from k-parallelism
   - Reduction overhead may outweigh benefits
""")

    # Load AutoAWQ kernel if requested
    autoawq_module = None
    if args.autoawq:
        print("Loading AutoAWQ CUDA kernel (compiled with HIP)...")
        autoawq_module = load_autoawq_kernel()
        print(f"AutoAWQ module loaded: {autoawq_module}")

        # Run AutoAWQ correctness test first
        autoawq_correct = test_autoawq_correctness(
            autoawq_module, SHAPES, awq_dequantize_triton
        )
        if not autoawq_correct:
            print(
                "\nWARNING: AutoAWQ correctness test failed! Benchmark results may not be valid."
            )

    # Run tests
    if args.gemm_quick:
        run_gemm_tuning(SHAPES, quick=True)
    elif args.gemm_tuning:
        run_gemm_tuning(SHAPES)
    elif args.exhaustive:
        run_exhaustive_search(SHAPES)
    elif args.correctness_only:
        run_correctness_test(SHAPES)
    elif args.benchmark_only:
        run_benchmark(
            SHAPES, include_autoawq=args.autoawq, autoawq_module=autoawq_module
        )
    else:
        # Run both
        passed = run_correctness_test(SHAPES)
        if passed:
            run_benchmark(
                SHAPES, include_autoawq=args.autoawq, autoawq_module=autoawq_module
            )
        else:
            print("\nSkipping benchmark due to correctness failures")

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    # Print GPU utilization status (checked at start of run)
    if gpu_error:
        print(f"Note: Could not check GPU utilization ({gpu_error})")
    elif gpu_idle:
        print(f"✓ GPU was idle at start (utilization: {gpu_utilization:.1f}%)")
    else:
        print(f"⚠️  WARNING: GPU was busy at start! Utilization: {gpu_utilization:.1f}%")
        print("   Benchmark results may be inaccurate.")

    print(f"""
GPU: {gpu_name}
Performance profile: {gpu_profile}
Peak memory bandwidth: {PEAK_BW} GiB/s
""")

    if args.autoawq:
        print("""
AutoAWQ vs vLLM Triton Comparison:
- AutoAWQ uses warp-based K-reduction (each warp handles one output channel)
- vLLM Triton uses N-split with split-K parallelization
- Speedup < 1.0: AutoAWQ is faster (small shapes benefit from warp reduction)
- Speedup > 1.0: vLLM Triton is faster (large shapes benefit from N-parallelism)

Key findings:
- Small K (< 2048): AutoAWQ's warp reduction is more efficient
- Large K and N: Triton's split-K approach scales better
- The crossover point depends on K, N, and group_size
""")


if __name__ == "__main__":
    main()
