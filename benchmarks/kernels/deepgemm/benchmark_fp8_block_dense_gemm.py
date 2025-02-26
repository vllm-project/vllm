# SPDX-License-Identifier: Apache-2.0
import time
from typing import Dict, Tuple

# Import DeepGEMM functions
import deep_gemm
import torch
from deep_gemm import calc_diff, cell_div, get_col_major_tma_aligned_tensor

# Import vLLM functions
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8, w8a8_block_fp8_matmul)


# Copied from
# https://github.com/deepseek-ai/DeepGEMM/blob/78cacf70d41d15d688bd493ebc85845f7f2a3d5d/tests/test_core.py#L9
def per_token_cast_to_fp8(
        x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert tensor to FP8 format with per-token scaling."""
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(
        torch.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)


# Copied from
# https://github.com/deepseek-ai/DeepGEMM/blob/78cacf70d41d15d688bd493ebc85845f7f2a3d5d/tests/test_core.py#L17
def per_block_cast_to_fp8(
        x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert tensor to FP8 format with per-block scaling."""
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((cell_div(m, 128) * 128, cell_div(n, 128) * 128),
                           dtype=x.dtype,
                           device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (
        x_amax / 448.0).view(x_view.size(0), x_view.size(2))


def benchmark_shape(m: int,
                    n: int,
                    k: int,
                    warmup: int = 10,
                    repeat: int = 1000) -> Dict[str, Dict[str, float]]:
    """Benchmark all implementations for a specific (m, n, k) shape."""
    print(f"\n=== Benchmarking shape: m={m}, n={n}, k={k} ===")

    # Create test tensors
    A = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    B = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)

    # Reference result in BF16
    torch.cuda.synchronize()
    C_ref = A @ B.t()

    # Pre-quantize B for all implementations
    # (weights can be pre-quantized offline)
    B_deepgemm, B_scale_deepgemm = per_block_cast_to_fp8(B)
    B_vllm, B_scale_vllm = per_block_cast_to_fp8(B)

    # Block size configuration
    block_size = [128, 128]

    results = {}

    # === DeepGEMM Implementation ===
    def deepgemm_gemm():
        # A quantization is inside the loop as it depends on activations
        A_deepgemm, A_scale_deepgemm = per_token_cast_to_fp8(A)
        A_scale_aligned = get_col_major_tma_aligned_tensor(A_scale_deepgemm)
        C_deepgemm = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
        deep_gemm.gemm_fp8_fp8_bf16_nt((A_deepgemm, A_scale_aligned),
                                       (B_deepgemm, B_scale_deepgemm),
                                       C_deepgemm)
        return C_deepgemm

    # === vLLM Triton Implementation ===
    def vllm_triton_gemm():
        # A quantization is inside the loop as it depends on activations
        A_vllm, A_scale_vllm = per_token_group_quant_fp8(A, block_size[1])
        return w8a8_block_fp8_matmul(A_vllm,
                                     B_vllm,
                                     A_scale_vllm,
                                     B_scale_vllm,
                                     block_size,
                                     output_dtype=torch.bfloat16)

    # === vLLM CUTLASS Implementation ===
    def vllm_cutlass_gemm():
        # A quantization is inside the loop as it depends on activations
        A_vllm_cutlass, A_scale_vllm_cutlass = per_token_group_quant_fp8(
            A, block_size[1], column_major_scales=True)
        return ops.cutlass_scaled_mm(A_vllm_cutlass,
                                     B_vllm.T,
                                     scale_a=A_scale_vllm_cutlass,
                                     scale_b=B_scale_vllm.T,
                                     out_dtype=torch.bfloat16)

    # Run correctness check first
    print("Running correctness check...")
    C_deepgemm = deepgemm_gemm()
    C_vllm_triton = vllm_triton_gemm()
    C_vllm_cutlass = vllm_cutlass_gemm()

    deepgemm_diff = calc_diff(C_deepgemm, C_ref)
    vllm_triton_diff = calc_diff(C_vllm_triton, C_ref)
    vllm_cutlass_diff = calc_diff(C_vllm_cutlass, C_ref)

    print(f"DeepGEMM vs Reference difference: {deepgemm_diff:.6f}")
    print(f"vLLM Triton vs Reference difference: {vllm_triton_diff:.6f}")
    print(f"vLLM CUTLASS vs Reference difference: {vllm_cutlass_diff:.6f}")
    print("vLLM Triton vs DeepGEMM difference: "
          f"{calc_diff(C_vllm_triton, C_deepgemm):.6f}")
    print("vLLM CUTLASS vs DeepGEMM difference: "
          f"{calc_diff(C_vllm_cutlass, C_deepgemm):.6f}")

    # Benchmark implementations
    implementations = {
        "DeepGEMM": deepgemm_gemm,
        "vLLM Triton": vllm_triton_gemm,
        "vLLM CUTLASS": vllm_cutlass_gemm
    }

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
        flops = 2 * m * n * k  # multiply-adds
        tflops = flops / (avg_time_ms * 1e-3) / 1e12

        results[name] = {
            "time_ms": avg_time_ms,
            "tflops": tflops,
            "diff": {
                "DeepGEMM":
                deepgemm_diff if name == "DeepGEMM" else calc_diff(
                    func(), C_deepgemm),
                "Reference":
                deepgemm_diff if name == "DeepGEMM" else
                (vllm_triton_diff
                 if name == "vLLM Triton" else vllm_cutlass_diff)
            }
        }

        print(f"{name}: {avg_time_ms:.3f} ms, {tflops:.2f} TFLOPS")

    # Calculate speedups
    baseline = results["DeepGEMM"]["time_ms"]
    for name, data in results.items():
        if name != "DeepGEMM":
            speedup = data["time_ms"] / baseline
            print(f"DeepGEMM is {speedup:.2f}x "
                  f"{'faster' if speedup > 1 else 'slower'} than {name}")

    vllm_triton_time = results["vLLM Triton"]["time_ms"]
    vllm_cutlass_time = results["vLLM CUTLASS"]["time_ms"]
    cutlass_vs_triton = vllm_triton_time / vllm_cutlass_time
    print(
        f"vLLM CUTLASS is {cutlass_vs_triton:.2f}x "
        f"{'faster' if cutlass_vs_triton > 1 else 'slower'} than vLLM Triton")

    return results


def run_benchmarks():
    """Run benchmarks for a set of common shapes."""
    print("===== STARTING FP8 GEMM BENCHMARK =====")

    # Make sure we're using the GPU
    if not torch.cuda.is_available():
        print("CUDA not available! Tests require GPU.")
        return

    print(f"Using device: {torch.cuda.get_device_name()}")

    # Enable TF32 for better performance
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Set seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Define benchmark shapes (m, n, k)
    shapes = [
        (8, 4096, 7168),  # Small batch
        (8, 7168, 18432),  # Small batch MLP up-proj
        (8, 18432, 7168),  # Small batch MLP down-proj
        (128, 4096, 7168),  # Typical batch
        (128, 7168, 18432),  # MLP up-projection
        (128, 18432, 7168),  # MLP down-projection
        (1024, 4096, 7168),  # Larger batch
        (1024, 18432, 7168),  # Larger batch with MLP down-proj
        (2048, 4096, 7168),  # Very large batch 
    ]

    all_results = {}
    for m, n, k in shapes:
        shape_key = f"m{m}_n{n}_k{k}"
        all_results[shape_key] = benchmark_shape(m, n, k)

    print("\n===== BENCHMARK SUMMARY =====")
    print("Matrix multiplication: C[m,n] = A[m,k] @ B[n,k].T")
    print("\nAverage speedups:")

    # Calculate average speedups across all shapes
    speedups = {
        "DeepGEMM vs vLLM Triton": [],
        "DeepGEMM vs vLLM CUTLASS": [],
        "vLLM CUTLASS vs vLLM Triton": []
    }

    for shape_key, results in all_results.items():
        deepgemm_time = results["DeepGEMM"]["time_ms"]
        vllm_triton_time = results["vLLM Triton"]["time_ms"]
        vllm_cutlass_time = results["vLLM CUTLASS"]["time_ms"]

        speedups["DeepGEMM vs vLLM Triton"].append(vllm_triton_time /
                                                   deepgemm_time)
        speedups["DeepGEMM vs vLLM CUTLASS"].append(vllm_cutlass_time /
                                                    deepgemm_time)
        speedups["vLLM CUTLASS vs vLLM Triton"].append(vllm_triton_time /
                                                       vllm_cutlass_time)

    for comparison, values in speedups.items():
        avg_speedup = sum(values) / len(values)
        print(f"{comparison}: {avg_speedup:.2f}x "
              f"{'faster' if avg_speedup > 1 else 'slower'}")

    print("\nAverage TFLOPS:")
    implementations = ["DeepGEMM", "vLLM Triton", "vLLM CUTLASS"]
    for impl in implementations:
        avg_tflops = sum(
            results[impl]["tflops"]
            for results in all_results.values()) / len(all_results)
        print(f"{impl}: {avg_tflops:.2f} TFLOPS")

    print("\nAverage accuracy difference vs reference:")
    for impl in implementations:
        avg_diff = sum(results[impl]["diff"]["Reference"]
                       for results in all_results.values()) / len(all_results)
        print(f"{impl}: {avg_diff:.6f}")


if __name__ == "__main__":
    run_benchmarks()
