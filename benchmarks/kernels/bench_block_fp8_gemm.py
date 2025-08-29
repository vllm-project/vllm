# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    w8a8_block_fp8_matmul,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton as vllm_triton

assert current_platform.is_cuda(), (
    "Only support benchmarking w8a8 block fp8 kernel on CUDA device."
)

# DeepSeek-V3 weight shapes
DEEPSEEK_V3_SHAPES = [
    (512 + 64, 7168),
    (2112, 7168),
    ((128 + 64) * 128, 7168),
    (128 * (128 + 128), 512),
    (7168, 16384),
    (7168, 18432),
    (18432 * 2, 7168),
    (24576, 1536),
    (12288, 7168),
    (4096, 7168),
    (7168, 2048),
]


def build_w8a8_block_fp8_runner(M, N, K, block_size, device):
    """Build runner function for w8a8 block fp8 matmul."""
    factor_for_scale = 1e-2

    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    # Create random FP8 tensors
    A_fp32 = (torch.rand(M, K, dtype=torch.float32, device=device) - 0.5) * 2 * fp8_max
    A = A_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    B_fp32 = (torch.rand(N, K, dtype=torch.float32, device=device) - 0.5) * 2 * fp8_max
    B = B_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    # Create scales
    block_n, block_k = block_size[0], block_size[1]
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k

    As = torch.rand(M, k_tiles, dtype=torch.float32, device=device) * factor_for_scale
    Bs = (
        torch.rand(n_tiles, k_tiles, dtype=torch.float32, device=device)
        * factor_for_scale
    )

    def run():
        return w8a8_block_fp8_matmul(A, B, As, Bs, block_size, torch.bfloat16)

    return run


@vllm_triton.testing.perf_report(
    vllm_triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[1, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        x_log=False,
        line_arg="provider",
        line_vals=["torch-bf16", "w8a8-block-fp8"],
        line_names=["torch-bf16", "w8a8-block-fp8"],
        ylabel="TFLOP/s (larger is better)",
        plot_name="BF16 vs W8A8 Block FP8 GEMMs",
        args={},
    )
)
def benchmark_tflops(batch_size, provider, N, K, block_size=(128, 128)):
    M = batch_size
    device = "cuda"

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch-bf16":
        a = torch.randn((M, K), device=device, dtype=torch.bfloat16)
        b = torch.randn((N, K), device=device, dtype=torch.bfloat16)
        ms, min_ms, max_ms = vllm_triton.testing.do_bench_cudagraph(
            lambda: torch.nn.functional.linear(a, b), quantiles=quantiles
        )
    else:  # w8a8-block-fp8
        run_w8a8 = build_w8a8_block_fp8_runner(M, N, K, block_size, device)
        ms, min_ms, max_ms = vllm_triton.testing.do_bench_cudagraph(
            lambda: run_w8a8(), quantiles=quantiles
        )

    to_tflops = lambda t_ms: (2 * M * N * K) * 1e-12 / (t_ms * 1e-3)
    return to_tflops(ms), to_tflops(max_ms), to_tflops(min_ms)


if __name__ == "__main__":
    block_size = (128, 128)

    for N, K in DEEPSEEK_V3_SHAPES:
        print(f"\nBenchmarking DeepSeek-V3, N={N} K={K}")

        print(f"TFLOP/s comparison (block_size={block_size}):")
        benchmark_tflops.run(
            print_data=True,
            # show_plots=False,
            # save_path=f"bench_w8a8_block_fp8_tflops_n{N}_k{K}",
            N=N,
            K=K,
            block_size=block_size,
        )

    print("\nBenchmark finished!")
