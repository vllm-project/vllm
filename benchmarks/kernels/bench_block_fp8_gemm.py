# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

# Disable DeepGEMM for this benchmark to use CUTLASS
os.environ["VLLM_USE_DEEP_GEMM"] = "0"

import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    W8A8BlockFp8LinearOp,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    CUTLASS_BLOCK_FP8_SUPPORTED,
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


def build_w8a8_block_fp8_runner(M, N, K, block_size, device, use_cutlass):
    """Build runner function for w8a8 block fp8 matmul."""
    factor_for_scale = 1e-2

    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    # Create random input tensor (bfloat16, will be quantized by W8A8BlockFp8LinearOp)
    A_ref = (torch.rand(M, K, dtype=torch.bfloat16, device=device) - 0.5) * 2 * fp8_max

    # Create quantized weight tensor
    B_ref = (torch.rand(N, K, dtype=torch.bfloat16, device=device) - 0.5) * 2 * fp8_max
    B = B_ref.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    # Create weight scales
    block_n, block_k = block_size[0], block_size[1]
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k

    Bs = (
        torch.rand(n_tiles, k_tiles, dtype=torch.float32, device=device)
        * factor_for_scale
    )

    # Create W8A8BlockFp8LinearOp instance
    weight_group_shape = GroupShape(block_n, block_k)
    act_quant_group_shape = GroupShape(1, block_k)  # Per-token, per-group quantization

    linear_op = W8A8BlockFp8LinearOp(
        weight_group_shape=weight_group_shape,
        act_quant_group_shape=act_quant_group_shape,
        cutlass_block_fp8_supported=use_cutlass,
        use_aiter_and_is_supported=False,
    )

    def run():
        return linear_op.apply(
            input=A_ref,
            weight=B,
            weight_scale=Bs,
            input_scale=None,
            bias=None,
        )

    return run


# Determine available providers
available_providers = ["torch-bf16", "w8a8-block-fp8-triton"]
plot_title = "BF16 vs W8A8 Block FP8 GEMMs"

if CUTLASS_BLOCK_FP8_SUPPORTED:
    available_providers.append("w8a8-block-fp8-cutlass")


@vllm_triton.testing.perf_report(
    vllm_triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[1, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        x_log=False,
        line_arg="provider",
        line_vals=available_providers,
        line_names=available_providers,
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
    elif provider == "w8a8-block-fp8-triton":
        run_w8a8_triton = build_w8a8_block_fp8_runner(
            M, N, K, block_size, device, use_cutlass=False
        )
        ms, min_ms, max_ms = vllm_triton.testing.do_bench_cudagraph(
            lambda: run_w8a8_triton(), quantiles=quantiles
        )
    elif provider == "w8a8-block-fp8-cutlass":
        run_w8a8_cutlass = build_w8a8_block_fp8_runner(
            M, N, K, block_size, device, use_cutlass=True
        )
        ms, min_ms, max_ms = vllm_triton.testing.do_bench_cudagraph(
            lambda: run_w8a8_cutlass(), quantiles=quantiles
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

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
