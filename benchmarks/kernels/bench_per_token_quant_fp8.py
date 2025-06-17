# SPDX-License-Identifier: Apache-2.0
import itertools

import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.triton_utils import triton


def torch_per_token_quant_fp8(
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return ops.dynamic_per_token_quant_fp8(input)


def cuda_per_token_quant_fp8(
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = torch.empty((input.shape[0], 1), device=input.device, dtype=torch.float32)
    # For ROCm on MI300, the output fp8 dtype is torch.float_e3m3fnuz
    out_dtype: torch.dtype = current_platform.fp8_dtype()
    output = torch.empty(input.shape, device=input.device, dtype=out_dtype)
    scale_ub = None
    torch.ops._C.dynamic_per_token_scaled_fp8_quant(output, input, scale, scale_ub)
    return output, scale


def calculate_diff(batch_size: int, seq_len: int):
    """Calculate difference between Triton and CUDA implementations."""
    device = torch.device("cuda")
    x = torch.rand((batch_size * seq_len, 4096), dtype=torch.float16, device=device)

    torch_out, torch_scale = torch_per_token_quant_fp8(x)
    cuda_out, cuda_scale = cuda_per_token_quant_fp8(x)

    if torch.allclose(
        cuda_out.to(torch.float32), torch_out.to(torch.float32), rtol=1e-3, atol=1e-5
    ) and torch.allclose(cuda_scale, torch_scale, rtol=1e-3, atol=1e-5):
        print("✅ All implementations match")
    else:
        print("❌ Implementations differ")


batch_size_range = [1, 16, 32, 64, 128]
seq_len_range = [1, 16, 64, 128, 256, 512, 1024, 2048, 4096]

configs = list(itertools.product(batch_size_range, seq_len_range))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "seq_len"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["torch", "cuda"],
        line_names=["Torch", "CUDA"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="per-token-dynamic-quant-fp8-performance",
        args={},
    )
)
def benchmark_quantization(batch_size, seq_len, provider):
    dtype = torch.float16
    device = torch.device("cuda")

    x = torch.randn(batch_size * seq_len, 4096, device=device, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        fn = lambda: torch_per_token_quant_fp8(x.clone())
    elif provider == "cuda":
        fn = lambda: cuda_per_token_quant_fp8(x.clone())

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    calculate_diff(batch_size=4, seq_len=4096)
    benchmark_quantization.run(print_data=True)
