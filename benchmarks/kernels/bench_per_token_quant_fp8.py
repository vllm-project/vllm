# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from typing import Callable

import torch

from vllm import _custom_ops as ops
from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.triton_utils import triton


# TODO(luka): use standalone_compile utility
def with_dyn_arg(fn: Callable, arg_index: int, dim_index: int):
    def inner(*args):
        torch._dynamo.mark_dynamic(args[arg_index], dim_index)
        return fn(*args)

    return inner


torch._dynamo.config.recompile_limit = 8888
compilation_config = CompilationConfig(custom_ops=["none"])
with set_current_vllm_config(VllmConfig(compilation_config=compilation_config)):
    torch_per_token_quant_fp8 = torch.compile(
        QuantFP8(False, GroupShape.PER_TOKEN),
        fullgraph=True,
        dynamic=False,  # recompile for different shapes
    )

    # First dim is explicitly dynamic to simulate vLLM usage
    torch_per_token_quant_fp8 = with_dyn_arg(torch_per_token_quant_fp8, 0, 0)


def cuda_per_token_quant_fp8(
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return ops.scaled_fp8_quant(input)


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
