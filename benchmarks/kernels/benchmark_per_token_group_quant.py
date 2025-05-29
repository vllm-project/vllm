# SPDX-License-Identifier: Apache-2.0
import itertools

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.triton_utils import triton

fp8_type_ = torch.float8_e4m3fn


def calculate_diff(batch_size, seq_len, group_size):
    device = torch.device("cuda")
    hidden_dim = 7168

    x = torch.randn(
        batch_size * seq_len, hidden_dim, device=device, dtype=torch.float16
    )

    x_q_triton, x_s_triton = per_token_group_quant_fp8(x.clone(), group_size)
    x_q_cuda, x_s_cuda = ops.per_token_group_quant_fp8(x.clone(), group_size)

    if torch.allclose(
        x_q_triton.to(torch.float32), x_q_cuda.to(torch.float32), rtol=1e-3, atol=1e-5
    ) and torch.allclose(x_s_triton, x_s_cuda, rtol=1e-3, atol=1e-5):
        print("✅ Implementations match")
    else:
        print("❌ Implementations differ")


batch_size_range = [1, 2, 4, 8, 16, 32, 64]
seq_len_range = [64, 128, 256, 512, 1024, 2048]
group_size_range = [128]  # For DeepSeek V3/R1

configs = list(itertools.product(batch_size_range, seq_len_range, group_size_range))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "seq_len", "group_size"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["triton", "cuda"],
        line_names=["Triton", "CUDA"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="per-token-group-quant-performance",
        args={},
    )
)
def benchmark(batch_size, seq_len, group_size, provider):
    device = torch.device("cuda")
    hidden_dim = 7168

    x = torch.randn(
        batch_size * seq_len, hidden_dim, device=device, dtype=torch.bfloat16
    )

    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        fn = lambda: per_token_group_quant_fp8(x.clone(), group_size)
    elif provider == "cuda":
        fn = lambda: ops.per_token_group_quant_fp8(x.clone(), group_size)

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    calculate_diff(batch_size=4, seq_len=128, group_size=128)

    benchmark.run(print_data=True)
