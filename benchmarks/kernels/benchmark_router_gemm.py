# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse

import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.triton_utils import triton

num_tokens_range = [2**x for x in range(14)]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens"],
        x_vals=num_tokens_range,
        x_log=False,
        line_arg="impl",
        line_vals=[
            "torch-32",
            "gpt_oss_router_gemm-32",
            "torch-128",
            "gpt_oss_router_gemm-128",
        ],
        line_names=(
            [
                "torch-32",
                "gpt_oss_router_gemm-32",
                "torch-128",
                "gpt_oss_router_gemm-128",
            ]
        ),
        styles=([("blue", "-"), ("orange", "-"), ("green", "-"), ("red", "-")]),
        ylabel="TFLOPs",
        plot_name="router gemm throughput",
        args={},
    )
)
def benchmark(num_tokens, impl):
    # M: num_tokens, K: hidden_dim, N: num_experts
    M, K = num_tokens, 2880

    if impl == "torch-32" or impl == "gpt_oss_router_gemm-32":
        N = 32
    elif impl == "torch-128" or impl == "gpt_oss_router_gemm-128":
        N = 128
    else:
        raise ValueError(f"Unknown impl: {impl}")

    mat_a = torch.randn((M, K), dtype=torch.bfloat16, device="cuda").contiguous()
    mat_b = torch.randn((N, K), dtype=torch.bfloat16, device="cuda").contiguous()
    bias = torch.randn(N, dtype=torch.bfloat16, device="cuda").contiguous()

    quantiles = [0.5, 0.2, 0.8]

    if impl == "torch-32" or impl == "torch-128":

        def runner():
            F.linear(mat_a, mat_b, bias)
    elif impl == "gpt_oss_router_gemm-32" or impl == "gpt_oss_router_gemm-128":

        def runner():
            ops.gpt_oss_router_gemm(mat_a, mat_b, bias)

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(runner, quantiles=quantiles)

    def tflops(t_ms):
        flops = 2 * M * K * N
        return flops / (t_ms * 1e-3) / 1e12

    return tflops(ms), tflops(max_ms), tflops(min_ms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    benchmark.run(print_data=True)
