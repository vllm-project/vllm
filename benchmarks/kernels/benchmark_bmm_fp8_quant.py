# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark: Fused BMM+FP8 quant vs separate BMM then FP8 quant.

Measures the latency of the MLA _v_up_proj operation for decode:
  - Baseline: torch.bmm → transpose+reshape → static_scaled_fp8_quant
  - Fused:    bmm_fp8_quant (single Triton kernel)

Usage:
    python benchmarks/kernels/benchmark_bmm_fp8_quant.py
"""

import torch

from vllm._custom_ops import static_scaled_fp8_quant
from vllm.platforms import current_platform
from vllm.triton_utils import triton as vllm_triton
from vllm.v1.attention.ops.triton_bmm_fp8 import bmm_fp8_quant

# DeepSeek-V3/R1 MLA dimensions
N_HEADS = [16, 64, 128]
KV_LORA_RANK = 512
V_HEAD_DIM = 128


@vllm_triton.testing.perf_report(
    vllm_triton.testing.Benchmark(
        x_names=["B"],
        x_vals=[1, 4, 8, 16, 32, 64, 128, 256],
        line_arg="provider",
        line_vals=["baseline", "fused"],
        line_names=["torch.bmm + fp8_quant", "fused bmm_fp8_quant"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="Time (ms)",
        plot_name="bmm_fp8_quant_N{}_L{}_V{}".format(
            N_HEADS[0], KV_LORA_RANK, V_HEAD_DIM
        ),
        args={
            "N": N_HEADS[0],
            "L": KV_LORA_RANK,
            "V": V_HEAD_DIM,
        },
    )
)
def benchmark(B, N, L, V, provider):
    device = "cuda"
    dtype = torch.bfloat16
    fp8_dtype = current_platform.fp8_dtype()

    inp = torch.randn(N, B, L, dtype=dtype, device=device)
    weight = torch.randn(N, L, V, dtype=dtype, device=device)
    scale = torch.tensor([0.01], dtype=torch.float32, device=device)

    out_bf16 = torch.empty(B, N * V, dtype=dtype, device=device)
    out_fp8 = torch.empty(B, N * V, dtype=fp8_dtype, device=device)

    if provider == "baseline":

        def fn():
            bmm_result = torch.bmm(inp, weight)
            out_bf16[:] = bmm_result.transpose(0, 1).reshape(B, N * V)
            static_scaled_fp8_quant(out_fp8, out_bf16, scale)

    elif provider == "fused":

        def fn():
            bmm_fp8_quant(inp, weight, scale, out_fp8)

    ms, min_ms, max_ms = vllm_triton.testing.do_bench_cudagraph(
        fn, rep=500, return_mode="median"
    )
    return ms, min_ms, max_ms


if __name__ == "__main__":
    benchmark.run(print_data=True)
