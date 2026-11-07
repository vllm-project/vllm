# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare compiled SiLU + FlashInfer GEMM against the fused SM90 producer.

Run after rebuilding vLLM: python benchmarks/kernels/benchmark_silu_mul_fp8_sm90.py
"""

import json
import statistics

import torch
from flashinfer.gemm import fp8_blockscale_gemm_sm90

from vllm import _custom_ops as ops


@torch.compile(fullgraph=True)
def activation(x):
    k = x.shape[-1] // 2
    return torch.nn.functional.silu(x[..., :k]) * x[..., k:]


def time_graph(fn):
    for _ in range(5):
        fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(50):
            fn()
    samples = []
    for _ in range(7):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(20):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    # 1000 operations per sample: milliseconds numerically equal microseconds/op.
    return statistics.median(samples)


def main():
    torch.manual_seed(42)
    k, n = 17408, 5120
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    scales = torch.ones(n // 128, k // 128, device="cuda")
    for m in [3, 6, 16, 31]:
        x = torch.randn(m, 2 * k, device="cuda", dtype=torch.bfloat16)

        def baseline(x=x):
            return fp8_blockscale_gemm_sm90(activation(x), weight, weight_scale=scales)

        def fused(x=x):
            q, s = ops.silu_mul_fp8_sm90(x)
            return fp8_blockscale_gemm_sm90(
                q, weight, input_scale=s, weight_scale=scales
            )

        assert torch.equal(baseline(), fused())
        old, new = time_graph(baseline), time_graph(fused)
        print(json.dumps(dict(m=m, n=n, k=k, baseline_us=old, fused_us=new)))


if __name__ == "__main__":
    main()
