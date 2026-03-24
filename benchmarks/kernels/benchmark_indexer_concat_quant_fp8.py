# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    indexer_concat_quant_fp8,
    per_token_group_quant_fp8,
)
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser

num_tokens_range = [2**x for x in range(14)]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens"],
        x_vals=num_tokens_range,
        x_log=False,
        line_arg="provider",
        line_vals=[
            "torch",
            "vllm",
        ],
        line_names=["PyTorch", "vLLM"],
        styles=([("blue", "-"), ("red", "-")]),
        ylabel="us",
        plot_name="indexer_concat_quant_fp8 latency",
        args={},
    )
)
def benchmark(num_tokens, provider):
    num_heads = 64
    head_dim = 128
    rope_dim = 64
    group_size = 128
    nope_dim = head_dim - rope_dim
    q_pe = torch.randn(
        num_tokens, num_heads, rope_dim, dtype=torch.bfloat16, device="cuda"
    )
    q_nope = torch.randn(
        num_tokens, num_heads, nope_dim, dtype=torch.bfloat16, device="cuda"
    )

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":

        def runner():
            q = torch.cat([q_pe, q_nope], dim=-1)
            q = q.view(-1, head_dim)
            per_token_group_quant_fp8(q, group_size, column_major_scales=False)

    elif provider == "vllm":

        def runner():
            indexer_concat_quant_fp8(
                q_pe, q_nope, group_size, column_major_scales=False
            )

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(runner, quantiles=quantiles)

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument(
        "--save-path", type=str, default=None, help="Path to save benchmark results"
    )
    args = parser.parse_args()

    # Run performance benchmark
    benchmark.run(print_data=True, save_path=args.save_path)
