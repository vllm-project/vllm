# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools

import torch

from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser

num_tokens_range = [2**i for i in range(0, 8, 2)]
num_experts_range = [16, 32, 64, 128, 256, 512]
topk_range = [3, 4]
configs = list(itertools.product(num_tokens_range, num_experts_range, topk_range))


def torch_topk(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    scoring_func: str = "softmax",
):
    if scoring_func == "softmax":
        scores = torch.softmax(gating_output.float(), dim=-1)
    else:
        scores = torch.sigmoid(gating_output.float())
    topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids


def get_benchmark(scoring_func):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["num_tokens", "num_experts", "topk"],
            x_vals=[list(_) for _ in configs],
            line_arg="provider",
            line_vals=["torch", "vllm"],
            line_names=["Torch", "vLLM"],
            styles=[("blue", "-"), ("red", "-")],
            ylabel="us",
            plot_name=f"fused-topk-perf-{scoring_func}",
            args={},
        )
    )
    def benchmark(num_tokens, num_experts, topk, provider):
        dtype = torch.bfloat16
        hidden_size = 1024
        renormalize = True
        hidden_states = torch.randn(
            (num_tokens, hidden_size), dtype=dtype, device="cuda"
        )
        gating_output = torch.randn(
            (num_tokens, num_experts), dtype=dtype, device="cuda"
        )

        quantiles = [0.5, 0.2, 0.8]

        if provider == "torch":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch_topk(
                    gating_output=gating_output,
                    topk=topk,
                    renormalize=renormalize,
                    scoring_func=scoring_func,
                ),
                quantiles=quantiles,
            )
        else:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: fused_topk(
                    hidden_states=hidden_states,
                    gating_output=gating_output,
                    topk=topk,
                    renormalize=renormalize,
                    scoring_func=scoring_func,
                ),
                quantiles=quantiles,
            )

        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark the MoE topk kernel.")
    parser.add_argument("--scoring-func", type=str, default="softmax")
    parser.add_argument("--save-path", type=str, default="./configs/fused_topk/")
    args = parser.parse_args()

    # Get the benchmark function
    benchmark = get_benchmark(args.scoring_func)
    # Run performance benchmark
    benchmark.run(print_data=True, save_path=args.save_path)
