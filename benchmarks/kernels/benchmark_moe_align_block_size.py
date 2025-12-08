# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import itertools

import torch

from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.triton_utils import triton


def get_topk_ids(num_tokens: int, num_experts: int, topk: int) -> torch.Tensor:
    return torch.stack(
        [
            torch.randperm(num_experts, dtype=torch.int32, device="cuda")[:topk]
            for _ in range(num_tokens)
        ]
    )


# test configurations
num_tokens_range = [1, 16, 256, 4096]
num_experts_range = [16, 64, 224, 256, 280, 512]
topk_range = [1, 2, 8]
ep_size_range = [1, 8]
configs = list(
    itertools.product(num_tokens_range, num_experts_range, topk_range, ep_size_range)
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens", "num_experts", "topk", "ep_size"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["vllm"],
        line_names=["vLLM"],
        plot_name="moe-align-block-size-performance",
        args={},
    )
)
def benchmark(num_tokens, num_experts, topk, ep_size, provider):
    """Benchmark function for Triton."""
    block_size = 256
    torch.cuda.manual_seed_all(0)
    topk_ids = get_topk_ids(num_tokens, num_experts, topk)

    e_map = None
    if ep_size != 1:
        local_e = num_experts // ep_size
        e_ids = torch.randperm(num_experts, device="cuda", dtype=torch.int32)[:local_e]
        e_map = torch.full((num_experts,), -1, device="cuda", dtype=torch.int32)
        e_map[e_ids] = torch.arange(local_e, device="cuda", dtype=torch.int32)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "vllm":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: moe_align_block_size(
                topk_ids, block_size, num_experts, e_map, ignore_invalid_experts=True
            ),
            quantiles=quantiles,
        )

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_experts",
        type=int,
        default=64,
        choices=[8, 16, 32, 64, 128, 256],
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=8,
        choices=[2, 4, 8],
        help="Top-k value for correctness check.",
    )
    args = parser.parse_args()

    benchmark.run(print_data=True, show_plots=True)
