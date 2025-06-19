# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import itertools

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size_triton,
)
from vllm.triton_utils import triton


def get_topk_ids(num_tokens: int, num_experts: int, topk: int) -> torch.Tensor:
    return torch.stack(
        [
            torch.randperm(num_experts, dtype=torch.int32, device="cuda")[:topk]
            for _ in range(num_tokens)
        ]
    )


def check_correctness(num_tokens, num_experts=256, block_size=256, topk=8):
    """
    Verifies vllm vs. Triton
    """
    topk_ids = get_topk_ids(num_tokens, num_experts, topk)

    # 1. malloc space for triton and vllm
    # malloc enough space (max_num_tokens_padded) for the sorted ids
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids_triton = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device="cuda"
    )
    sorted_ids_triton.fill_(topk_ids.numel())  # fill with sentinel value
    expert_ids_triton = torch.zeros(
        (max_num_tokens_padded // block_size,), dtype=torch.int32, device="cuda"
    )
    num_tokens_post_pad_triton = torch.empty((1,), dtype=torch.int32, device="cuda")

    sorted_ids_vllm = torch.empty_like(sorted_ids_triton)
    sorted_ids_vllm.fill_(topk_ids.numel())
    expert_ids_vllm = torch.zeros_like(expert_ids_triton)
    num_tokens_post_pad_vllm = torch.empty_like(num_tokens_post_pad_triton)

    # 2. run implementations
    moe_align_block_size_triton(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids_triton,
        expert_ids_triton,
        num_tokens_post_pad_triton,
    )

    ops.moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids_vllm,
        expert_ids_vllm,
        num_tokens_post_pad_vllm,
    )
    print(f"✅ VLLM implementation works with {num_experts} experts!")

    # 3. compare results
    if torch.allclose(expert_ids_triton, expert_ids_vllm) and torch.allclose(
        num_tokens_post_pad_triton, num_tokens_post_pad_vllm
    ):
        print("✅ Triton and VLLM implementations match.")
    else:
        print("❌ Triton and VLLM implementations DO NOT match.")
        print("Triton expert_ids:", expert_ids_triton)
        print("VLLM expert_ids:", expert_ids_vllm)
        print("Triton num_tokens_post_pad:", num_tokens_post_pad_triton)
        print("VLLM num_tokens_post_pad:", num_tokens_post_pad_vllm)


# test configurations
num_tokens_range = [1, 16, 256, 4096]
num_experts_range = [16, 64, 224, 256, 280, 512]
topk_range = [1, 2, 8]
configs = list(itertools.product(num_tokens_range, num_experts_range, topk_range))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens", "num_experts", "topk"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["vllm", "triton"],  # "triton"
        line_names=["VLLM", "Triton"],  # "Triton"
        plot_name="moe-align-block-size-performance",
        args={},
    )
)
def benchmark(num_tokens, num_experts, topk, provider):
    """Benchmark function for Triton."""
    block_size = 256
    topk_ids = get_topk_ids(num_tokens, num_experts, topk)

    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device="cuda")
    sorted_ids.fill_(topk_ids.numel())
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device="cuda")
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device="cuda")

    quantiles = [0.5, 0.2, 0.8]

    if provider == "vllm":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: ops.moe_align_block_size(
                topk_ids,
                num_experts,
                block_size,
                sorted_ids.clone(),
                expert_ids.clone(),
                num_tokens_post_pad.clone(),
            ),
            quantiles=quantiles,
        )
    elif provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: moe_align_block_size_triton(
                topk_ids,
                num_experts,
                block_size,
                sorted_ids.clone(),
                expert_ids.clone(),
                num_tokens_post_pad.clone(),
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

    print("Running correctness check...")
    check_correctness(num_tokens=1024, num_experts=args.num_experts, topk=args.topk)
    benchmark.run(print_data=True, show_plots=True)
