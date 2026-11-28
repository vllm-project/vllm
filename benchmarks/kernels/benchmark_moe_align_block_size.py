# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import itertools

import torch

import vllm._custom_ops as ops
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
    moe_align_block_size_pair,
)
from vllm.triton_utils import triton
from vllm.utils.torch_utils import set_random_seed


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
    set_random_seed(0)
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


batched_configs = list(
    itertools.product(
        [8, 32, 64, 256],
        [16, 256, 512, 2048],
        [8, 32, 48, 64],
        ["uniform", "skewed", "mostly_empty"],
    )
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=[
            "num_batches",
            "max_tokens_per_batch",
            "block_size",
            "distribution",
        ],
        x_vals=batched_configs,
        line_arg="provider",
        line_vals=["vllm"],
        line_names=["vLLM"],
        plot_name="batched-moe-align-block-size-performance",
        args={},
    )
)
def benchmark_batched(
    num_batches,
    max_tokens_per_batch,
    block_size,
    distribution,
    provider,
):
    set_random_seed(0)
    if distribution == "uniform":
        expert_num_tokens = torch.full(
            (num_batches,),
            max_tokens_per_batch,
            dtype=torch.int32,
            device="cuda",
        )
    elif distribution == "skewed":
        expert_num_tokens = torch.ones((num_batches,), dtype=torch.int32, device="cuda")
        expert_num_tokens[0] = max_tokens_per_batch
    else:
        expert_num_tokens = torch.zeros(
            (num_batches,), dtype=torch.int32, device="cuda"
        )
        expert_num_tokens[::8] = max_tokens_per_batch

    max_num_tokens_padded = (
        num_batches
        * ((max_tokens_per_batch + block_size - 1) // block_size)
        * block_size
    )
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device="cuda")
    expert_ids = torch.empty(
        (max_num_tokens_padded // block_size,), dtype=torch.int32, device="cuda"
    )
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device="cuda")

    quantiles = [0.5, 0.2, 0.8]
    if provider == "vllm":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: ops.batched_moe_align_block_size(
                max_tokens_per_batch,
                block_size,
                expert_num_tokens,
                sorted_ids,
                expert_ids,
                num_tokens_post_pad,
            ),
            quantiles=quantiles,
        )

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


def benchmark_paired_alignment():
    """Compare two native alignments with paired dispatch using cold-L2 CUPTI."""
    import json
    import statistics

    from flashinfer.testing import bench_gpu_time_with_cupti

    cases = itertools.product(
        [(1, 1), (1, 6), (2, 8), (8, 6), (8, 8), (13, 5), (512, 8)],
        [16, 64, 384],
        [(8, 40), (128, 32), (64, 64)],
        [1, 2],
    )
    set_random_seed(0)
    for (num_tokens, topk), num_experts, blocks, ep_size in cases:
        topk_ids = torch.randint(
            0, num_experts, (num_tokens, topk), device="cuda", dtype=torch.int32
        )
        expert_map = None
        if ep_size > 1:
            expert_map = torch.full(
                (num_experts,), -1, device="cuda", dtype=torch.int32
            )
            local_experts = torch.randperm(num_experts, device="cuda")[::ep_size]
            expert_map[local_experts] = torch.arange(
                local_experts.numel(), device="cuda", dtype=torch.int32
            )

        def baseline(
            topk_ids=topk_ids,
            blocks=blocks,
            num_experts=num_experts,
            expert_map=expert_map,
        ):
            first = moe_align_block_size(
                topk_ids,
                blocks[0],
                num_experts,
                expert_map,
                ignore_invalid_experts=True,
            )
            second = (
                first
                if blocks[0] == blocks[1]
                else moe_align_block_size(
                    topk_ids,
                    blocks[1],
                    num_experts,
                    expert_map,
                    ignore_invalid_experts=True,
                )
            )
            return first, second

        def paired(
            topk_ids=topk_ids,
            blocks=blocks,
            num_experts=num_experts,
            expert_map=expert_map,
        ):
            return moe_align_block_size_pair(topk_ids, *blocks, num_experts, expert_map)

        for block, reference, candidate in zip(blocks, baseline(), paired()):
            torch.testing.assert_close(candidate[2], reference[2], atol=0, rtol=0)
            num_blocks = candidate[2].item() // block
            torch.testing.assert_close(
                candidate[1][:num_blocks], reference[1][:num_blocks], atol=0, rtol=0
            )

        samples = {"baseline": [], "paired": []}
        for name, fn in [
            ("baseline", baseline),
            ("paired", paired),
            ("paired", paired),
            ("baseline", baseline),
        ]:
            fn()
            torch.accelerator.synchronize()
            samples[name].extend(
                bench_gpu_time_with_cupti(
                    fn,
                    dry_run_iters=5,
                    repeat_iters=100,
                    use_cuda_graph=True,
                    cold_l2_cache=True,
                )
            )
        baseline_us = statistics.median(samples["baseline"]) * 1000
        paired_us = statistics.median(samples["paired"]) * 1000
        print(
            json.dumps(
                dict(
                    num_tokens=num_tokens,
                    topk=topk,
                    num_experts=num_experts,
                    blocks=blocks,
                    ep_size=ep_size,
                    baseline_us=baseline_us,
                    paired_us=paired_us,
                    speedup=baseline_us / paired_us,
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batched", action="store_true")
    parser.add_argument("--paired", action="store_true")
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

    if args.paired:
        benchmark_paired_alignment()
    else:
        selected_benchmark = benchmark_batched if args.batched else benchmark
        selected_benchmark.run(print_data=True, show_plots=True)
