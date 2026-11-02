# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare shuffled MXFP4 input quantization with fused row gathering."""

from __future__ import annotations

import argparse
import random
import statistics
import subprocess
from collections.abc import Callable
from functools import partial

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm import _custom_ops as ops


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", default="4,32,128,1024")
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=6144)
    parser.add_argument("--repeat-iters", type=int, default=60)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warm-cache", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    args.token_counts = [int(value) for value in args.tokens.split(",")]
    if not args.token_counts or min(args.token_counts) < 1:
        parser.error("--tokens must contain positive comma-separated values")
    if args.experts < args.top_k or args.top_k < 1:
        parser.error("require experts >= top-k >= 1")
    if args.hidden_size % 128:
        parser.error("--hidden-size must be a multiple of 128")
    if min(args.repeat_iters, args.trials) < 1 or args.warmup < 0:
        parser.error("repeat-iters and trials must be positive; warmup nonnegative")
    return args


def time_graphs(
    invokes: dict[str, Callable[[], None]],
    input_tensors: tuple[torch.Tensor, ...],
    repeat_iters: int,
    warmup: int,
    trials: int,
    cold_l2_cache: bool,
    seed: int,
) -> dict[str, float]:
    timings = {name: [] for name in invokes}
    rng = random.Random(seed)
    for _ in range(trials):
        order = list(invokes)
        rng.shuffle(order)
        for name in order:
            samples_ms = bench_gpu_time_with_cupti(
                partial(_invoke_with_tensors, invokes[name]),
                use_cuda_graph=True,
                cold_l2_cache=cold_l2_cache,
                dry_run_iters=warmup,
                repeat_iters=repeat_iters,
                input_args=input_tensors,
            )
            timings[name].append(statistics.median(samples_ms) * 1000)
    return {name: statistics.median(values) for name, values in timings.items()}


def _invoke_with_tensors(invoke: Callable[[], None], *_tensors: torch.Tensor) -> None:
    invoke()


def run_case(args: argparse.Namespace, tokens: int) -> dict[str, object]:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed + tokens)
    activations = torch.randn(
        tokens,
        args.hidden_size,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).contiguous()
    logits = torch.randn(
        tokens,
        args.experts,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    topk_ids = logits.topk(args.top_k, dim=-1).indices.to(torch.int32).contiguous()

    route_count = tokens * args.top_k
    expert_offsets = torch.empty(args.experts + 1, device=device, dtype=torch.int32)
    blockscale_offsets = torch.empty_like(expert_offsets)
    problem_sizes1 = torch.empty(args.experts, 3, device=device, dtype=torch.int32)
    problem_sizes2 = torch.empty_like(problem_sizes1)
    input_permutation = torch.empty(route_count, device=device, dtype=torch.int32)
    output_permutation = torch.empty_like(input_permutation)
    sorted_expert_ids = torch.empty_like(input_permutation)
    ops.get_cutlass_moe_mm_data(
        topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        input_permutation,
        output_permutation,
        args.experts,
        128,
        args.hidden_size,
        blockscale_offsets,
        is_gated=True,
        sorted_expert_ids=sorted_expert_ids,
    )
    torch.accelerator.synchronize()

    routed = torch.empty(
        route_count, args.hidden_size, dtype=activations.dtype, device=device
    )
    packed_shape = (route_count, args.hidden_size // 2)
    packed_old = torch.empty(packed_shape, dtype=torch.uint8, device=device)
    packed_new = torch.empty_like(packed_old)
    scale_rows = int(blockscale_offsets[-1].item())
    scale_cols = (args.hidden_size // 32 + 3) // 4
    scales_old = torch.empty(scale_rows, scale_cols, dtype=torch.int32, device=device)
    scales_new = torch.empty_like(scales_old)

    def shuffle() -> None:
        torch.ops._moe_C.shuffle_rows(activations, input_permutation, routed)

    def quant_old() -> None:
        torch.ops._C.mxfp4_experts_quant(
            packed_old,
            scales_old,
            routed,
            expert_offsets,
            blockscale_offsets,
            args.experts,
            sorted_expert_ids,
        )

    def quant_new() -> None:
        torch.ops._C.mxfp4_experts_quant(
            packed_new,
            scales_new,
            activations,
            expert_offsets,
            blockscale_offsets,
            args.experts,
            sorted_expert_ids,
            input_permutation,
        )

    def old_pair() -> None:
        shuffle()
        quant_old()

    def check_correctness() -> None:
        packed_old.fill_(0xA5)
        packed_new.fill_(0xA5)
        scales_old.fill_(0x5A5A5A5A)
        scales_new.fill_(0x5A5A5A5A)
        old_pair()
        quant_new()
        torch.accelerator.synchronize()
        if not torch.equal(packed_old, packed_new):
            raise AssertionError(f"packed output mismatch at M={tokens}")
        if not torch.equal(scales_old, scales_new):
            raise AssertionError(f"scale output mismatch at M={tokens}")

    check_correctness()
    timings = time_graphs(
        {
            "shuffle": shuffle,
            "quant_contiguous": quant_old,
            "old_shuffle_plus_quant": old_pair,
            "quant_gather": quant_new,
        },
        (
            activations,
            input_permutation,
            expert_offsets,
            blockscale_offsets,
            sorted_expert_ids,
            routed,
            packed_old,
            packed_new,
            scales_old,
            scales_new,
        ),
        args.repeat_iters,
        args.warmup,
        args.trials,
        not args.warm_cache,
        args.seed + tokens,
    )
    counts = torch.bincount(topk_ids.reshape(-1).long(), minlength=args.experts)
    histogram = {
        value: int((counts == value).sum().item())
        for value in range(int(counts.max().item()) + 1)
    }
    return {
        "tokens": tokens,
        "routed_rows": route_count,
        "valid_rows": int(expert_offsets[-1].item()),
        "nonempty_experts": int((counts > 0).sum().item()),
        "max_rows_per_expert": int(counts.max().item()),
        "expert_row_count_histogram": histogram,
        "median_us": timings,
        "speedup_old_pair_over_gather": (
            timings["old_shuffle_plus_quant"] / timings["quant_gather"]
        ),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device_name = torch.cuda.get_device_name()
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True
    ).strip()
    print(
        f"GPU={device_name} torch={torch.__version__} "
        f"CUDA={torch.version.cuda} commit={commit}"
    )
    print(
        f"experts={args.experts} top_k={args.top_k} hidden={args.hidden_size} "
        f"timing=FlashInfer_CUPTI_CUDA_graph_"
        f"{'hot_L2' if args.warm_cache else 'cold_L2'} "
        f"repeat_iters={args.repeat_iters} trials={args.trials} "
        f"warmup={args.warmup}"
    )
    for tokens in args.token_counts:
        print(run_case(args, tokens), flush=True)


if __name__ == "__main__":
    main()
