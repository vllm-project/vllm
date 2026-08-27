# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark clamped SiLU+mul+FP8 quantization for DSV4-style MoE.

Compares the correct two-kernel Triton MoE path with the fused CUDA path
adapted from SGLang PR #32058. The defaults match DeepSeek-V4-Flash-FP8:
top-k 6, intermediate size 2048, FP8 group size 128, and clamp limit 10.

For GLM-5.3-Flash on Hopper TP4, use ``--topk 8 --hidden-size 512
--num-experts 288 --no-expert-filter --use-ue8m0``.
"""

import argparse
import statistics

import torch

import vllm._custom_ops as ops
from vllm.model_executor.layers.fused_moe.utils import swiglu_limit_func
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.platforms import current_platform


def time_cuda(fn, warmup: int, repeats: int, iterations: int) -> float:
    """Return median latency in microseconds."""
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.Event(enable_timing=True)
        end = torch.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iterations)
    return statistics.median(samples)


def run_case(
    num_tokens: int,
    topk: int,
    hidden_size: int,
    num_experts: int,
    ep_size: int,
    clamp_limit: float,
    use_expert_filter: bool,
    use_ue8m0: bool,
    warmup: int,
    repeats: int,
    iterations: int,
) -> tuple[float, float]:
    """Benchmark one routed-token shape."""
    rows = num_tokens * topk
    x = torch.randn((rows, hidden_size * 2), dtype=torch.bfloat16, device="cuda")
    activated = torch.empty((rows, hidden_size), dtype=torch.bfloat16, device="cuda")
    expert_ids = None
    expert_map = None
    if use_expert_filter:
        expert_ids = torch.randint(
            0, num_experts, (rows,), dtype=torch.int32, device="cuda"
        )
        expert_map = torch.full((num_experts,), -1, dtype=torch.int32, device="cuda")
        local_experts = num_experts // ep_size
        expert_map[:local_experts] = torch.arange(
            local_experts, dtype=torch.int32, device="cuda"
        )

    def baseline():
        if use_expert_filter:
            swiglu_limit_func(activated, x, clamp_limit, expert_ids, expert_map)
        else:
            torch.ops._C.silu_and_mul_with_clamp(activated, x, clamp_limit, 1.0, 0.0)
        return per_token_group_quant_fp8(activated, group_size=128, use_ue8m0=use_ue8m0)

    def fused():
        return ops.silu_and_mul_per_block_quant(
            x,
            group_size=128,
            quant_dtype=current_platform.fp8_dtype(),
            clamp_limit=clamp_limit,
            expert_ids=expert_ids,
            expert_map=expert_map,
            use_ue8m0=use_ue8m0,
        )

    baseline_us = time_cuda(baseline, warmup, repeats, iterations)
    fused_us = time_cuda(fused, warmup, repeats, iterations)
    return baseline_us, fused_us


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 8, 32, 128, 1024])
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument(
        "--no-expert-filter",
        action="store_true",
        help="Benchmark tensor-parallel routing without EP row filtering",
    )
    parser.add_argument(
        "--use-ue8m0",
        action="store_true",
        help="Round FP8 group scales to Hopper DeepGEMM UE8M0 powers of two",
    )
    parser.add_argument("--clamp-limit", type=float, default=10.0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(0)
    print("tokens rows baseline_us fused_us speedup")
    for num_tokens in args.tokens:
        baseline_us, fused_us = run_case(
            num_tokens,
            args.topk,
            args.hidden_size,
            args.num_experts,
            args.ep_size,
            args.clamp_limit,
            not args.no_expert_filter,
            args.use_ue8m0,
            args.warmup,
            args.repeats,
            args.iterations,
        )
        print(
            f"{num_tokens:6d} {num_tokens * args.topk:6d} "
            f"{baseline_us:11.3f} {fused_us:8.3f} "
            f"{baseline_us / fused_us:7.2f}x"
        )


if __name__ == "__main__":
    main()
