# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import time

import numpy as np
import torch

from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser

# Check if CPU MoE operations are available
try:
    from vllm._custom_ops import cpu_fused_moe, cpu_prepack_moe_weight
except (ImportError, AttributeError) as e:
    print("ERROR: CPU fused MoE operations are not available on this platform.")
    print("This benchmark requires x86 CPU with proper vLLM CPU extensions compiled.")
    print(
        "The cpu_fused_moe kernel is typically available on Linux x86_64 "
        "with AVX2/AVX512."
    )
    print(f"Import error: {e}")
    sys.exit(1)

# ISA selection following test_cpu_fused_moe.py pattern
ISA_CHOICES = ["amx", "vec"] if torch._C._cpu._is_amx_tile_supported() else ["vec"]


@torch.inference_mode()
def main(
    batch_size: int,
    expert_num: int,
    hidden_size: int,
    intermediate_size: int,
    topk_num: int,
    use_bias: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    activation: str = "silu",
    isa: str = "vec",
    seed: int = 0,
    iters: int = 20,
) -> None:
    current_platform.seed_everything(seed)
    # up_dim = 2 * intermediate_size for gate + up projection
    up_dim = 2 * intermediate_size

    input_tensor = torch.randn((batch_size, hidden_size), dtype=dtype) / (
        0.5 * hidden_size**0.5
    )

    w13 = torch.randn((expert_num, up_dim, hidden_size), dtype=dtype) / (
        0.5 * hidden_size**0.5
    )
    w2 = torch.randn((expert_num, hidden_size, intermediate_size), dtype=dtype) / (
        0.5 * intermediate_size**0.5
    )

    w13_bias = None
    w2_bias = None
    if use_bias:
        w13_bias = torch.randn((expert_num, up_dim), dtype=dtype) / (0.5 * up_dim**0.5)
        w2_bias = torch.randn((expert_num, hidden_size), dtype=dtype) / (
            0.5 * hidden_size**0.5
        )

    router_logits = torch.randn((batch_size, expert_num), dtype=dtype)
    score = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(score, topk_num)
    topk_ids = topk_ids.to(torch.int32)

    packed_w13 = cpu_prepack_moe_weight(w13, isa)
    packed_w2 = cpu_prepack_moe_weight(w2, isa)

    def run_benchmark(iters: int) -> list[float]:
        times = []
        for _ in range(iters):
            start_time = time.perf_counter_ns()
            _ = cpu_fused_moe(
                input_tensor,
                packed_w13,
                packed_w2,
                w13_bias,
                w2_bias,
                topk_weights,
                topk_ids,
                activation,
                isa,
            )
            end_time = time.perf_counter_ns()
            times.append((end_time - start_time) / 1e6)
        return times

    # warmup
    run_benchmark(5)
    # benchmark
    times = run_benchmark(iters)

    if not times:
        print("No iterations to measure. Set --iters > 0.")
        return

    time_min = min(times)
    time_max = max(times)
    time_mean = np.mean(times)
    time_std = np.std(times)

    print("\tmin (ms) = ", time_min)
    print("\tmax (ms) = ", time_max)
    print("\tmean (ms) = ", time_mean)
    print("\tstd = ", time_std)
    print("\tmedian (ms) = ", np.median(times))

    # Calculate throughput metrics
    # FLOPs estimation: 2 * batch * topk * (hidden * up_dim + intermediate * hidden)
    flops_per_token = (
        2 * topk_num * (hidden_size * up_dim + intermediate_size * hidden_size)
    )
    total_flops = batch_size * flops_per_token
    tflops = total_flops / (time_mean * 1e-3) / 1e12
    print(f"\tthroughput (TFLOP/s) = {tflops:.4f}")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark the CPU fused MoE kernel.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--expert-num", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=2880)
    parser.add_argument("--intermediate-size", type=int, default=2880)
    parser.add_argument(
        "--topk-num",
        type=int,
        default=None,
        help="Number of experts to route each token to (default: expert_num // 2)",
    )
    parser.add_argument("--use-bias", action="store_true")
    parser.add_argument(
        "--activation",
        type=str,
        choices=["silu", "swigluoai"],
        default="silu",
        help="Activation function",
    )
    parser.add_argument(
        "--isa",
        type=str,
        choices=ISA_CHOICES,
        default=ISA_CHOICES[0],
        help=f"ISA to use (available: {ISA_CHOICES})",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters", type=int, default=20)

    args = parser.parse_args()

    # Default topk_num to expert_num // 2, minimum 1
    topk_num = (
        args.topk_num if args.topk_num is not None else max(args.expert_num // 2, 1)
    )

    print(args)

    main(
        batch_size=args.batch_size,
        expert_num=args.expert_num,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        topk_num=topk_num,
        use_bias=args.use_bias,
        dtype=torch.bfloat16,  # Following test_cpu_fused_moe.py
        activation=args.activation,
        isa=args.isa,
        seed=args.seed,
        iters=args.iters,
    )
