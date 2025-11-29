# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# This script benchmarks the mrope kernel (mainly for Qwen2VL and Qwen2.5VL models).
# It generates test data, runs benchmarks, and saves results to a CSV file.
#
# The CSV file (named with current date/time) contains these columns:
# model_name, tp_size, num_tokens, num_heads, num_kv_heads, head_dim, max_position,
# is_neox_style, rope_parameters, dtype, torch_mean, torch_median, torch_p99,
# torch_min, torch_max, triton_mean, triton_median, triton_p99, triton_min, triton_max,
# speedup
#
# == Usage Examples ==
#
# Single model benchmark:
# python3 benchmark_mrope.py --model-name Qwen/Qwen2-VL-7B-Instruct --tp-size 1 \
#   --warmup-iter 10 --benchmark-iter 100 --dtype bfloat16 --seed 0 --num-tokens 1024
#
# All models benchmark:
# python3 benchmark_mrope.py --model-name "" --tp-size 1 --warmup-iter 10 \
#   --benchmark-iter 100 --dtype bfloat16 --seed 0 --num-tokens 1024
#
# All models with different TP sizes:
# python3 benchmark_mrope.py --model-name "" --tp-size 1 2 4 8 --warmup-iter 10 \
#   --benchmark-iter 100 --dtype bfloat16 --seed 0 --num-tokens 1024
#
# All models with different token counts:
# python3 benchmark_mrope.py --model-name "" --tp-size 1 --warmup-iter 10 \
#   --benchmark-iter 100 --dtype bfloat16 --seed 0 --num-tokens 1024 4096 16384
import csv
import os
import time
from datetime import datetime
from typing import Any

import numpy as np
import torch

from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.platforms import current_platform
from vllm.transformers_utils.config import get_config
from vllm.utils.argparse_utils import FlexibleArgumentParser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_test_data(
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    max_position_embeddings: int,
    dtype: torch.dtype,
    device: torch.device,
):
    """Generate test data for given configuration."""
    # Create 2D positions (3, num_tokens) for multimodal case
    positions = torch.randint(
        0, max_position_embeddings // 4, (3, num_tokens), device=device
    )

    # Create query and key tensors
    query = torch.randn(num_tokens, num_q_heads * head_size, dtype=dtype, device=device)
    key = torch.randn(num_tokens, num_kv_heads * head_size, dtype=dtype, device=device)

    return positions, query, key


def calculate_stats(times: list[float]) -> dict[str, float]:
    """Calculate statistics from a list of times."""
    times_array = np.array(times)
    return {
        "mean": np.mean(times_array),
        "median": np.median(times_array),
        "p99": np.percentile(times_array, 99),
        "min": np.min(times_array),
        "max": np.max(times_array),
    }


def benchmark_mrope(
    model_name: str,
    num_tokens: int,
    head_dim: int,
    tp_size: int,
    num_heads: int,
    num_kv_heads: int,
    max_position: int = 8192,
    is_neox_style: bool = True,
    rope_parameters: dict[str, Any] | None = None,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
    warmup_iter: int = 10,
    benchmark_iter: int = 100,
    csv_writer=None,
):
    current_platform.seed_everything(seed)
    torch.set_default_device(device)
    # the parameters to compute the q k v size based on tp_size
    mrope_helper_class = get_rope(
        head_size=head_dim,
        rotary_dim=head_dim,
        max_position=max_position,
        is_neox_style=is_neox_style,
        rope_parameters=rope_parameters,
        dtype=dtype,
    ).to(device=device)

    print(80 * "=")
    print(
        f"Evaluating model: {model_name} "
        f"with tp_size: {tp_size} "
        f"and num_tokens: {num_tokens}, "
        f"dtype: {dtype}"
    )

    # create q k v input tensors
    # create rotary pos emb input tensors
    positions, query, key = generate_test_data(
        num_tokens, num_heads, num_kv_heads, head_dim, max_position, dtype, device
    )

    # Warm up
    for _ in range(warmup_iter):
        mrope_helper_class.forward_native(
            positions,
            query.clone(),
            key.clone(),
        )

        mrope_helper_class.forward_cuda(
            positions,
            query.clone(),
            key.clone(),
        )

    torch.cuda.synchronize()

    # Time reference implementation
    torch_times = []
    for _ in range(benchmark_iter):
        query_clone = query.clone()
        key_clone = key.clone()
        torch.cuda.synchronize()
        start_time = time.time()

        mrope_helper_class.forward_native(
            positions,
            query_clone,
            key_clone,
        )

        torch.cuda.synchronize()
        torch_times.append(time.time() - start_time)

    # Time triton kernel implementation
    triton_times = []
    for _ in range(benchmark_iter):
        query_clone = query.clone()
        key_clone = key.clone()
        torch.cuda.synchronize()
        start_time = time.time()
        mrope_helper_class.forward_cuda(
            positions,
            query_clone,
            key_clone,
        )
        torch.cuda.synchronize()
        triton_times.append(time.time() - start_time)

    # Calculate statistics
    torch_stats = calculate_stats(torch_times)
    triton_stats = calculate_stats(triton_times)
    print(f"\nPerformance for config ({num_tokens}, {num_heads}, {num_kv_heads}):")

    print(
        f"Torch implementation: "
        f"mean={torch_stats['mean']:.8f}s, "
        f"median={torch_stats['median']:.8f}s, "
        f"p99={torch_stats['p99']:.8f}s"
    )

    print(
        f"Triton implementation: "
        f"mean={triton_stats['mean']:.8f}s, "
        f"median={triton_stats['median']:.8f}s, "
        f"p99={triton_stats['p99']:.8f}s"
    )

    print(
        f"Triton Speedup over Torch: {torch_stats['mean'] / triton_stats['mean']:.8f}x"
    )

    # Write to CSV
    if csv_writer:
        row = [
            model_name,
            tp_size,
            num_tokens,
            num_heads,
            num_kv_heads,
            head_dim,
            max_position,
            is_neox_style,
            str(rope_parameters),
            str(dtype).split(".")[-1],
            torch_stats["mean"],
            torch_stats["median"],
            torch_stats["p99"],
            torch_stats["min"],
            torch_stats["max"],
            triton_stats["mean"],
            triton_stats["median"],
            triton_stats["p99"],
            triton_stats["min"],
            triton_stats["max"],
            torch_stats["mean"] / triton_stats["mean"],  # speedup
        ]
        csv_writer.writerow(row)

    return torch_stats, triton_stats


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the rotary embedding kernels."
    )
    parser.add_argument("--model-name", type=str, default="")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--warmup-iter", type=int, default=10)
    parser.add_argument("--benchmark-iter", type=int, default=100)
    parser.add_argument("--dtype", type=str, choices=["bfloat16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, nargs="+", required=False)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-csv", type=str, default="mrope_benchmark_results.csv")
    args = parser.parse_args()
    print(args)

    # Create CSV file for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{os.path.splitext(args.output_csv)[0]}_{timestamp}.csv"

    with open(csv_filename, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        header = [
            "model_name",
            "tp_size",
            "num_tokens",
            "num_heads",
            "num_kv_heads",
            "head_dim",
            "max_position",
            "is_neox_style",
            "rope_parameters",
            "dtype",
            "torch_mean",
            "torch_median",
            "torch_p99",
            "torch_min",
            "torch_max",
            "triton_mean",
            "triton_median",
            "triton_p99",
            "triton_min",
            "triton_max",
            "speedup",
        ]
        csv_writer.writerow(header)

        model_tp_dict = {}
        if args.model_name == "":
            model_tp_dict = {
                "Qwen/Qwen2-VL-2B-Instruct": [1],
                "Qwen/Qwen2-VL-7B-Instruct": [1],
                "Qwen/Qwen2-VL-72B-Instruct": [2, 4, 8],
                "Qwen/Qwen2.5-VL-3B-Instruct": [1, 2, 4, 8],
                "Qwen/Qwen2.5-VL-7B-Instruct": [1, 2, 4, 8],
                "Qwen/Qwen2.5-VL-72B-Instruct": [2, 4, 8],
            }
        else:
            model_tp_dict[args.model_name] = [args.tp_size]

        if args.num_tokens is None:
            num_tokens_list = [2**i for i in range(0, 18)]
        else:
            num_tokens_list = args.num_tokens

        for model_name, tp_list in model_tp_dict.items():
            config = get_config(model_name, trust_remote_code=args.trust_remote_code)
            for tp_size in tp_list:
                # get the model config
                total_num_kv_heads = config.num_key_value_heads
                total_num_heads = config.num_attention_heads
                num_heads = total_num_heads // tp_size
                num_kv_heads = max(1, total_num_kv_heads // tp_size)
                head_dim = config.hidden_size // total_num_heads
                q_size = num_heads * head_dim
                kv_size = num_kv_heads * head_dim
                is_neox_style = True
                rope_parameters = config.rope_parameters
                max_position = config.max_position_embeddings

                for num_tokens in num_tokens_list:
                    benchmark_mrope(
                        model_name=model_name,
                        num_tokens=num_tokens,
                        head_dim=head_dim,
                        tp_size=tp_size,
                        num_heads=num_heads,
                        num_kv_heads=num_kv_heads,
                        max_position=max_position,
                        is_neox_style=is_neox_style,
                        rope_parameters=rope_parameters,
                        dtype=getattr(torch, args.dtype),
                        seed=args.seed,
                        warmup_iter=args.warmup_iter,
                        benchmark_iter=args.benchmark_iter,
                        csv_writer=csv_writer,
                    )

    print(f"Benchmark results saved to {csv_filename}")
