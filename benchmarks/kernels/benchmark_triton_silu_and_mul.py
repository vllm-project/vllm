# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch
import triton

from vllm import _custom_ops as ops  # noqa: F401
from vllm.model_executor.layers.fused_moe.fused_moe import _silu_and_mul
from vllm.utils.argparse_utils import FlexibleArgumentParser

DEFAULT_HIDDEN_SIZES = [
    128,
    256,
    512,
    1110,
    1024,
    1110,
    2048,
    4096,
    8192,
    16384,
]
DEFAULT_NUM_TOKENS = [i for i in range(2, 8192, 128)]


def get_benchmark(token_list: list[int], hs_list: list[int]):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["num_tokens", "hidden_size"],
            x_vals=[(t, h) for t in token_list for h in hs_list],
            line_arg="provider",
            line_vals=["triton", "cuda"],
            line_names=["Triton", "CUDA"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="us",
            plot_name="triton silu_and_mul bench",
            args={},
        )
    )
    def benchmark(num_tokens, hidden_size, provider):
        assert hidden_size % 2 == 0
        dtype = torch.bfloat16

        hidden_states = torch.randn(
            (num_tokens, hidden_size), dtype=dtype, device="cuda"
        )

        out_triton = torch.empty(
            (num_tokens, hidden_size // 2), dtype=dtype, device="cuda"
        )

        out_cuda = torch.empty(
            (num_tokens, hidden_size // 2), dtype=dtype, device="cuda"
        )
        quantiles = [0.5, 0.2, 0.8]

        if provider == "triton":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: _silu_and_mul(output=out_triton, input=hidden_states),
                quantiles=quantiles,
            )
        else:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch.ops._C.silu_and_mul(out_cuda, hidden_states),
                quantiles=quantiles,
            )

        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",")]


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the Triton silu_and_mul kernel."
    )

    parser.add_argument(
        "--num-tokens",
        type=str,
        default=None,
        help="Number of tokens, comma-separated for tokens mode "
        "(e.g., '1,64,256,1024,8192')",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=str,
        default=None,
        help="Comma-separated hidden sizes (e.g., '1024,2048,4096'). "
        "Default uses preset configs.",
    )
    parser.add_argument("--save-path", type=str, default="./configs/")

    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    if args.num_tokens:
        token_list = parse_int_list(args.num_tokens)
    else:
        token_list = DEFAULT_NUM_TOKENS

    if args.hidden_sizes:
        hidden_sizes = parse_int_list(args.hidden_sizes)
    else:
        hidden_sizes = DEFAULT_HIDDEN_SIZES

    print(f"Num tokens: {token_list}")
    print(f"Hidden sizes: {hidden_sizes}")

    print(f"Save path: {args.save_path}")
    print("-" * 50)

    benchmark = get_benchmark(token_list, hidden_sizes)
    benchmark.run(print_data=True, save_path=args.save_path)
