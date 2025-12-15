# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools

import torch

from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser

batch_size_range = [2**i for i in range(0, 8, 2)]
seq_len_range = [2**i for i in range(6, 10, 1)]
num_heads_range = [32, 48]
configs = list(itertools.product(batch_size_range, seq_len_range, num_heads_range))


def get_benchmark(head_size, rotary_dim, is_neox_style, device):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["batch_size", "seq_len", "num_heads"],
            x_vals=[list(_) for _ in configs],
            line_arg="provider",
            line_vals=["torch", "flashinfer", "vllm"],
            line_names=["PyTorch", "FlashInfer", "vLLM"],
            styles=[("blue", "-"), ("green", "-"), ("red", "-")],
            ylabel="us",
            plot_name=f"rope-perf{'-neox-style' if is_neox_style else ''}",
            args={},
        )
    )
    def benchmark(batch_size, seq_len, num_heads, provider):
        dtype = torch.bfloat16
        max_position = 8192
        rope_parameters = {"partial_rotary_factor": rotary_dim / head_size}
        rope = get_rope(head_size, max_position, is_neox_style, rope_parameters)
        rope = rope.to(dtype=dtype, device=device)
        cos_sin_cache = rope.cos_sin_cache.to(dtype=torch.float, device=device)

        positions = torch.randint(0, max_position, (batch_size, seq_len), device=device)
        query = torch.randn(
            (batch_size, seq_len, num_heads * head_size), dtype=dtype, device=device
        )
        key = torch.randn_like(query)

        quantiles = [0.5, 0.2, 0.8]

        if provider == "torch":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rope.forward_native(positions, query.clone(), key.clone()),
                quantiles=quantiles,
            )
        elif provider == "flashinfer":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch.ops.vllm.flashinfer_rotary_embedding(
                    positions,
                    query.clone(),
                    key.clone(),
                    head_size,
                    cos_sin_cache,
                    is_neox_style,
                ),
                quantiles=quantiles,
            )
        else:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rope.forward_cuda(positions, query.clone(), key.clone()),
                quantiles=quantiles,
            )

        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the rotary embedding kernels."
    )
    parser.add_argument("--is-neox-style", type=bool, default=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument(
        "--head-size",
        type=int,
        choices=[64, 80, 96, 112, 120, 128, 192, 256],
        default=128,
    )
    parser.add_argument("--rotary-dim", type=int, choices=[16, 32], default=32)
    parser.add_argument(
        "--dtype", type=str, choices=["bfloat16", "float"], default="float"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", type=str, choices=["cuda:0", "cuda:1"], default="cuda:0"
    )
    parser.add_argument("--save-path", type=str, default="./configs/rope/")
    args = parser.parse_args()

    # Get the benchmark function
    benchmark = get_benchmark(
        args.head_size, args.rotary_dim, args.is_neox_style, args.device
    )
    # Run performance benchmark
    benchmark.run(print_data=True, save_path=args.save_path)
