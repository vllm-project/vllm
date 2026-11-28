# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
import time

import numpy as np
import torch

from vllm.lora.layers import LoRAMapping
from vllm.lora.punica_wrapper.punica_cpu import PunicaWrapperCPU
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, set_random_seed

# Enable AMX tile data registers so isolated runs don't rely on other operations
# to trigger oneDNN's _init_amx() first (which can cause hangs otherwise).
if torch.cpu._is_amx_tile_supported():
    torch.cpu._init_amx()


def make_index_mapping(
    num_tokens: int, lora_fraction: float, max_loras: int, seed: int
) -> list[int]:
    """Per-token LoRA ids: 0 is base model, 1..max_loras are adapters."""
    num_lora_tokens = round(num_tokens * lora_fraction)
    ids = [i % max_loras + 1 for i in range(num_lora_tokens)]
    ids += [0] * (num_tokens - num_lora_tokens)
    g = torch.Generator().manual_seed(seed)
    return [ids[i] for i in torch.randperm(num_tokens, generator=g).tolist()]


@torch.inference_mode()
def bench_add_lora_linear(
    num_tokens: int,
    lora_fraction: float,
    hidden_size: int,
    output_slices: tuple[int, ...],
    rank: int,
    max_loras: int,
    dtype: torch.dtype,
    iters: int,
    seed: int = 0,
) -> list[float]:
    set_random_seed(seed)
    punica = PunicaWrapperCPU(max(num_tokens, 1), 256, "cpu")
    index_mapping = make_index_mapping(num_tokens, lora_fraction, max_loras, seed)
    punica.update_metadata(
        LoRAMapping(index_mapping, index_mapping, is_prefill=False),
        list(range(1, max_loras + 1)),
        max_loras,
        vocab_size=32000,
    )

    lora_a = tuple(
        torch.randn(max_loras, 1, rank, hidden_size, dtype=dtype) for _ in output_slices
    )
    lora_b = tuple(
        torch.randn(max_loras, 1, size, rank, dtype=dtype) for size in output_slices
    )
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    y = torch.zeros(num_tokens, sum(output_slices), dtype=dtype)

    def run(n: int) -> list[float]:
        times = []
        for _ in range(n):
            start = time.perf_counter_ns()
            punica.add_lora_linear(y, x, lora_a, lora_b, 1.0, output_slices)
            times.append((time.perf_counter_ns() - start) / 1e6)
        return times

    run(5)
    return run(iters)


def main(
    num_tokens: list[int],
    lora_fractions: list[float],
    hidden_size: int,
    output_slices: tuple[int, ...],
    rank: int,
    max_loras: int,
    dtype: torch.dtype,
    iters: int,
    seed: int = 0,
) -> list[dict]:
    results = []
    print(f"{'num_tokens':>10} {'lora_frac':>9} {'median_ms':>10} {'mean_ms':>9}")
    for t, frac in itertools.product(num_tokens, lora_fractions):
        times = bench_add_lora_linear(
            t, frac, hidden_size, output_slices, rank, max_loras, dtype, iters, seed
        )
        median, mean = float(np.median(times)), float(np.mean(times))
        results.append(
            dict(num_tokens=t, lora_fraction=frac, median_ms=median, mean_ms=mean)
        )
        print(f"{t:>10} {frac:>9.2f} {median:>10.3f} {mean:>9.3f}")
    return results


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark PunicaWrapperCPU.add_lora_linear in the decode "
        "path with a mix of LoRA and base-model tokens."
    )
    parser.add_argument(
        "--num-tokens", type=int, nargs="+", default=[1, 8, 32, 128, 256]
    )
    parser.add_argument(
        "--lora-fractions", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0]
    )
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument(
        "--output-slices",
        type=int,
        nargs="+",
        default=[4096, 1024, 1024],
        help="Packed output slice sizes; the default is a GQA qkv_proj.",
    )
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--max-loras", type=int, default=4)
    parser.add_argument(
        "--dtype", type=str, choices=["half", "bfloat16", "float"], default="bfloat16"
    )
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(args)

    main(
        num_tokens=args.num_tokens,
        lora_fractions=args.lora_fractions,
        hidden_size=args.hidden_size,
        output_slices=tuple(args.output_slices),
        rank=args.rank,
        max_loras=args.max_loras,
        dtype=STR_DTYPE_TO_TORCH_DTYPE[args.dtype],
        iters=args.iters,
        seed=args.seed,
    )
