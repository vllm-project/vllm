# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random
import time

import torch
from tabulate import tabulate

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    create_kv_caches_with_random,
    set_random_seed,
)

logger = init_logger(__name__)


@torch.inference_mode()
def run_benchmark(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    num_iters: int,
    benchmark_mode: str,
    device: str = "cuda",
) -> float:
    """Return latency (seconds) for given num_tokens."""

    if kv_cache_dtype == "fp8" and head_size % 16:
        raise ValueError("fp8 kv-cache requires head_size to be a multiple of 16.")

    set_random_seed(42)
    torch.set_default_device(device)

    # create random key / value tensors [T, H, D].
    key = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device=device)
    value = torch.randn_like(key)

    # prepare the slot mapping.
    # each token is assigned a unique slot in the KV-cache.
    num_slots = block_size * num_blocks
    if num_tokens > num_slots:
        raise ValueError("num_tokens cannot exceed the total number of cache slots")
    slot_mapping_lst = random.sample(range(num_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long, device=device)

    key_caches, value_caches = create_kv_caches_with_random(
        num_blocks,
        block_size,
        1,  # num_layers
        num_heads,
        head_size,
        kv_cache_dtype,
        dtype,
        device=device,
    )
    key_cache, value_cache = key_caches[0], value_caches[0]
    # to free unused memory
    del key_caches, value_caches

    # compute per-kernel scaling factors for fp8 conversion (if used).
    k_scale = (key.amax() / 64.0).to(torch.float32)
    v_scale = (value.amax() / 64.0).to(torch.float32)

    function_under_test = lambda: ops.reshape_and_cache(
        key,  # noqa: F821
        value,  # noqa: F821
        key_cache,  # noqa: F821
        value_cache,  # noqa: F821
        slot_mapping,  # noqa: F821
        kv_cache_dtype,
        k_scale,
        v_scale,
    )

    if benchmark_mode == "cudagraph":
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            function_under_test()
        torch.cuda.synchronize()
        function_under_test = lambda: g.replay()

    def run_cuda_benchmark(n_iters: int) -> float:
        nonlocal key, value, key_cache, value_cache, slot_mapping
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iters):
            function_under_test()
            torch.cuda.synchronize()
        end = time.perf_counter()
        return (end - start) / n_iters

    # warm-up
    run_cuda_benchmark(3)

    lat = run_cuda_benchmark(num_iters)

    # free tensors to mitigate OOM when sweeping
    del key, value, key_cache, value_cache, slot_mapping
    torch.cuda.empty_cache()

    return lat


def main(args):
    rows = []
    for exp in range(1, 17):
        n_tok = 2**exp
        lat = run_benchmark(
            num_tokens=n_tok,
            num_heads=args.num_heads,
            head_size=args.head_size,
            block_size=args.block_size,
            num_blocks=args.num_blocks,
            dtype=STR_DTYPE_TO_TORCH_DTYPE[args.dtype],
            kv_cache_dtype=args.kv_cache_dtype,
            num_iters=args.iters,
            benchmark_mode=args.mode,
            device="cuda",
        )
        rows.append([n_tok, lat * 1e6])  # convert to microseconds

    print(f"Benchmark results for implementation cuda (measuring with {args.mode}):")
    print(tabulate(rows, headers=["num_tokens", "latency (µs)"], floatfmt=".3f"))


if __name__ == "__main__":
    parser = FlexibleArgumentParser()

    parser.add_argument("--num-heads", type=int, default=128)
    parser.add_argument(
        "--head-size",
        type=int,
        choices=[64, 80, 96, 112, 120, 128, 192, 256],
        default=128,
    )
    parser.add_argument("--block-size", type=int, choices=[16, 32], default=16)
    parser.add_argument("--num-blocks", type=int, default=128 * 128)

    parser.add_argument(
        "--dtype",
        type=str,
        choices=["half", "bfloat16", "float"],
        default="bfloat16",
    )

    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=["auto", "fp8"],
        default="auto",
    )

    parser.add_argument("--iters", type=int, default=200)

    parser.add_argument(
        "--mode",
        type=str,
        choices=["cudagraph", "no_graph"],
        default="cudagraph",
    )

    args = parser.parse_args()

    main(args)
