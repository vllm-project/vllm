# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import random
import time

import torch
from tabulate import tabulate

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    FlexibleArgumentParser,
    create_kv_caches_with_random_flash,
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
    kv_cache_layout: str,
    num_iters: int,
    device: str = "cuda",
) -> float:
    """Return latency (seconds) for given num_tokens."""

    if kv_cache_dtype == "fp8" and head_size % 16:
        raise ValueError("fp8 kv-cache requires head_size to be a multiple of 16.")

    current_platform.seed_everything(42)
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

    key_caches, value_caches = create_kv_caches_with_random_flash(
        num_blocks,
        block_size,
        1,  # num_layers
        num_heads,
        head_size,
        kv_cache_dtype,
        dtype,
        device=device,
        cache_layout=kv_cache_layout,
    )
    key_cache, value_cache = key_caches[0], value_caches[0]

    # compute per-kernel scaling factors for fp8 conversion (if used).
    k_scale = (key.amax() / 64.0).to(torch.float32)
    v_scale = (value.amax() / 64.0).to(torch.float32)

    def run_cuda_benchmark(n_iters: int) -> float:
        nonlocal key, value, key_cache, value_cache, slot_mapping
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iters):
            ops.reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                kv_cache_dtype,
                k_scale,
                v_scale,
            )
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
    for layout in ["NHD", "HND"]:
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
                kv_cache_layout=layout,
                num_iters=args.iters,
                device="cuda",
            )
            rows.append([n_tok, layout, f"{lat * 1e6:.3f}"])

    print(tabulate(rows, headers=["num_tokens", "layout", "latency (µs)"]))


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
    parser.add_argument("--num-blocks", type=int, default=128 * 512)

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

    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    main(args)
