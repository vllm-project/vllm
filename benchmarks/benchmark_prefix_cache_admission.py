# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark request-level prefix-cache write admission.

The workload mixes reusable hot prompts with one-off RAG-style prompts. Every
one-off prompt can reuse a shared prefix, but its suffix is unique. The
benchmark compares admitting those suffixes to the prefix cache against using
``skip_writing_prefix_cache`` for the one-off requests.

This is a CPU-only cache-policy simulation: saved prefill tokens are reported
instead of model-dependent wall-clock latency.
"""

import json
from collections.abc import Callable
from typing import Any

import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request


def make_manager(block_size: int, num_cache_blocks: int) -> KVCacheManager:
    cache_config = KVCacheConfig(
        num_blocks=num_cache_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    return KVCacheManager(
        cache_config,
        max_model_len=1 << 30,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
        enable_caching=True,
    )


def make_request(
    request_id: str,
    token_ids: list[int],
    block_size: int,
    hash_fn: Callable[..., bytes],
    skip_writing_prefix_cache: bool,
) -> Request:
    extra_args: dict[str, Any] | None = None
    if skip_writing_prefix_cache:
        extra_args = {"skip_writing_prefix_cache": True}
    return Request(
        request_id=request_id,
        prompt_token_ids=token_ids,
        sampling_params=SamplingParams(max_tokens=1, extra_args=extra_args),
        pooling_params=None,
        block_hasher=get_request_block_hasher(block_size, hash_fn),
    )


def run_request(manager: KVCacheManager, request: Request) -> int:
    computed_blocks, num_computed_tokens, _ = manager.get_computed_blocks(request)
    new_blocks = manager.allocate_slots(
        request,
        request.num_tokens - num_computed_tokens,
        num_new_computed_tokens=num_computed_tokens,
        new_computed_blocks=computed_blocks,
    )
    if new_blocks is None:
        raise RuntimeError("The synthetic request could not allocate KV blocks")
    manager.free(request)
    return num_computed_tokens


def block_tokens(namespace: int, num_blocks: int, block_size: int) -> list[int]:
    start = namespace * num_blocks * block_size
    return list(range(start, start + num_blocks * block_size))


def run_workload(args, skip_cold_writes: bool) -> dict[str, int | float | str]:
    manager = make_manager(args.block_size, args.num_cache_blocks)
    common = block_tokens(1, args.common_prefix_blocks, args.block_size)

    hot_prompts = []
    hot_suffix_blocks = args.hot_prefix_blocks - args.common_prefix_blocks
    for hot_idx in range(args.num_hot_prefixes):
        suffix = block_tokens(10 + hot_idx, hot_suffix_blocks, args.block_size)
        hot_prompts.append(common + suffix + [-(hot_idx + 1)])

    for hot_idx, prompt in enumerate(hot_prompts):
        run_request(
            manager,
            make_request(
                f"seed-{hot_idx}",
                prompt,
                args.block_size,
                sha256,
                skip_writing_prefix_cache=False,
            ),
        )

    hot_hit_tokens = 0
    cold_hit_tokens = 0
    for cycle in range(args.num_cycles):
        cold_suffix = block_tokens(
            1000 + cycle,
            args.cold_suffix_blocks,
            args.block_size,
        )
        cold_hit_tokens += run_request(
            manager,
            make_request(
                f"cold-{cycle}",
                common + cold_suffix + [-(1000 + cycle)],
                args.block_size,
                sha256,
                skip_writing_prefix_cache=skip_cold_writes,
            ),
        )

        hot_idx = cycle % args.num_hot_prefixes
        hot_hit_tokens += run_request(
            manager,
            make_request(
                f"hot-{cycle}",
                hot_prompts[hot_idx],
                args.block_size,
                sha256,
                skip_writing_prefix_cache=False,
            ),
        )

    hot_query_tokens = args.num_cycles * args.hot_prefix_blocks * args.block_size
    cold_query_tokens = (
        args.num_cycles
        * (args.common_prefix_blocks + args.cold_suffix_blocks)
        * args.block_size
    )
    return {
        "policy": "no-store" if skip_cold_writes else "admit-all",
        "hot_hit_tokens": hot_hit_tokens,
        "hot_query_tokens": hot_query_tokens,
        "hot_hit_rate": hot_hit_tokens / hot_query_tokens,
        "hot_recomputed_tokens": hot_query_tokens - hot_hit_tokens,
        "cold_hit_rate": cold_hit_tokens / cold_query_tokens,
        "cached_hashes": len(manager.block_pool.cached_block_hash_to_block),
    }


def validate_args(args) -> None:
    if args.common_prefix_blocks >= args.hot_prefix_blocks:
        raise ValueError("common-prefix-blocks must be smaller than hot-prefix-blocks")
    hot_cache_blocks = args.common_prefix_blocks + args.num_hot_prefixes * (
        args.hot_prefix_blocks - args.common_prefix_blocks
    )
    usable_blocks = args.num_cache_blocks - 1
    if hot_cache_blocks + args.cold_suffix_blocks > usable_blocks:
        raise ValueError(
            "The active cold request does not fit beside the seeded hot cache; "
            "increase --num-cache-blocks so the benchmark measures admission "
            "pollution rather than unavoidable active-request pressure."
        )


def print_results(results: list[dict[str, int | float | str]]) -> None:
    print(
        f"{'policy':<12} {'hot hit rate':>12} {'hot recompute':>15} "
        f"{'cold hit rate':>14} {'cached hashes':>15}"
    )
    for result in results:
        print(
            f"{result['policy']:<12} "
            f"{result['hot_hit_rate']:>11.1%} "
            f"{result['hot_recomputed_tokens']:>15,} "
            f"{result['cold_hit_rate']:>13.1%} "
            f"{result['cached_hashes']:>15,}"
        )

    baseline, no_store = results
    saved = int(baseline["hot_recomputed_tokens"]) - int(
        no_store["hot_recomputed_tokens"]
    )
    print(f"\nHot-prefix prefill tokens saved by no-store: {saved:,}")


def parse_args():
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-cache-blocks", type=int, default=160)
    parser.add_argument("--num-hot-prefixes", type=int, default=8)
    parser.add_argument("--common-prefix-blocks", type=int, default=4)
    parser.add_argument("--hot-prefix-blocks", type=int, default=16)
    parser.add_argument("--cold-suffix-blocks", type=int, default=24)
    parser.add_argument("--num-cycles", type=int, default=64)
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    init_none_hash(sha256)
    results = [
        run_workload(args, skip_cold_writes=False),
        run_workload(args, skip_cold_writes=True),
    ]
    if args.json_output:
        print(json.dumps(results, indent=2))
    else:
        print_results(results)


if __name__ == "__main__":
    main()
