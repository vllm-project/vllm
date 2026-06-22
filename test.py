#!/usr/bin/env python3
"""White-box validation for prefix-chain fracture in vLLM KV cache.

This script does not start a model or allocate real GPU KV tensors. It tests
the KVCacheManager metadata path directly:

1. Warm up and cache a full prefix G + B + S.
2. Verify a clean victim request reuses all of G + B + S.
3. White-box evict the first B block from the prefix-cache hash map.
4. Verify the next victim request only reuses G, because lookup stops at B_1.
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path


if sys.version_info < (3, 10):
    raise SystemExit(
        "This vLLM checkout requires Python >= 3.10. "
        f"Current interpreter: {sys.version.split()[0]}"
    )

ROOT = Path(__file__).resolve().parent
VLLM_REPO = ROOT / "vllm"
if str(VLLM_REPO) not in sys.path:
    sys.path.insert(0, str(VLLM_REPO))

import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.request import Request


def register_full_attention_manager() -> None:
    KVCacheSpecRegistry.register(
        FullAttentionSpec,
        FullAttentionManager,
        uniform_type_base_spec=FullAttentionSpec,
    )


def make_manager(num_blocks: int, block_size: int) -> KVCacheManager:
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
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
        kv_cache_config=kv_cache_config,
        max_model_len=4096,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
        enable_caching=True,
        log_stats=True,
    )


def make_request(request_id: str, token_ids: list[int], block_size: int) -> Request:
    sampling_params = SamplingParams(max_tokens=1)
    return Request(
        request_id=request_id,
        prompt_token_ids=token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(block_size, sha256),
    )


def allocate_and_free(
    manager: KVCacheManager,
    request: Request,
    *,
    expected_hit_len: int | None = None,
) -> int:
    computed_blocks, hit_len = manager.get_computed_blocks(request)
    if expected_hit_len is not None:
        assert hit_len == expected_hit_len, (
            f"{request.request_id}: expected hit_len={expected_hit_len}, "
            f"got {hit_len}"
        )

    new_tokens = request.num_tokens - hit_len
    assert new_tokens > 0
    allocated = manager.allocate_slots(
        request,
        num_new_tokens=new_tokens,
        num_new_computed_tokens=hit_len,
        new_computed_blocks=computed_blocks,
    )
    assert allocated is not None
    manager.free(request)
    return hit_len


def block_ids_for_request(manager: KVCacheManager, request_id: str) -> list[int]:
    block_ids = manager.get_blocks(request_id).get_block_ids()
    assert len(block_ids) == 1
    return block_ids[0]


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--block-size",
        type=int,
        default=4,
        help="KV cache block size. Use 4 for a tiny mechanism test, 16 for a "
        "more vLLM-like setting.",
    )
    parser.add_argument(
        "--s-blocks",
        type=int,
        default=8,
        help="Number of full blocks in reusable suffix S.",
    )
    args = parser.parse_args()

    block_size = args.block_size
    register_full_attention_manager()
    init_none_hash(sha256)

    manager = make_manager(num_blocks=64, block_size=block_size)

    # Keep every component aligned to full block boundaries. q_* is one token,
    # so it is not a reusable full block and does not affect prefix reuse.
    G = list(range(0, 2 * block_size))  # 2 blocks
    B = list(range(10_000, 10_000 + 2 * block_size))  # 2 blocks
    S = list(range(20_000, 20_000 + args.s_blocks * block_size))
    prefix = G + B + S

    g_blocks = len(G) // block_size
    b1_logical_index = g_blocks
    expected_clean_hit = len(prefix)

    warm = make_request("warm", prefix + [9001], block_size)
    warm_hit = allocate_and_free(manager, warm, expected_hit_len=0)
    assert warm_hit == 0

    # Recreate the warm request to recover the original logical block IDs before
    # freeing it again. This is still a clean full-prefix hit.
    warm_probe = make_request("warm_probe", prefix + [9004], block_size)
    computed_blocks, warm_probe_hit = manager.get_computed_blocks(warm_probe)
    assert warm_probe_hit == expected_clean_hit
    manager.allocate_slots(
        warm_probe,
        num_new_tokens=warm_probe.num_tokens - warm_probe_hit,
        num_new_computed_tokens=warm_probe_hit,
        new_computed_blocks=computed_blocks,
    )
    warm_probe_block_ids = block_ids_for_request(manager, "warm_probe")
    b1_block_id = warm_probe_block_ids[b1_logical_index]
    manager.free(warm_probe)

    clean = make_request("clean", prefix + [9002], block_size)
    clean_hit = allocate_and_free(
        manager, clean, expected_hit_len=expected_clean_hit
    )

    manager.evict_blocks({b1_block_id})

    fractured = make_request("fractured", prefix + [9003], block_size)
    fractured_hit = allocate_and_free(
        manager, fractured, expected_hit_len=len(G)
    )

    print("prefix-chain fracture mechanism validated")
    print(f"block_size                 = {block_size}")
    print(f"blocks(G)                  = {g_blocks}")
    print(f"B_1 logical block index     = {b1_logical_index}")
    print(f"B_1 physical block id       = {b1_block_id}")
    print(f"clean victim hit length     = {clean_hit} tokens")
    print(f"fractured victim hit length = {fractured_hit} tokens")
    print(f"recomputed after fracture   = {clean_hit - fractured_hit} tokens")


if __name__ == "__main__":
    main()
