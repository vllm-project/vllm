# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Read-only single-full-attention-group risk prototype; NOT a scheduler policy.

Demand is full remaining prompt allocation, not this step's allocation.
No prediction of future frees, decode growth or other running requests is made.
A risk result permits considering reordering; it must NOT reject admission.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Risk:
    reason: str
    scanned: int
    selected: int
    first_cached_block: int | None = None


def full_prompt_new_blocks(prompt_tokens, block_size, covered_blocks=0):
    """covered_blocks: distinct owned/hit blocks, counted once in ONE group."""
    if prompt_tokens < 0 or block_size <= 0 or covered_blocks < 0:
        raise ValueError("invalid block demand inputs")
    return max(0, (prompt_tokens + block_size - 1) // block_size - covered_blocks)


def inspect_risk(queue, new_blocks, claimed_hit_ids=frozenset(), scan_limit=512):
    """Skip free hits that touch() would remove before fresh allocation.

    Never call popleft/touch or modify metadata. Scan-limit is UNKNOWN, not safe.
    block_hash is used only for this baseline's full-block caching scope.
    """
    if new_blocks < 0 or scan_limit <= 0:
        raise ValueError("invalid scan inputs")
    node = queue.fake_free_list_head.next_free_block
    scanned = selected = 0
    while selected < new_blocks:
        if node is queue.fake_free_list_tail or node is None:
            return Risk("insufficient_current_free", scanned, selected)
        if scanned >= scan_limit:
            return Risk("unknown_scan_limit", scanned, selected)
        scanned += 1
        if node.block_id not in claimed_hit_ids:
            selected += 1
            if node.block_hash is not None:
                return Risk("cached_block_exposed", scanned, selected, node.block_id)
        node = node.next_free_block
    return Risk("no_exposure_in_snapshot", scanned, selected)


def self_test():
    from vllm.v1.core.kv_cache_utils import (
        BlockHash,
        FreeKVCacheBlockQueue,
        KVCacheBlock,
        make_block_hash_with_group_id,
    )

    def make(flags):
        blocks = [KVCacheBlock(block_id=i) for i in range(len(flags))]
        for b, cached in zip(blocks, flags):
            if cached:
                b.set_block_hash(
                    make_block_hash_with_group_id(
                        BlockHash(b"prototype-" + str(b.block_id).encode()), 0
                    )
                )
        return FreeKVCacheBlockQueue(blocks), blocks

    def snapshot(q, bs):
        def bid(b):
            return None if b is None else b.block_id

        return (
            q.num_free_blocks,
            bid(q.fake_free_list_head.next_free_block),
            bid(q.fake_free_list_tail.prev_free_block),
            tuple(
                (
                    b.block_id,
                    b.ref_cnt,
                    b.block_hash,
                    bid(b.prev_free_block),
                    bid(b.next_free_block),
                )
                for b in bs
            ),
        )

    cases = [
        ("current chunk fits", [0, 0, 1, 1], 2, set(), 512, "no_exposure_in_snapshot"),
        (
            "full prompt exposes cache",
            [0, 0, 1, 1],
            3,
            set(),
            512,
            "cached_block_exposed",
        ),
        ("own hit excluded", [1, 0, 0], 2, {0}, 512, "no_exposure_in_snapshot"),
        (
            "other cached block still exposed",
            [1, 0, 1],
            2,
            {0},
            512,
            "cached_block_exposed",
        ),
        ("no fresh blocks needed", [1], 0, set(), 512, "no_exposure_in_snapshot"),
        ("scan capped", [0, 0, 0], 3, set(), 2, "unknown_scan_limit"),
        ("free capacity insufficient", [0], 2, set(), 512, "insufficient_current_free"),
        ("cached at queue head", [1, 0, 0], 1, set(), 512, "cached_block_exposed"),
    ]
    for label, flags, demand, hits, limit, expected in cases:
        q, bs = make(flags)
        before = snapshot(q, bs)
        result = inspect_risk(q, demand, hits, limit)
        assert result.reason == expected, (label, result)
        assert snapshot(q, bs) == before, "queue/refcount/hash mutated"
        print(f"PASS: {label}: {result.reason}")
    assert full_prompt_new_blocks(2176, 16) == 136
    assert full_prompt_new_blocks(2176, 16, 128) == 8
    assert full_prompt_new_blocks(2177, 16, 128) == 9
    print("PASS: full prompt demand and partial last block")
    print("ALL PASS: prototype only; NOT integrated or benchmarked.")


if __name__ == "__main__":
    self_test()
