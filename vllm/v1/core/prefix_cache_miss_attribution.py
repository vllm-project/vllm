# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional per-request prefix-cache miss attribution.

Implements the diagnostic proposed in
https://github.com/vllm-project/vllm/issues/60044.

A short prefix-cache hit has several very different explanations: the prompt
changed, the block was evicted under memory pressure, or the lookup was capped
because vLLM always recomputes the last prompt token for logits. vLLM cannot
infer the cause, so this module only reports what it *observed* for the first
block after the longest hit:

* `evicted`: the block hash was cached and dropped from the cache recently.
* `no_match`: the block hash was probed but not found in the cache or in the
  bounded eviction history. Deliberately ambiguous: it also covers changed
  extra keys and evictions older than the history window.
* `lookup_limit`: no block was probed because the hit reached the lookup
  cap. The next block is recomputed by design, not missed.

The feature is disabled by default. When enabled it costs one bounded-set
lookup per request plus a fixed-size LRU of recently evicted block hashes.
Only names of extra hash keys are ever reported, never their values.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from vllm.v1.core.kv_cache_utils import (
    BlockHashWithGroupId,
    generate_block_hash_extra_keys,
    make_block_hash_with_group_id,
)

if TYPE_CHECKING:
    from vllm.v1.request import Request

MissReason = Literal["evicted", "no_match", "lookup_limit"]


@dataclass(frozen=True, slots=True)
class PrefixCacheMissAttribution:
    """What vLLM observed for the first uncached block of a request."""

    group_id: int
    reason: MissReason
    extra_key_kinds: tuple[str, ...]
    num_cached_tokens: int
    max_cache_hit_length: int
    block_index: int | None


class RecentEvictions:
    """Fixed-size LRU of `BlockHashWithGroupId` values evicted from a pool.

    Sized to `num_gpu_blocks` by default, i.e. roughly one full cache
    turnover. A hit in this set is definite evidence of an eviction; a miss is
    not evidence that no eviction ever happened.
    """

    def __init__(self, max_size: int) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        self._entries: OrderedDict[BlockHashWithGroupId, None] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, hashes: Iterable[BlockHashWithGroupId]) -> None:
        for block_hash in hashes:
            if block_hash in self._entries:
                self._entries.move_to_end(block_hash)
            else:
                self._entries[block_hash] = None
                if len(self._entries) > self.max_size:
                    self._entries.popitem(last=False)

    def __contains__(self, block_hash: BlockHashWithGroupId) -> bool:
        return block_hash in self._entries


def extra_key_kinds(
    request: Request, block_index: int, hash_block_size: int
) -> tuple[str, ...]:
    """Names of extra keys mixed into a block's hash, without their values.

    Reuses the production hash helper so the reported kinds cannot drift from
    what actually went into the hash. Salt values, adapter paths, multimodal
    identifiers and prompt-embedding digests are all discarded here.
    """
    start = block_index * hash_block_size
    extra_keys, _ = generate_block_hash_extra_keys(
        request, start, start + hash_block_size, 0
    )
    if not extra_keys:
        return ()
    return tuple(
        sorted(
            {
                str(key[0])
                for key in extra_keys
                if isinstance(key, tuple) and key and isinstance(key[0], str)
            }
        )
    )


def attribute_miss(
    request: Request,
    *,
    group_id: int,
    num_cached_tokens: int,
    max_cache_hit_length: int,
    hash_block_size: int,
    block_size: int,
    recent_evictions: RecentEvictions | None,
) -> PrefixCacheMissAttribution | None:
    """Classify the first uncached block after the longest cache hit.

    Returns None when the group's layout is outside the supported scope
    (fine-grained/DCP hashing), because the index arithmetic below only holds
    when one probed hash corresponds to exactly one cache block.
    """
    if block_size != hash_block_size:
        return None

    first_miss_idx = num_cached_tokens // block_size
    probed_blocks = max_cache_hit_length // block_size
    kinds = (
        extra_key_kinds(request, first_miss_idx, hash_block_size)
        if first_miss_idx < len(request.block_hashes)
        else ()
    )

    if first_miss_idx >= probed_blocks:
        # The next block was never probed: recomputed for logits by design.
        reason: MissReason = "lookup_limit"
        block_index: int | None = None
    elif first_miss_idx >= len(request.block_hashes):
        # No complete block hash exists past the hit (partial prompt tail).
        reason = "lookup_limit"
        block_index = None
    else:
        block_index = first_miss_idx
        key = make_block_hash_with_group_id(
            request.block_hashes[first_miss_idx], group_id
        )
        reason = (
            "evicted"
            if recent_evictions is not None and key in recent_evictions
            else "no_match"
        )

    return PrefixCacheMissAttribution(
        group_id=group_id,
        reason=reason,
        extra_key_kinds=kinds,
        num_cached_tokens=num_cached_tokens,
        max_cache_hit_length=max_cache_hit_length,
        block_index=block_index,
    )
