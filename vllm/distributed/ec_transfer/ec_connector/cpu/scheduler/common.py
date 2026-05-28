# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared FIFO eviction helpers."""

import threading

from vllm.distributed.ec_transfer.ec_connector.ec_shared_region import (
    AllocationError,
    ECSharedRegion,
)


def producer_fifo_alloc(
    n_blocks: int,
    region: ECSharedRegion,
    enc: dict[str, list[int]],
    lock: threading.Lock,
) -> list[int]:
    """FIFO alloc for producer: skips pinned entries, holds lock during eviction."""
    try:
        return region.alloc(n_blocks)
    except AllocationError:
        pass
    with lock:
        result = evict_and_alloc(n_blocks, enc, region, skip_pinned=True)
    if result is not None:
        return result
    raise AllocationError(
        f"ECSharedRegion exhausted: cannot satisfy {n_blocks} blocks "
        f"even after evicting all unpinned encodings"
    )


def consumer_fifo_alloc(
    n_blocks: int,
    region: ECSharedRegion,
    loaded: dict[str, list[int]],
    pending: set[str],
) -> list[int]:
    """FIFO alloc for consumer: skips protected (pending-reload) entries."""
    try:
        return region.alloc(n_blocks)
    except AllocationError:
        pass
    result = evict_and_alloc(n_blocks, loaded, region, protected=pending)
    if result is not None:
        return result
    raise AllocationError(
        f"ECSharedRegion exhausted: cannot satisfy {n_blocks} blocks "
        f"even after evicting all evictable cache entries"
    )


def evict_and_alloc(
    n_blocks: int,
    cache: dict[str, list[int]],
    region: ECSharedRegion,
    *,
    skip_pinned: bool = False,
    protected: set[str] | None = None,
) -> list[int] | None:
    """Evict `cache` entries in insertion order until `alloc` succeeds.

    skip_pinned=True uses try_free (producer path; caller must hold lock).
    protected entries are skipped — their blocks are promised this step.
    Returns allocated block list, or None if exhausted.
    """
    for mm_hash in list(cache.keys()):
        if protected is not None and mm_hash in protected:
            continue
        indices = cache[mm_hash]
        if skip_pinned:
            if not region.try_free(indices):
                continue
        else:
            region.free(indices)
        del cache[mm_hash]
        try:
            return region.alloc(n_blocks)
        except AllocationError:
            continue
    return None
