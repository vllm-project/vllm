# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared FIFO eviction helper."""

from vllm.distributed.ec_transfer.ec_connector.cpu.ec_shared_region import (
    AllocationError,
    ECSharedRegion,
)


def evict_and_alloc(
    n_blocks: int,
    cache: dict[str, None],
    blocks: dict[str, list[int]],
    region: ECSharedRegion,
    *,
    skip_pinned: bool = False,
) -> list[int] | None:
    """Evict `cache` entries in insertion order until `alloc` succeeds.

    ``cache`` is the ordered set of cached mm_hashes (``dict[str, None]``);
    ``blocks`` maps each mm_hash to its allocated block indices.

    ``skip_pinned=True`` uses ``try_free`` so that blocks held by an active
    NIXL READ pin or a ``_pending_reload`` pin are transparently skipped.
    Caller must hold the shared lock when calling this function.
    Returns allocated block list, or None if all candidates were exhausted.
    """
    for mm_hash in list(cache.keys()):
        indices = blocks[mm_hash]
        if skip_pinned:
            if not region.try_free(indices):
                continue
        else:
            region.free(indices)
        del cache[mm_hash]
        del blocks[mm_hash]
        try:
            return region.alloc(n_blocks)
        except AllocationError:
            continue
    return None
