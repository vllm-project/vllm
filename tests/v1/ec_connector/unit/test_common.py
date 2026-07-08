# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FIFO evict/alloc coverage for the unified EC CPU scheduler.

The scheduler's ``_fifo_alloc`` (and the NIXL reload path) both reclaim
blocks through the module-private ``_evict_and_alloc`` helper. These tests
exercise that helper directly against a real ``ECSharedRegion``.
"""

import uuid

from vllm.distributed.ec_transfer.ec_connector.cpu.ec_shared_region import (
    ECSharedRegion,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler import (
    _evict_and_alloc as evict_and_alloc,
)

_N = 8
_BS = 64


def _region() -> ECSharedRegion:
    return ECSharedRegion(
        instance_id=str(uuid.uuid4()), num_blocks=_N, block_size_bytes=_BS
    )


def test_evict_and_alloc_evicts_oldest_unpinned():
    r = _region()
    old = r.alloc(4)
    new = r.alloc(4)
    cache: dict[str, None] = {"old": None, "new": None}
    blocks = {"old": old, "new": new}
    result = evict_and_alloc(3, cache, blocks, r, skip_pinned=False)
    assert result is not None and len(result) == 3
    assert "old" not in cache
    assert "new" in cache
    r.cleanup()


def test_evict_and_alloc_skips_pinned():
    r = _region()
    pinned = r.alloc(4)
    unpinned = r.alloc(4)
    r.pin(pinned)
    cache: dict[str, None] = {"pinned": None, "unpinned": None}
    blocks = {"pinned": pinned, "unpinned": unpinned}
    result = evict_and_alloc(3, cache, blocks, r, skip_pinned=True)
    assert result is not None and len(result) == 3
    assert "pinned" in cache
    assert "unpinned" not in cache
    r.unpin(pinned)
    r.cleanup()


def test_evict_and_alloc_returns_none_when_all_pinned():
    r = _region()
    all_idx = r.alloc(_N)
    r.pin(all_idx)
    cache: dict[str, None] = {"only": None}
    blocks = {"only": all_idx}
    result = evict_and_alloc(1, cache, blocks, r, skip_pinned=True)
    assert result is None
    assert "only" in cache
    r.unpin(all_idx)
    r.cleanup()


def test_evict_and_alloc_evicts_unprotected_keeps_pinned():
    r = _region()
    cold = r.alloc(_N // 2)
    hot = r.alloc(_N // 2)
    r.pin(hot)
    cache: dict[str, None] = {"cold": None, "hot": None}
    blocks = {"cold": cold, "hot": hot}
    result = evict_and_alloc(1, cache, blocks, r, skip_pinned=True)
    assert result is not None and len(result) == 1
    assert "hot" in cache
    assert "cold" not in cache
    r.unpin(hot)
    r.cleanup()
