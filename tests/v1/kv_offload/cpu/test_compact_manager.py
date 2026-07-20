# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused tests for compact layout support in CPUOffloadingManager.

Extends the canonical CPUOffloadingManager with a FixedPageAllocator-based
compact path.  Verifies allocation, eviction, load, store, and lifecycle
behaviour, plus legacy regression and policy guards.

No global mocks, conftest, shared TP1, bounded-tail, ExtentAllocator,
compact_layout, scheduler, or lifecycle dependencies beyond the code under
test.
"""

import numpy as np
import pytest

from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadKey,
    ReqContext,
    make_offload_key,
)
from vllm.v1.kv_offload.config import (
    CompactGroupSliceConfig,
    CompactSliceConfig,
)
from vllm.v1.kv_offload.cpu.common import (
    CompactCPUAddress,
    CompactCPULoadStoreSpec,
    CPULoadStoreSpec,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _req_ctx(req_id: str = "") -> ReqContext:
    return ReqContext(req_id=req_id)


_EMPTY_REQ = _req_ctx()


def _key(int_hash: int, group_idx: int = 0) -> OffloadKey:
    return make_offload_key(str(int_hash).encode(), group_idx)


def _keys(int_hashes: list[int], group_idx: int = 0) -> list[OffloadKey]:
    return [_key(h, group_idx) for h in int_hashes]


def _single_group_cfg(
    group_idx: int = 0,
    real_bytes: int = 80,
    blocks_per_chunk: int = 1,
) -> tuple[CompactGroupSliceConfig, ...]:
    """One group with one slice."""
    return (
        CompactGroupSliceConfig(
            group_idx=group_idx,
            slices=(
                CompactSliceConfig(
                    offset_bytes=0,
                    real_bytes_per_gpu_block=real_bytes,
                    padded_bytes_per_gpu_block=real_bytes,
                    layer_name="k",
                ),
            ),
            compact_real_bytes_per_rank=real_bytes,
            compact_padded_bytes_per_rank=real_bytes,
        ),
    )


def _two_group_cfg(
    real_bytes_0: int = 80,
    real_bytes_1: int = 120,
    blocks_per_chunk: int = 1,
) -> tuple[CompactGroupSliceConfig, ...]:
    """Two groups, each with one slice."""
    return (
        CompactGroupSliceConfig(
            group_idx=0,
            slices=(
                CompactSliceConfig(
                    offset_bytes=0,
                    real_bytes_per_gpu_block=real_bytes_0,
                    padded_bytes_per_gpu_block=real_bytes_0,
                    layer_name="k",
                ),
            ),
            compact_real_bytes_per_rank=real_bytes_0,
            compact_padded_bytes_per_rank=real_bytes_0,
        ),
        CompactGroupSliceConfig(
            group_idx=1,
            slices=(
                CompactSliceConfig(
                    offset_bytes=0,
                    real_bytes_per_gpu_block=real_bytes_1,
                    padded_bytes_per_gpu_block=real_bytes_1,
                    layer_name="v",
                ),
            ),
            compact_real_bytes_per_rank=real_bytes_1,
            compact_padded_bytes_per_rank=real_bytes_1,
        ),
    )


def make_compact_manager(
    cpu_budget: int = 4096,
    page_size: int = 256,
    cache_policy: str = "lru",
    blocks_per_chunk: int = 1,
    group_slice_configs: tuple[CompactGroupSliceConfig, ...] | None = None,
    num_blocks: int = 4,
) -> CPUOffloadingManager:
    if group_slice_configs is None:
        group_slice_configs = _single_group_cfg(real_bytes=80)
    return CPUOffloadingManager(
        num_blocks=num_blocks,
        cache_policy=cache_policy,
        compact_group_slice_configs=group_slice_configs,
        blocks_per_chunk=blocks_per_chunk,
        compact_cpu_budget_bytes=cpu_budget,
        compact_page_size=page_size,
    )


def make_legacy_manager(
    num_blocks: int = 4, cache_policy: str = "lru"
) -> CPUOffloadingManager:
    return CPUOffloadingManager(num_blocks=num_blocks, cache_policy=cache_policy)


# ---------------------------------------------------------------------------
# 0. Init guards
# ---------------------------------------------------------------------------


def test_compact_partial_args_raises() -> None:
    """Partial compact args must raise."""
    with pytest.raises(ValueError, match="all four compact args"):
        CPUOffloadingManager(
            num_blocks=4,
            compact_group_slice_configs=_single_group_cfg(),
            blocks_per_chunk=1,
            # missing compact_cpu_budget_bytes, compact_page_size
        )


def test_compact_supported_policy_accepted() -> None:
    """LRU and ARC policies are accepted with compact layout."""
    for policy in ("lru", "arc"):
        mgr = CPUOffloadingManager(
            num_blocks=4,
            cache_policy=policy,
            compact_group_slice_configs=_single_group_cfg(),
            blocks_per_chunk=1,
            compact_cpu_budget_bytes=1024,
            compact_page_size=256,
        )
        assert mgr._compact_enabled is True


def test_compact_negative_payload_raises() -> None:
    """Zero or negative compact_real_bytes_per_rank must raise."""
    cfg = (
        CompactGroupSliceConfig(
            group_idx=0,
            slices=(
                CompactSliceConfig(
                    offset_bytes=0,
                    real_bytes_per_gpu_block=0,
                    padded_bytes_per_gpu_block=0,
                    layer_name="k",
                ),
            ),
            compact_real_bytes_per_rank=0,
            compact_padded_bytes_per_rank=0,
        ),
    )
    with pytest.raises(ValueError, match="non-positive"):
        make_compact_manager(cpu_budget=1024, page_size=256, group_slice_configs=cfg)


def test_compact_duplicate_group_idx_raises() -> None:
    """Duplicate group index must raise."""
    cfg = (
        CompactGroupSliceConfig(
            group_idx=0,
            slices=(
                CompactSliceConfig(
                    offset_bytes=0,
                    real_bytes_per_gpu_block=80,
                    padded_bytes_per_gpu_block=80,
                    layer_name="k",
                ),
            ),
            compact_real_bytes_per_rank=80,
            compact_padded_bytes_per_rank=80,
        ),
        CompactGroupSliceConfig(
            group_idx=0,  # duplicate
            slices=(
                CompactSliceConfig(
                    offset_bytes=0,
                    real_bytes_per_gpu_block=80,
                    padded_bytes_per_gpu_block=80,
                    layer_name="v",
                ),
            ),
            compact_real_bytes_per_rank=80,
            compact_padded_bytes_per_rank=80,
        ),
    )
    with pytest.raises(ValueError, match="duplicate"):
        make_compact_manager(cpu_budget=1024, page_size=256, group_slice_configs=cfg)


# ---------------------------------------------------------------------------
# 1. Non-contiguous pages via sentinel pattern
# ---------------------------------------------------------------------------


def test_compact_non_contiguous_pages() -> None:
    """Allocate single-page sentinels, free alternating pages, then request
    a multi-page payload.  Assert len(physical_spans) > 1, sum logical bytes
    exact, offsets distinct."""
    page_size = 256
    payload = 700  # needs 3 pages (700/256 = 2.73 -> 3)
    manager = make_compact_manager(
        cpu_budget=8 * page_size,  # 8 pages
        page_size=page_size,
        blocks_per_chunk=1,
        group_slice_configs=_single_group_cfg(
            real_bytes=payload,
            blocks_per_chunk=1,
        ),
    )

    # Allocate 8 single-page sentinels via prepare_store
    allocator = manager._compact_allocator
    assert allocator is not None
    # Manually allocate sentinel pages to create fragmentation
    sentinels = [allocator.allocate(page_size) for _ in range(8)]
    assert all(s is not None for s in sentinels)
    # Free alternating sentinels (indices 1, 3, 5, 7)
    for idx in (1, 3, 5, 7):
        allocator.free(sentinels[idx])

    # Now prepare_store for a key should allocate from fragmented free pages
    key = _key(1, group_idx=0)
    result = manager.prepare_store([key], _EMPTY_REQ)
    assert result is not None, "prepare_store returned None"
    assert len(result.keys_to_store) == 1

    # The returned store_spec must be CompactCPULoadStoreSpec with addresses
    spec = result.store_spec
    assert isinstance(spec, CompactCPULoadStoreSpec), (
        f"Expected CompactCPULoadStoreSpec, got {type(spec).__name__}"
    )
    assert len(spec.compact_addresses) == 1
    addr = spec.compact_addresses[0]

    # Assert non-contiguous physical spans
    assert len(addr.physical_spans) > 1, (
        f"Expected fragmented spans > 1, got {len(addr.physical_spans)}"
    )
    # Sum of logical bytes must equal the payload
    assert sum(span.logical_length for span in addr.physical_spans) == payload
    # Each span offset must be distinct
    offsets = [span.byte_offset for span in addr.physical_spans]
    assert len(set(offsets)) == len(offsets), "offsets must be distinct"

    # Verify the address has physical_spans matching expected pattern
    # (1, 3, 5) -> bytes [256, 512, 1280)
    # page 1 -> offset 256, page 3 -> offset 768, page 5 -> offset 1280
    # Payload = 700 across 3 pages
    assert addr.byte_offset == 256  # first span starts at page 1

    # Verify the address is usable by plan_compact_transfer
    from vllm.v1.kv_offload.cpu.compact_transfer import plan_compact_transfer

    plan = plan_compact_transfer(
        gpu_base_ptr=0x1000,
        gpu_row_stride=payload,
        cpu_base_ptr=0xA000,
        cpu_region_size=8 * page_size,
        gpu_block_ids=np.array([0], dtype=np.int64),
        group_sizes=[1],
        block_indices=[0],
        compact_addresses=[addr],
        group_slice_configs=_single_group_cfg(
            real_bytes=payload,
            blocks_per_chunk=1,
        ),
        block_size_factor=1,
    )
    assert plan is not None
    assert plan.num_cpu_addresses == 1
    # The plan should consume the pre-completion store address
    assert sum(plan.sizes) == payload


# ---------------------------------------------------------------------------
# 2. Mixed groups sharing one allocator
# ---------------------------------------------------------------------------


def test_mixed_groups_non_overlapping_addresses() -> None:
    """Two groups sharing one global FixedPageAllocator produce non-overlapping
    byte offsets."""
    page_size = 256
    payload_0 = 300  # group 0: 2 pages
    payload_1 = 500  # group 1: 2 pages
    manager = make_compact_manager(
        cpu_budget=8 * page_size,
        page_size=page_size,
        blocks_per_chunk=1,
        group_slice_configs=_two_group_cfg(
            real_bytes_0=payload_0,
            real_bytes_1=payload_1,
        ),
    )

    key0 = _key(1, group_idx=0)
    key1 = _key(2, group_idx=1)

    # Store both groups in the same batch
    result = manager.prepare_store([key0, key1], _EMPTY_REQ)
    assert result is not None
    spec = result.store_spec
    assert isinstance(spec, CompactCPULoadStoreSpec)
    assert len(spec.compact_addresses) == 2

    addrs = spec.compact_addresses
    # Group index must match
    assert addrs[0].group_idx == 0
    assert addrs[1].group_idx == 1

    # Byte offsets must be non-overlapping
    def _byte_ranges(addr: CompactCPUAddress) -> set[tuple[int, int]]:
        ranges: set[tuple[int, int]] = set()
        for span in addr.physical_spans:
            ranges.add((span.byte_offset, span.byte_offset + span.allocated_length))
        return ranges

    ranges0 = _byte_ranges(addrs[0])
    ranges1 = _byte_ranges(addrs[1])
    for start0, end0 in ranges0:
        for start1, end1 in ranges1:
            # No overlap
            assert end0 <= start1 or end1 <= start0, (
                f"Overlapping ranges: group 0 [{start0},{end0}) "
                f"and group 1 [{start1},{end1})"
            )


# ---------------------------------------------------------------------------
# 3. blocks_per_chunk exactly once
# ---------------------------------------------------------------------------


def test_blocks_per_chunk_exactly_once() -> None:
    """blocks_per_chunk=4 means one compact address covers 4 GPU blocks
    worth of payload."""
    page_size = 256
    real_bytes = 80
    bpc = 4
    payload = real_bytes * bpc  # 320 bytes
    manager = make_compact_manager(
        cpu_budget=page_size * 4,
        page_size=page_size,
        blocks_per_chunk=bpc,
        group_slice_configs=_single_group_cfg(
            real_bytes=real_bytes,
            blocks_per_chunk=bpc,
        ),
    )

    key = _key(1, group_idx=0)
    result = manager.prepare_store([key], _EMPTY_REQ)
    assert result is not None
    spec = result.store_spec
    assert isinstance(spec, CompactCPULoadStoreSpec)
    assert len(spec.compact_addresses) == 1
    addr = spec.compact_addresses[0]
    # Payload must be real_bytes * blocks_per_chunk
    assert addr.logical_length == payload, (
        f"Expected {payload}, got {addr.logical_length}"
    )
    # The address's allocated_length covers at least logical_length
    assert addr.allocated_length >= addr.logical_length

    # Two separate keys should get separate addresses
    key2 = _key(2, group_idx=0)
    result2 = manager.prepare_store([key2], _EMPTY_REQ)
    assert result2 is not None
    spec2 = result2.store_spec
    assert isinstance(spec2, CompactCPULoadStoreSpec)
    addr2 = spec2.compact_addresses[0]
    assert addr2.byte_offset != addr.byte_offset


# ---------------------------------------------------------------------------
# 4. Batch aggregate fail / no mutation
# ---------------------------------------------------------------------------


def test_batch_allocation_fail_no_mutation() -> None:
    """When the batch does not fit (even after eviction), no allocations
    are made and no addresses are registered.

    Also verifies that when eviction *does* free space, the new key is
    accepted and the evicted key is removed from committed tracking.
    """
    page_size = 256
    payload = 200  # 1 page
    manager = make_compact_manager(
        cpu_budget=2 * page_size,  # only 2 pages
        page_size=page_size,
        blocks_per_chunk=1,
        group_slice_configs=_single_group_cfg(real_bytes=payload),
    )

    key1 = _key(1, group_idx=0)
    key2 = _key(2, group_idx=0)
    key3 = _key(3, group_idx=0)

    # Fill the allocator with 2 successful allocations
    r1 = manager.prepare_store([key1], _EMPTY_REQ)
    assert r1 is not None
    manager.complete_store([key1], _EMPTY_REQ, success=True)

    r2 = manager.prepare_store([key2], _EMPTY_REQ)
    assert r2 is not None
    manager.complete_store([key2], _EMPTY_REQ, success=True)

    assert len(manager._compact_allocations) == 2
    assert key1 in manager._compact_allocations
    assert key2 in manager._compact_allocations

    # Third attempt must evict key1 (LRU) to make room for key3.
    r3 = manager.prepare_store([key3], _EMPTY_REQ)
    assert r3 is not None, "Expected eviction to make room"
    assert len(r3.evicted_keys) == 1
    assert r3.evicted_keys[0] == key1

    # After complete_store, key1 should be gone, key2 and key3 present.
    manager.complete_store([key3], _EMPTY_REQ, success=True)
    assert key1 not in manager._compact_allocations
    assert key2 in manager._compact_allocations
    assert key3 in manager._compact_allocations
    assert len(manager._compact_allocations) == 2


def test_batch_allocation_fail_no_evictable_keys() -> None:
    """When the allocator is full and *no* evictable keys exist, the batch
    fails with None and zero mutation."""
    page_size = 256
    payload = 200
    manager = make_compact_manager(
        cpu_budget=page_size,  # only 1 page
        page_size=page_size,
        blocks_per_chunk=1,
        group_slice_configs=_single_group_cfg(real_bytes=payload),
    )

    key1 = _key(1, group_idx=0)
    key2 = _key(2, group_idx=0)

    # Fill the allocator with 1 allocation but DO NOT complete_store.
    # key1 stays pending (ref_cnt = -1 in policy, not evictable).
    r1 = manager.prepare_store([key1], _EMPTY_REQ)
    assert r1 is not None

    assert len(manager._compact_allocations) == 0
    assert len(manager._compact_pending) == 1

    # key2 cannot fit because key1 is pending and not evictable
    r2 = manager.prepare_store([key2], _EMPTY_REQ)
    assert r2 is None, "Expected None when only pending (non-evictable) entry exists"

    # No state mutation
    assert len(manager._compact_pending) == 1
    assert len(manager._compact_allocations) == 0


# ---------------------------------------------------------------------------
# 5. Eviction / protected / refcount
# ---------------------------------------------------------------------------


def test_compact_eviction_frees_space() -> None:
    """Eviction frees compact allocations so a new key can be stored."""
    page_size = 256
    payload = 200  # 1 page
    manager = make_compact_manager(
        cpu_budget=2 * page_size,  # 2 pages
        page_size=page_size,
        blocks_per_chunk=1,
        group_slice_configs=_single_group_cfg(real_bytes=payload),
    )

    key1 = _key(1, group_idx=0)
    key2 = _key(2, group_idx=0)
    key3 = _key(3, group_idx=0)

    # Store key1 and key2 (fills both pages)
    r1 = manager.prepare_store([key1], _EMPTY_REQ)
    assert r1 is not None
    manager.complete_store([key1], _EMPTY_REQ, success=True)
    r2 = manager.prepare_store([key2], _EMPTY_REQ)
    assert r2 is not None
    manager.complete_store([key2], _EMPTY_REQ, success=True)

    assert len(manager._compact_allocations) == 2

    # Prepare-store for key3 should trigger eviction of key1 (LRU)
    r3 = manager.prepare_store([key3], _EMPTY_REQ)
    assert r3 is not None, "Expected eviction to make room"
    assert len(r3.evicted_keys) == 1
    assert r3.evicted_keys[0] == key1

    # Evicted key should no longer be in committed allocations
    assert key1 not in manager._compact_allocations
    assert key2 in manager._compact_allocations


def test_compact_protected_blocks_not_evicted() -> None:
    """Blocks that are part of the incoming batch are protected from eviction."""
    page_size = 256
    payload = 200  # 1 page
    manager = make_compact_manager(
        cpu_budget=page_size,  # only 1 page
        page_size=page_size,
        blocks_per_chunk=1,
        group_slice_configs=_single_group_cfg(real_bytes=payload),
    )

    key1 = _key(1, group_idx=0)
    key2 = _key(2, group_idx=0)

    # Store key1 (fills the only page)
    r1 = manager.prepare_store([key1], _EMPTY_REQ)
    assert r1 is not None
    manager.complete_store([key1], _EMPTY_REQ, success=True)

    # Prepare-store for both key1 (already stored) and key2.
    # key1 must be in the protected set so it is NOT evicted.
    r2 = manager.prepare_store([key1, key2], _EMPTY_REQ)
    # Protected + only 1 page and key1 can't be evicted -> should fail
    assert r2 is None, (
        "Expected None because key1 is protected and no other evictable key exists"
    )


# ---------------------------------------------------------------------------
# 6. Failed store rolls back
# ---------------------------------------------------------------------------


def test_failed_store_frees_pages() -> None:
    """Failed complete_store frees pages and removes policy tracking."""
    page_size = 256
    payload = 200
    manager = make_compact_manager(
        cpu_budget=4 * page_size,
        page_size=page_size,
        blocks_per_chunk=1,
        group_slice_configs=_single_group_cfg(real_bytes=payload),
    )

    key = _key(1, group_idx=0)
    # Before store, pending is empty
    assert len(manager._compact_pending) == 0

    result = manager.prepare_store([key], _EMPTY_REQ)
    assert result is not None
    assert len(manager._compact_pending) == 1

    # Fail the store
    manager.complete_store([key], _EMPTY_REQ, success=False)

    # After failure: pending cleared, no committed allocations
    assert len(manager._compact_pending) == 0
    assert key not in manager._compact_allocations

    # The freed pages should be reusable
    result2 = manager.prepare_store([key], _EMPTY_REQ)
    assert result2 is not None, "Freed pages should be reusable"


# ---------------------------------------------------------------------------
# 7. Pending load rejection
# ---------------------------------------------------------------------------


def test_pending_store_not_loadable() -> None:
    """A key that is pending (prepared but not completed) returns
    HIT_PENDING on lookup."""
    page_size = 256
    payload = 200
    manager = make_compact_manager(
        cpu_budget=4 * page_size,
        page_size=page_size,
        blocks_per_chunk=1,
        group_slice_configs=_single_group_cfg(real_bytes=payload),
    )

    key = _key(1, group_idx=0)
    result = manager.prepare_store([key], _EMPTY_REQ)
    assert result is not None

    # Pending state: lookup should return HIT_PENDING
    lookup_result = manager.lookup(key, _EMPTY_REQ)
    assert lookup_result == LookupResult.HIT_PENDING, (
        f"Expected HIT_PENDING, got {lookup_result}"
    )

    # After successful complete_store, lookup should return HIT
    manager.complete_store([key], _EMPTY_REQ, success=True)
    assert manager.lookup(key, _EMPTY_REQ) == LookupResult.HIT


# ---------------------------------------------------------------------------
# 8. Reset clears all state
# ---------------------------------------------------------------------------


def test_compact_reset_clears_everything() -> None:
    """reset_cache clears allocator, pending, committed, and policy."""
    page_size = 256
    payload = 200
    manager = make_compact_manager(
        cpu_budget=4 * page_size,
        page_size=page_size,
        blocks_per_chunk=1,
        group_slice_configs=_single_group_cfg(real_bytes=payload),
    )

    key = _key(1, group_idx=0)
    result = manager.prepare_store([key], _EMPTY_REQ)
    assert result is not None
    manager.complete_store([key], _EMPTY_REQ, success=True)

    assert len(manager._compact_allocations) == 1

    manager.reset_cache()

    assert len(manager._compact_allocations) == 0
    assert len(manager._compact_pending) == 0
    assert manager._compact_allocator is not None
    assert manager._compact_allocator.used_bytes == 0

    # After reset, a new store should succeed
    key2 = _key(2, group_idx=0)
    r2 = manager.prepare_store([key2], _EMPTY_REQ)
    assert r2 is not None


# ---------------------------------------------------------------------------
# 9. Unknown group fails loud
# ---------------------------------------------------------------------------


def test_compact_unknown_group_fails() -> None:
    """A key with an unknown group index must raise."""
    page_size = 256
    manager = make_compact_manager(
        cpu_budget=4 * page_size,
        page_size=page_size,
        blocks_per_chunk=1,
        # Only group 0 is configured
        group_slice_configs=_single_group_cfg(group_idx=0, real_bytes=80),
    )

    # Key with group_idx=1 (not configured)
    bad_key = _key(99, group_idx=1)
    with pytest.raises(ValueError, match="unknown compact group index"):
        manager.prepare_store([bad_key], _EMPTY_REQ)


# ---------------------------------------------------------------------------
# 10. Legacy regression
# ---------------------------------------------------------------------------


def test_legacy_path_byte_for_byte() -> None:
    """When compact is disabled, the manager behaves identically to the
    original block-pool path."""
    legacy = make_legacy_manager(num_blocks=4)
    keys = _keys([1, 2, 3])
    # This follows the existing test pattern from test_manager.py
    result = legacy.prepare_store(keys, _EMPTY_REQ)
    assert result is not None
    assert len(result.keys_to_store) == 3
    assert isinstance(result.store_spec, CPULoadStoreSpec)
    assert len(result.evicted_keys) == 0

    legacy.complete_store([keys[0], keys[1]], _EMPTY_REQ, success=True)

    # Lookup should show HIT for stored, MISS for not-yet-stored
    assert legacy.lookup(keys[0], _EMPTY_REQ) == LookupResult.HIT
    assert legacy.lookup(keys[1], _EMPTY_REQ) == LookupResult.HIT
    assert legacy.lookup(keys[2], _EMPTY_REQ) == LookupResult.HIT_PENDING

    # Complete the third store
    legacy.complete_store([keys[2]], _EMPTY_REQ, success=True)
    assert legacy.lookup(keys[2], _EMPTY_REQ) == LookupResult.HIT

    # Now load
    load_spec = legacy.prepare_load(keys, _EMPTY_REQ)
    assert isinstance(load_spec, CPULoadStoreSpec)
    assert len(load_spec.block_ids) == 3

    legacy.complete_load(keys, _EMPTY_REQ)

    # Evict by overfilling
    more_keys = _keys([4, 5, 6])
    result2 = legacy.prepare_store(more_keys, _EMPTY_REQ)
    assert result2 is not None
    # At least some keys were evicted
    assert len(result2.evicted_keys) > 0

    # Reset
    legacy.reset_cache()
    assert legacy.lookup(keys[0], _EMPTY_REQ) == LookupResult.MISS


def test_legacy_empty_prepare_store() -> None:
    """Empty keys list in legacy mode returns empty spec."""
    legacy = make_legacy_manager()
    result = legacy.prepare_store([], _EMPTY_REQ)
    assert result is not None
    assert len(result.keys_to_store) == 0
    assert isinstance(result.store_spec, CPULoadStoreSpec)
    assert len(result.evicted_keys) == 0


# ---------------------------------------------------------------------------
# 11. Basic compact lifecycle
# ---------------------------------------------------------------------------


def test_compact_store_load_cycle() -> None:
    """Basic compact store then load cycle preserves addresses."""
    page_size = 256
    payload = 200
    manager = make_compact_manager(
        cpu_budget=4 * page_size,
        page_size=page_size,
        blocks_per_chunk=1,
        group_slice_configs=_single_group_cfg(real_bytes=payload),
    )

    key = _key(1, group_idx=0)

    # Store
    store_result = manager.prepare_store([key], _EMPTY_REQ)
    assert store_result is not None
    spec = store_result.store_spec
    assert isinstance(spec, CompactCPULoadStoreSpec)
    store_addr = spec.compact_addresses[0]
    assert store_addr.physical_spans is not None
    assert store_addr.logical_length == payload

    manager.complete_store([key], _EMPTY_REQ, success=True)

    # Load
    load_spec = manager.prepare_load([key], _EMPTY_REQ)
    assert isinstance(load_spec, CompactCPULoadStoreSpec)
    load_addr = load_spec.compact_addresses[0]

    # Same address through pending -> committed
    assert load_addr.byte_offset == store_addr.byte_offset
    assert load_addr.logical_length == store_addr.logical_length
    assert load_addr.physical_spans == store_addr.physical_spans


# ---------------------------------------------------------------------------
# 12. Compact empty prepare_store
# ---------------------------------------------------------------------------


def test_compact_empty_prepare_store() -> None:
    """Empty keys list in compact mode returns empty CompactCPULoadStoreSpec."""
    manager = make_compact_manager(
        cpu_budget=1024,
        page_size=256,
        group_slice_configs=_single_group_cfg(real_bytes=80),
    )
    result = manager.prepare_store([], _EMPTY_REQ)
    assert result is not None
    assert len(result.keys_to_store) == 0
    assert isinstance(result.store_spec, CompactCPULoadStoreSpec)
    assert len(result.evicted_keys) == 0


# ---------------------------------------------------------------------------
# 13. ARC policy also works
# ---------------------------------------------------------------------------


def test_compact_arc_policy() -> None:
    """ARC policy should work with compact layout."""
    page_size = 256
    payload = 200
    manager = make_compact_manager(
        cpu_budget=2 * page_size,
        page_size=page_size,
        cache_policy="arc",
        blocks_per_chunk=1,
        group_slice_configs=_single_group_cfg(real_bytes=payload),
    )

    key1 = _key(1, group_idx=0)
    key2 = _key(2, group_idx=0)

    r1 = manager.prepare_store([key1], _EMPTY_REQ)
    assert r1 is not None
    manager.complete_store([key1], _EMPTY_REQ, success=True)

    r2 = manager.prepare_store([key2], _EMPTY_REQ)
    assert r2 is not None
    manager.complete_store([key2], _EMPTY_REQ, success=True)

    # ARC should evict key1 (oldest) when we need space for key3
    key3 = _key(3, group_idx=0)
    r3 = manager.prepare_store([key3], _EMPTY_REQ)
    assert r3 is not None
    assert len(r3.evicted_keys) >= 1


# ---------------------------------------------------------------------------
# 14. Load ownership parity with canonical manager
# ---------------------------------------------------------------------------


def test_compact_prepare_load_rejects_pending() -> None:
    """prepare_load for a pending (not yet committed) key must fail loud."""
    page_size = 256
    payload = 200
    manager = make_compact_manager(
        cpu_budget=4 * page_size,
        page_size=page_size,
        blocks_per_chunk=1,
        group_slice_configs=_single_group_cfg(real_bytes=payload),
    )

    key = _key(1, group_idx=0)

    # Store but do NOT complete_store — key stays pending.
    result = manager.prepare_store([key], _EMPTY_REQ)
    assert result is not None
    assert manager.lookup(key, _EMPTY_REQ) == LookupResult.HIT_PENDING

    # prepare_load must reject pending keys.
    with pytest.raises(AssertionError, match="not committed"):
        manager.prepare_load([key], _EMPTY_REQ)

    # After complete_store, prepare_load succeeds.
    manager.complete_store([key], _EMPTY_REQ, success=True)
    spec = manager.prepare_load([key], _EMPTY_REQ)
    assert isinstance(spec, CompactCPULoadStoreSpec)
    assert len(spec.compact_addresses) == 1


def test_compact_prepare_load_removes_from_evictable() -> None:
    """prepare_load on a committed key increments ref_cnt and removes from
    evictable set."""
    page_size = 256
    payload = 200
    manager = make_compact_manager(
        cpu_budget=4 * page_size,
        page_size=page_size,
        blocks_per_chunk=1,
        group_slice_configs=_single_group_cfg(real_bytes=payload),
    )

    key = _key(1, group_idx=0)

    # Store and commit.
    result = manager.prepare_store([key], _EMPTY_REQ)
    assert result is not None
    manager.complete_store([key], _EMPTY_REQ, success=True)

    # After commit: ref_cnt == 0, evictable count == 1.
    block = manager._policy.get(key)
    assert block is not None
    assert block.ref_cnt == 0
    assert manager._num_evictable_cache_blocks == 1
    assert manager._num_write_pending_blocks == 0

    # prepare_load: removes from evictable, increments ref_cnt.
    manager.prepare_load([key], _EMPTY_REQ)
    assert block.ref_cnt == 1
    assert manager._num_evictable_cache_blocks == 0
    # complete_load restores evictable.
    manager.complete_load([key], _EMPTY_REQ)
    assert block.ref_cnt == 0
    assert manager._num_evictable_cache_blocks == 1
    assert manager._num_write_pending_blocks == 0


def test_compact_load_active_prevents_eviction() -> None:
    """While a load is active (ref_cnt > 0), the key cannot be evicted by
    evict_until, so a new store that would require eviction fails."""
    page_size = 256
    payload = 200
    manager = make_compact_manager(
        cpu_budget=2 * page_size,  # exactly 2 pages
        page_size=page_size,
        blocks_per_chunk=1,
        group_slice_configs=_single_group_cfg(real_bytes=payload),
    )

    key1 = _key(1, group_idx=0)
    key2 = _key(2, group_idx=0)
    key3 = _key(3, group_idx=0)

    # Store and commit key1 and key2 (fills the 2-page budget).
    r1 = manager.prepare_store([key1], _EMPTY_REQ)
    assert r1 is not None
    manager.complete_store([key1], _EMPTY_REQ, success=True)

    r2 = manager.prepare_store([key2], _EMPTY_REQ)
    assert r2 is not None
    manager.complete_store([key2], _EMPTY_REQ, success=True)
    assert manager._num_evictable_cache_blocks == 2

    # Start a load on key1 — removes it from evictable set.
    manager.prepare_load([key1], _EMPTY_REQ)
    assert manager._num_evictable_cache_blocks == 1

    # Try to store key3. Only key1 (loaded, ref_cnt=1) and key2 (ref_cnt=0)
    # exist. key1 is protected by ref_cnt, leaving key2 alone with 1 page.
    # Actually, evict_until checks ref_cnt: key1 has ref_cnt=1 so it is skipped;
    # key2 has ref_cnt=0 so it can be evicted. key2 frees 1 page but key3 also
    # needs 1 page, so this should succeed.
    # Verify key2 is evicted, not key1.
    r3 = manager.prepare_store([key3], _EMPTY_REQ)
    assert r3 is not None
    assert len(r3.evicted_keys) == 1
    assert r3.evicted_keys[0] == key2  # key1 is loaded, not evictable

    manager.complete_store([key3], _EMPTY_REQ, success=True)

    # key1 still loaded, key3 committed.
    assert manager.lookup(key1, _EMPTY_REQ) == LookupResult.HIT
    assert manager.lookup(key3, _EMPTY_REQ) == LookupResult.HIT

    # complete_load on key1 restores evictable.
    manager.complete_load([key1], _EMPTY_REQ)
    assert manager._num_evictable_cache_blocks == 2


def test_compact_load_active_store_no_free_space() -> None:
    """When allocator is full and the only key is loaded (ref_cnt > 0),
    new store fails with no mutation."""
    page_size = 256
    payload = 200
    manager = make_compact_manager(
        cpu_budget=page_size,  # only 1 page
        page_size=page_size,
        blocks_per_chunk=1,
        group_slice_configs=_single_group_cfg(real_bytes=payload),
    )

    key1 = _key(1, group_idx=0)
    key2 = _key(2, group_idx=0)

    # Store and commit key1 (fills the budget).
    r1 = manager.prepare_store([key1], _EMPTY_REQ)
    assert r1 is not None
    manager.complete_store([key1], _EMPTY_REQ, success=True)
    assert manager._num_evictable_cache_blocks == 1

    # Start load on key1 — removes it from evictable.
    manager.prepare_load([key1], _EMPTY_REQ)
    assert manager._num_evictable_cache_blocks == 0

    # key2 cannot fit: key1 is loaded (ref_cnt > 0, not evictable).
    r2 = manager.prepare_store([key2], _EMPTY_REQ)
    assert r2 is None, "Expected None when only loaded key exists and no free space"

    # No mutation: key1 still loaded.
    assert manager.lookup(key1, _EMPTY_REQ) == LookupResult.HIT
    assert manager.lookup(key2, _EMPTY_REQ) == LookupResult.MISS

    # After complete_load, key2 can be stored (key1 becomes evictable).
    manager.complete_load([key1], _EMPTY_REQ)
    r3 = manager.prepare_store([key2], _EMPTY_REQ)
    assert r3 is not None
    assert len(r3.evicted_keys) == 1
    assert r3.evicted_keys[0] == key1


def test_compact_double_complete_load_fails() -> None:
    """Calling complete_load twice on the same key must fail."""
    page_size = 256
    payload = 200
    manager = make_compact_manager(
        cpu_budget=4 * page_size,
        page_size=page_size,
        blocks_per_chunk=1,
        group_slice_configs=_single_group_cfg(real_bytes=payload),
    )

    key = _key(1, group_idx=0)

    # Store and commit.
    result = manager.prepare_store([key], _EMPTY_REQ)
    assert result is not None
    manager.complete_store([key], _EMPTY_REQ, success=True)

    # Load then complete_load once.
    manager.prepare_load([key], _EMPTY_REQ)
    manager.complete_load([key], _EMPTY_REQ)

    # Second complete_load must fail: ref_cnt is already 0.
    with pytest.raises(AssertionError, match="ref_cnt is already 0"):
        manager.complete_load([key], _EMPTY_REQ)


def test_compact_counters_exact() -> None:
    """Verify pending/write/evictable counters exactly across success store,
    failed store, eviction, and reset."""
    page_size = 256
    payload = 200
    # Budget = 3 pages (768 bytes) so key4 triggers eviction.
    manager = make_compact_manager(
        cpu_budget=3 * page_size,
        page_size=page_size,
        blocks_per_chunk=1,
        group_slice_configs=_single_group_cfg(real_bytes=payload),
    )

    key1 = _key(1, group_idx=0)
    key2 = _key(2, group_idx=0)
    key3 = _key(3, group_idx=0)
    key4 = _key(4, group_idx=0)

    # --- Initial: all zeros ---
    assert manager._num_write_pending_blocks == 0
    assert manager._num_evictable_cache_blocks == 0

    # --- prepare_store: increment write_pending ---
    r1 = manager.prepare_store([key1], _EMPTY_REQ)
    assert r1 is not None
    assert manager._num_write_pending_blocks == 1
    assert manager._num_evictable_cache_blocks == 0

    # --- complete_store success: decrement write_pending, increment evictable ---
    manager.complete_store([key1], _EMPTY_REQ, success=True)
    assert manager._num_write_pending_blocks == 0
    assert manager._num_evictable_cache_blocks == 1

    # --- Failed store: prepare then fail ---
    r2 = manager.prepare_store([key2], _EMPTY_REQ)
    assert r2 is not None
    assert manager._num_write_pending_blocks == 1
    assert manager._num_evictable_cache_blocks == 1

    manager.complete_store([key2], _EMPTY_REQ, success=False)
    assert manager._num_write_pending_blocks == 0
    assert manager._num_evictable_cache_blocks == 1  # unchanged (key1 still)

    # key2 should not be in committed allocations.
    assert key2 not in manager._compact_allocations

    # --- Eviction: store 2 more keys to trigger eviction ---
    # Currently key1 is the only committed key (evictable count=1).
    # Budget is 4 pages, key1 uses 1 page, so we have 3 free pages.
    # key3 uses 1 page. Store key2 (retry) and key3 together.
    r3 = manager.prepare_store([key2, key3], _EMPTY_REQ)
    assert r3 is not None
    assert manager._num_write_pending_blocks == 2  # 2 new pending
    assert manager._num_evictable_cache_blocks == 1  # key1 still evictable

    # Commit both.
    manager.complete_store([key2, key3], _EMPTY_REQ, success=True)
    assert manager._num_write_pending_blocks == 0
    assert manager._num_evictable_cache_blocks == 3  # key1, key2, key3

    # --- Eviction via overfill ---
    # Budget is 3 pages, all 3 are used (key1+key2+key3).
    # Store key4 (1 page) triggers eviction of key1 (LRU).
    key4 = _key(4, group_idx=0)
    r4 = manager.prepare_store([key4], _EMPTY_REQ)
    assert r4 is not None
    assert len(r4.evicted_keys) == 1
    assert r4.evicted_keys[0] == key1
    assert manager._num_evictable_cache_blocks == 2  # key1 evicted, key2, key3 remain
    assert manager._num_write_pending_blocks == 1  # key4 pending

    manager.complete_store([key4], _EMPTY_REQ, success=True)
    assert manager._num_write_pending_blocks == 0
    assert manager._num_evictable_cache_blocks == 3  # key2, key3, key4

    # --- prepare_load / complete_load counters ---
    manager.prepare_load([key2], _EMPTY_REQ)
    assert manager._num_evictable_cache_blocks == 2  # key2 removed from evictable
    assert manager._num_write_pending_blocks == 0

    manager.complete_load([key2], _EMPTY_REQ)
    assert manager._num_evictable_cache_blocks == 3  # key2 restored
    assert manager._num_write_pending_blocks == 0

    # --- Reset zeros all counters ---
    manager.reset_cache()
    assert manager._num_write_pending_blocks == 0
    assert manager._num_evictable_cache_blocks == 0
    assert len(manager._compact_allocations) == 0
    assert len(manager._compact_pending) == 0
