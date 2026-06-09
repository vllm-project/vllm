# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pytest

from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    OffloadingEvent,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy


def make_req_context(
    req_id: str = "", kv_transfer_params: dict | None = None
) -> ReqContext:
    """Create a ReqContext as production code would, from a request's params."""
    return ReqContext(req_id=req_id, kv_transfer_params=kv_transfer_params)


_EMPTY_REQ_CTX = make_req_context()


@dataclass
class ExpectedPrepareStoreOutput:
    keys_to_store: list[int]
    store_block_ids: list[int]
    evicted_keys: list[int]


def to_key(int_hash: int) -> OffloadKey:
    return make_offload_key(str(int_hash).encode(), 0)


def to_keys(int_hashes: list[int]) -> list[OffloadKey]:
    return [to_key(i) for i in int_hashes]


def verify_store_output(
    prepare_store_output: PrepareStoreOutput | None,
    expected_prepare_store_output: ExpectedPrepareStoreOutput,
):
    assert prepare_store_output is not None
    assert prepare_store_output.keys_to_store == to_keys(
        expected_prepare_store_output.keys_to_store
    )
    assert prepare_store_output.evicted_keys == to_keys(
        expected_prepare_store_output.evicted_keys
    )
    store_spec = prepare_store_output.store_spec
    assert isinstance(store_spec, CPULoadStoreSpec)
    expected_array = np.array(
        expected_prepare_store_output.store_block_ids, dtype=np.int64
    )
    assert np.array_equal(expected_array, store_spec.block_ids)


def verify_load_output(
    prepare_load_output: LoadStoreSpec, expected_prepare_load_output: list[int]
):
    assert isinstance(prepare_load_output, CPULoadStoreSpec)
    expected_array = np.array(expected_prepare_load_output, dtype=np.int64)
    assert np.array_equal(expected_array, prepare_load_output.block_ids)


def verify_events(
    events: Iterable[OffloadingEvent],
    expected_stores: tuple[set[int], ...] = (),
    expected_evictions: tuple[set[int], ...] = (),
):
    stores: list[set[OffloadKey]] = []
    evictions: list[set[OffloadKey]] = []
    for event in events:
        assert event.medium == CPULoadStoreSpec.medium()
        if event.removed:
            evictions.append(set(event.keys))
        else:
            stores.append(set(event.keys))

    def to_key_sets(
        int_sets: tuple[set[int], ...],
    ) -> tuple[set[OffloadKey], ...]:
        return tuple([set(to_keys(list(int_set))) for int_set in int_sets])

    assert tuple(evictions) == to_key_sets(expected_evictions)
    assert tuple(stores) == to_key_sets(expected_stores)


@pytest.mark.parametrize("eviction_policy", ["lru", "arc"])
def test_already_stored_block_not_evicted_during_prepare_store(eviction_policy):
    """
    Regression test: a block that is already stored must not be evicted
    by prepare_store() when it needs to make room for new blocks.
    Applies to both lru and arc policies.

    Scenario:
        - Store blocks [1, 2] and complete.
        - touch([1]) makes block 2 the LRU candidate.
        - prepare_store([2, 3, 4, 5]):
            * block 2 is filtered out as "already stored"
            * but without the fix, block 2 would be evicted as the LRU
              candidate to make room for [3, 4, 5]
        - After complete_store([2, 3, 4, 5]), block 2 must still be present.
    """
    manager = CPUOffloadingManager(
        num_blocks=4,
        cache_policy=eviction_policy,
        enable_events=True,
    )

    # store [1, 2] and complete
    manager.prepare_store(to_keys([1, 2]), _EMPTY_REQ_CTX)
    manager.complete_store(to_keys([1, 2]), _EMPTY_REQ_CTX)

    # touch [1] to make block 2 the LRU candidate
    manager.touch(to_keys([1]), _EMPTY_REQ_CTX)

    # prepare_store([2, 3, 4, 5]):
    #   - block 2 is already stored -> filtered out of keys_to_store
    #   - block 2 must NOT be evicted even though it is the LRU candidate
    #   - block 1 (ID 0) is evicted instead; new blocks [3,4,5] get IDs 2,3,0
    prepare_store_output = manager.prepare_store(to_keys([2, 3, 4, 5]), _EMPTY_REQ_CTX)
    verify_store_output(
        prepare_store_output,
        ExpectedPrepareStoreOutput(
            keys_to_store=[3, 4, 5],
            store_block_ids=[2, 3, 0],
            evicted_keys=[1],  # block 1 evicted, not block 2
        ),
    )

    # complete_store must not silently drop block 2
    manager.complete_store(to_keys([2, 3, 4, 5]), _EMPTY_REQ_CTX)

    # block 2 must still be present in the cache
    assert manager.lookup(to_key(2), _EMPTY_REQ_CTX) is True


def test_cpu_manager():
    """
    Tests CPUOffloadingManager with lru policy.
    """
    # initialize a CPU manager with a capacity of 4 blocks
    cpu_manager = CPUOffloadingManager(
        num_blocks=4, cache_policy="lru", enable_events=True
    )

    # prepare store [1, 2]
    prepare_store_output = cpu_manager.prepare_store(to_keys([1, 2]), _EMPTY_REQ_CTX)
    verify_store_output(
        prepare_store_output,
        ExpectedPrepareStoreOutput(
            keys_to_store=[1, 2],
            store_block_ids=[0, 1],
            evicted_keys=[],
        ),
    )

    # lookup [1, 2] -> write in-flight, not yet ready
    assert cpu_manager.lookup(to_key(1), _EMPTY_REQ_CTX) is None
    assert cpu_manager.lookup(to_key(2), _EMPTY_REQ_CTX) is None

    # no events so far
    assert list(cpu_manager.take_events()) == []

    # complete store [1, 2]
    cpu_manager.complete_store(to_keys([1, 2]), _EMPTY_REQ_CTX)
    verify_events(cpu_manager.take_events(), expected_stores=({1, 2},))

    # lookup [1, 2]
    assert cpu_manager.lookup(to_key(1), _EMPTY_REQ_CTX) is True
    assert cpu_manager.lookup(to_key(2), _EMPTY_REQ_CTX) is True
    assert cpu_manager.lookup(to_key(3), _EMPTY_REQ_CTX) is False

    # prepare store [2, 3, 4, 5] -> evicts [1]
    prepare_store_output = cpu_manager.prepare_store(
        to_keys([2, 3, 4, 5]), _EMPTY_REQ_CTX
    )
    verify_store_output(
        prepare_store_output,
        ExpectedPrepareStoreOutput(
            keys_to_store=[3, 4, 5],
            store_block_ids=[2, 3, 0],
            evicted_keys=[1],
        ),
    )

    # verify eviction event
    verify_events(cpu_manager.take_events(), expected_evictions=({1},))

    # prepare store with no space
    assert cpu_manager.prepare_store(to_keys([1, 6]), _EMPTY_REQ_CTX) is None

    # complete store [2, 3, 4, 5]
    cpu_manager.complete_store(to_keys([2, 3, 4, 5]), _EMPTY_REQ_CTX)

    # lookup (now that we have [2, 3, 4, 5])
    assert cpu_manager.lookup(to_key(1), _EMPTY_REQ_CTX) is False
    assert cpu_manager.lookup(to_key(2), _EMPTY_REQ_CTX) is True
    assert cpu_manager.lookup(to_key(3), _EMPTY_REQ_CTX) is True
    assert cpu_manager.lookup(to_key(4), _EMPTY_REQ_CTX) is True
    assert cpu_manager.lookup(to_key(5), _EMPTY_REQ_CTX) is True
    assert cpu_manager.lookup(to_key(0), _EMPTY_REQ_CTX) is False

    # prepare load [2, 3]
    prepare_load_output = cpu_manager.prepare_load(to_keys([2, 3]), _EMPTY_REQ_CTX)
    verify_load_output(prepare_load_output, [1, 2])

    # prepare store with no space ([2, 3] is being loaded)
    assert cpu_manager.prepare_store(to_keys([6, 7, 8]), _EMPTY_REQ_CTX) is None

    # complete load [2, 3]
    cpu_manager.complete_load(to_keys([2, 3]), _EMPTY_REQ_CTX)

    # prepare store [6, 7, 8] -> evicts [2, 3, 4] (oldest)
    prepare_store_output = cpu_manager.prepare_store(to_keys([6, 7, 8]), _EMPTY_REQ_CTX)
    verify_store_output(
        prepare_store_output,
        ExpectedPrepareStoreOutput(
            keys_to_store=[6, 7, 8],
            store_block_ids=[3, 2, 1],
            evicted_keys=[2, 3, 4],
        ),
    )

    # complete store [6, 7, 8]
    cpu_manager.complete_store(to_keys([6, 7, 8]), _EMPTY_REQ_CTX)

    # touch [5, 6, 7] (move to end of LRU order)
    cpu_manager.touch(to_keys([5, 6, 7]), _EMPTY_REQ_CTX)

    # prepare store [7, 9] -> evicts [8] (oldest following previous touch)
    prepare_store_output = cpu_manager.prepare_store(to_keys([9]), _EMPTY_REQ_CTX)
    verify_store_output(
        prepare_store_output,
        ExpectedPrepareStoreOutput(
            keys_to_store=[9],
            store_block_ids=[1],
            evicted_keys=[8],
        ),
    )

    # complete store [7, 9] with failure
    cpu_manager.complete_store(to_keys([7, 9]), _EMPTY_REQ_CTX, success=False)

    # assert [7] is still stored, but [9] is not
    assert cpu_manager.lookup(to_key(7), _EMPTY_REQ_CTX) is True
    assert cpu_manager.lookup(to_key(9), _EMPTY_REQ_CTX) is False

    verify_events(
        cpu_manager.take_events(),
        expected_stores=({3, 4, 5}, {6, 7, 8}),
        expected_evictions=({2, 3, 4}, {8}),
    )


def test_prepare_load_preserves_key_order():
    """block_ids[i] must correspond to keys[i] (co-indexed invariant)."""
    manager = CPUOffloadingManager(num_blocks=4, cache_policy="lru")

    key_a, key_b, key_c = to_key(0), to_key(1), to_key(2)

    # Store all three keys and learn their block ID assignments
    store_output = manager.prepare_store([key_a, key_b, key_c], _EMPTY_REQ_CTX)
    assert store_output is not None
    assert isinstance(store_output.store_spec, CPULoadStoreSpec)
    key_to_block_id = {
        k: int(bid)
        for k, bid in zip(store_output.keys_to_store, store_output.store_spec.block_ids)
    }
    manager.complete_store([key_a, key_b, key_c], _EMPTY_REQ_CTX)

    # Forward order: [a, b, c]
    spec_fwd = manager.prepare_load([key_a, key_b, key_c], _EMPTY_REQ_CTX)
    assert isinstance(spec_fwd, CPULoadStoreSpec)
    assert [int(x) for x in spec_fwd.block_ids] == [
        key_to_block_id[key_a],
        key_to_block_id[key_b],
        key_to_block_id[key_c],
    ]
    manager.complete_load([key_a, key_b, key_c], _EMPTY_REQ_CTX)  # order irrelevant

    # Arbitrary permutation: [b, c, a]
    spec_perm = manager.prepare_load([key_b, key_c, key_a], _EMPTY_REQ_CTX)
    assert isinstance(spec_perm, CPULoadStoreSpec)
    assert [int(x) for x in spec_perm.block_ids] == [
        key_to_block_id[key_b],
        key_to_block_id[key_c],
        key_to_block_id[key_a],
    ]
    manager.complete_load([key_a, key_b, key_c], _EMPTY_REQ_CTX)  # order irrelevant


class TestARCPolicy:
    """Unit tests for CPUOffloadingManager with ARC eviction policy."""

    def _make_manager(
        self, num_blocks: int = 4, enable_events: bool = True
    ) -> tuple[CPUOffloadingManager, ARCCachePolicy]:
        manager = CPUOffloadingManager(
            num_blocks=num_blocks,
            cache_policy="arc",
            enable_events=enable_events,
        )
        policy = manager._policy
        assert isinstance(policy, ARCCachePolicy)
        return manager, policy

    def test_basic(self):
        """
        Tests CPUOffloadingManager with arc policy.
        Verifies that ARC handles store, load, and lookup operations correctly.
        """
        cpu_manager, arc_policy = self._make_manager()

        # prepare store [1, 2]
        prepare_store_output = cpu_manager.prepare_store(
            to_keys([1, 2]), _EMPTY_REQ_CTX
        )
        verify_store_output(
            prepare_store_output,
            ExpectedPrepareStoreOutput(
                keys_to_store=[1, 2],
                store_block_ids=[0, 1],
                evicted_keys=[],
            ),
        )

        # lookup [1, 2] -> write in-flight, not yet ready
        assert cpu_manager.lookup(to_key(1), _EMPTY_REQ_CTX) is None
        assert cpu_manager.lookup(to_key(2), _EMPTY_REQ_CTX) is None

        # no events so far
        assert list(cpu_manager.take_events()) == []

        # complete store [1, 2]
        cpu_manager.complete_store(to_keys([1, 2]), _EMPTY_REQ_CTX)
        verify_events(cpu_manager.take_events(), expected_stores=({1, 2},))

        # lookup [1, 2]
        assert cpu_manager.lookup(to_key(1), _EMPTY_REQ_CTX) is True
        assert cpu_manager.lookup(to_key(2), _EMPTY_REQ_CTX) is True
        assert cpu_manager.lookup(to_key(3), _EMPTY_REQ_CTX) is False

        # blocks should be in T1 (recent)
        assert len(arc_policy.t1) == 2
        assert len(arc_policy.t2) == 0

    def test_t1_to_t2_promotion(self):
        """
        Tests that accessing a block in T1 promotes it to T2 (frequent).
        This is a key feature of ARC's adaptive behavior.
        """
        cpu_manager, arc_policy = self._make_manager(enable_events=False)

        # store and complete block 1
        cpu_manager.prepare_store(to_keys([1]), _EMPTY_REQ_CTX)
        cpu_manager.complete_store(to_keys([1]), _EMPTY_REQ_CTX)

        # block 1 starts in T1 (recent)
        assert to_keys([1])[0] in arc_policy.t1
        assert to_keys([1])[0] not in arc_policy.t2

        # touch block 1 (simulate second access)
        cpu_manager.touch(to_keys([1]), _EMPTY_REQ_CTX)

        # block 1 should now be in T2 (frequent)
        assert to_keys([1])[0] not in arc_policy.t1
        assert to_keys([1])[0] in arc_policy.t2

    def test_eviction_with_load(self):
        """
        Tests ARC eviction behavior similar to LRU test.
        Verifies that blocks being loaded (ref_cnt > 0) cannot be evicted.
        """
        cpu_manager, _ = self._make_manager()

        # prepare and complete store [1, 2, 3, 4]
        prepare_store_output = cpu_manager.prepare_store(
            to_keys([1, 2, 3, 4]), _EMPTY_REQ_CTX
        )
        verify_store_output(
            prepare_store_output,
            ExpectedPrepareStoreOutput(
                keys_to_store=[1, 2, 3, 4],
                store_block_ids=[0, 1, 2, 3],
                evicted_keys=[],
            ),
        )
        cpu_manager.complete_store(to_keys([1, 2, 3, 4]), _EMPTY_REQ_CTX)

        # prepare load [2, 3] (increases ref_cnt)
        prepare_load_output = cpu_manager.prepare_load(to_keys([2, 3]), _EMPTY_REQ_CTX)
        verify_load_output(prepare_load_output, [1, 2])

        # prepare store [5, 6, 7] with [2, 3] being loaded
        # should fail because [2, 3] have ref_cnt > 0
        assert cpu_manager.prepare_store(to_keys([5, 6, 7]), _EMPTY_REQ_CTX) is None

        # complete load [2, 3]
        cpu_manager.complete_load(to_keys([2, 3]), _EMPTY_REQ_CTX)

        # now prepare store [5, 6, 7] should succeed
        # ARC will evict blocks one at a time from T1 as needed
        prepare_store_output = cpu_manager.prepare_store(
            to_keys([5, 6, 7]), _EMPTY_REQ_CTX
        )
        assert prepare_store_output is not None
        # Should successfully evict enough blocks to make room (at least 1)
        assert len(prepare_store_output.evicted_keys) >= 1

    def test_adaptive_target(self):
        """
        Tests ARC's adaptive target adjustment via ghost lists.
        When a block in B1 (ghost list) is accessed, target_t1_size increases.
        When a block in B2 is accessed, target_t1_size decreases.
        """
        cpu_manager, arc_policy = self._make_manager(num_blocks=2, enable_events=False)

        # store blocks 1, 2 (fills cache)
        cpu_manager.prepare_store(to_keys([1, 2]), _EMPTY_REQ_CTX)
        cpu_manager.complete_store(to_keys([1, 2]), _EMPTY_REQ_CTX)

        initial_target = arc_policy.target_t1_size

        # store block 3, evicting block 1 (moves to B1 ghost list)
        cpu_manager.prepare_store(to_keys([3]), _EMPTY_REQ_CTX)
        cpu_manager.complete_store(to_keys([3]), _EMPTY_REQ_CTX)

        # block 1 should be in B1 (ghost list)
        assert to_keys([1])[0] in arc_policy.b1

        # touch block 1 (cache miss, but in B1)
        # this should increase target_t1_size (favor recency)
        cpu_manager.touch(to_keys([1]), _EMPTY_REQ_CTX)

        # target should have increased
        assert arc_policy.target_t1_size > initial_target

    def test_t1_t2_eviction_policy(self):
        """
        Tests that ARC evicts from T1 or T2 based on target_t1_size.
        If |T1| >= target_t1_size, evict from T1, otherwise from T2.
        """
        cpu_manager, arc_policy = self._make_manager(enable_events=False)

        # store blocks 1, 2, 3, 4
        cpu_manager.prepare_store(to_keys([1, 2, 3, 4]), _EMPTY_REQ_CTX)
        cpu_manager.complete_store(to_keys([1, 2, 3, 4]), _EMPTY_REQ_CTX)

        # promote blocks 3, 4 to T2 by touching them
        cpu_manager.touch(to_keys([3, 4]), _EMPTY_REQ_CTX)

        # now: T1 = {1, 2}, T2 = {3, 4}
        assert len(arc_policy.t1) == 2
        assert len(arc_policy.t2) == 2

        # set target_t1_size to prefer evicting from T1
        # (when |T1| >= target, evict from T1)
        arc_policy.target_t1_size = 1

        # store block 5, should evict from T1 (block 1, LRU in T1)
        output = cpu_manager.prepare_store(to_keys([5]), _EMPTY_REQ_CTX)
        assert output is not None
        assert to_keys([1]) == output.evicted_keys

        cpu_manager.complete_store(to_keys([5]), _EMPTY_REQ_CTX)

        # block 1 should be in B1 (ghost list)
        assert to_keys([1])[0] in arc_policy.b1
        # block 5 should be in T1
        assert to_keys([5])[0] in arc_policy.t1

    def test_ghost_list_bounds(self):
        """
        Tests that ghost lists (B1, B2) don't grow unbounded.
        They should be capped at cache_capacity.
        """
        cpu_manager, arc_policy = self._make_manager(num_blocks=2, enable_events=False)

        # fill cache with blocks 1, 2
        cpu_manager.prepare_store(to_keys([1, 2]), _EMPTY_REQ_CTX)
        cpu_manager.complete_store(to_keys([1, 2]), _EMPTY_REQ_CTX)

        # store many blocks to fill ghost lists
        for i in range(3, 20):
            cpu_manager.prepare_store(to_keys([i]), _EMPTY_REQ_CTX)
            cpu_manager.complete_store(to_keys([i]), _EMPTY_REQ_CTX)

        # ghost lists should not exceed cache_capacity
        assert len(arc_policy.b1) <= arc_policy.cache_capacity
        assert len(arc_policy.b2) <= arc_policy.cache_capacity

    def test_touch_ordering(self):
        """
        Tests that touch() correctly updates access patterns.
        Similar to LRU test but verifies T1/T2 ordering.
        """
        cpu_manager, arc_policy = self._make_manager()

        # store blocks 1, 2, 3, 4
        cpu_manager.prepare_store(to_keys([1, 2, 3, 4]), _EMPTY_REQ_CTX)
        cpu_manager.complete_store(to_keys([1, 2, 3, 4]), _EMPTY_REQ_CTX)

        # promote 3, 4 to T2
        cpu_manager.touch(to_keys([3, 4]), _EMPTY_REQ_CTX)

        # T1 = {1, 2}, T2 = {3, 4}
        # touch [1, 3, 4] - should promote 1 to T2, and move 3,4 to end of T2
        cpu_manager.touch(to_keys([1, 3, 4]), _EMPTY_REQ_CTX)

        # T1 = {2}, T2 = {1, 3, 4} (in that order, with 4 most recent)
        assert len(arc_policy.t1) == 1
        assert len(arc_policy.t2) == 3

        # store block 5, should evict from T1 (block 2, only one in T1)
        prepare_store_output = cpu_manager.prepare_store(to_keys([5]), _EMPTY_REQ_CTX)
        verify_store_output(
            prepare_store_output,
            ExpectedPrepareStoreOutput(
                keys_to_store=[5],
                store_block_ids=[1],  # reuses block 2's storage
                evicted_keys=[2],
            ),
        )

    def test_failed_store(self):
        """
        Tests that failed store operations clean up correctly.
        Similar to LRU test but for ARC.
        """
        cpu_manager, arc_policy = self._make_manager()

        # store blocks 1, 2, 3, 4
        cpu_manager.prepare_store(to_keys([1, 2, 3, 4]), _EMPTY_REQ_CTX)
        cpu_manager.complete_store(to_keys([1, 2, 3, 4]), _EMPTY_REQ_CTX)

        # prepare store block 5 (will evict block 1)
        prepare_store_output = cpu_manager.prepare_store(to_keys([5]), _EMPTY_REQ_CTX)
        assert prepare_store_output is not None
        assert len(prepare_store_output.evicted_keys) == 1

        # complete store with failure
        cpu_manager.complete_store(to_keys([5]), _EMPTY_REQ_CTX, success=False)

        # block 5 should not be in cache
        assert cpu_manager.lookup(to_key(5), _EMPTY_REQ_CTX) is False
        # block 5 should not be in T1 or T2
        assert to_keys([5])[0] not in arc_policy.t1
        assert to_keys([5])[0] not in arc_policy.t2

        # evicted block should still be gone (in B1 ghost list)
        evicted_hash = prepare_store_output.evicted_keys[0]
        assert evicted_hash in arc_policy.b1

    def test_full_scenario(self):
        """
        Comprehensive test covering multiple ARC operations in sequence.
        Similar to the full LRU test but adapted for ARC behavior.
        """
        cpu_manager, arc_policy = self._make_manager()

        # store [1, 2]
        cpu_manager.prepare_store(to_keys([1, 2]), _EMPTY_REQ_CTX)
        cpu_manager.complete_store(to_keys([1, 2]), _EMPTY_REQ_CTX)

        # store [3, 4, 5] -> evicts [1]
        prepare_store_output = cpu_manager.prepare_store(
            to_keys([3, 4, 5]), _EMPTY_REQ_CTX
        )
        assert prepare_store_output is not None
        assert len(prepare_store_output.evicted_keys) == 1
        cpu_manager.complete_store(to_keys([3, 4, 5]), _EMPTY_REQ_CTX)

        # promote some blocks to T2
        cpu_manager.touch(to_keys([2, 3]), _EMPTY_REQ_CTX)

        # T1 has {4, 5}, T2 has {2, 3}
        assert len(arc_policy.t1) == 2
        assert len(arc_policy.t2) == 2

        # store [6] -> should evict from T1 (4 is oldest in T1)
        prepare_store_output = cpu_manager.prepare_store(to_keys([6]), _EMPTY_REQ_CTX)
        assert prepare_store_output is not None
        cpu_manager.complete_store(to_keys([6]), _EMPTY_REQ_CTX)

        # verify blocks 2, 3 (in T2) are still present
        assert cpu_manager.lookup(to_key(2), _EMPTY_REQ_CTX) is True
        assert cpu_manager.lookup(to_key(3), _EMPTY_REQ_CTX) is True

        # verify events
        events = list(cpu_manager.take_events())
        assert len(events) > 0  # should have store and eviction events


def test_filter_reused_manager():
    """
    Tests CPUOffloadingManager reuse filtering (store_threshold=2).
    """
    manager = CPUOffloadingManager(
        num_blocks=4,
        cache_policy="lru",
        enable_events=True,
        store_threshold=2,
        max_tracker_size=3,
    )

    # Lookup [1, 2] -> 1st time, added to tracker but not eligible for store yet
    assert manager.lookup(to_key(1), _EMPTY_REQ_CTX) is False
    assert manager.lookup(to_key(2), _EMPTY_REQ_CTX) is False

    # prepare store [1, 2] -> should be filtered
    prepare_store_output = manager.prepare_store(to_keys([1, 2]), _EMPTY_REQ_CTX)
    assert prepare_store_output is not None
    assert prepare_store_output.keys_to_store == []

    # Lookup [1] -> 2nd time, eligible now
    assert manager.lookup(to_key(1), _EMPTY_REQ_CTX) is False

    # prepare store [1, 2] -> [1] should be eligible, [2] should be filtered
    prepare_store_output = manager.prepare_store(to_keys([1, 2]), _EMPTY_REQ_CTX)
    assert prepare_store_output is not None
    assert prepare_store_output.keys_to_store == to_keys([1])

    # Lookup [3, 4] -> 1st time
    # (evicts [2] from tracker since max_size is 3 and tracker has [1])
    assert manager.lookup(to_key(3), _EMPTY_REQ_CTX) is False
    assert manager.lookup(to_key(4), _EMPTY_REQ_CTX) is False
    # Verify [2] was evicted from the tracker (tracker now has: [1], [3], [4])
    assert to_keys([2])[0] not in manager.counts

    # Lookup [2] again -> (this adds [2] back to the tracker as 1st time)
    assert manager.lookup(to_key(2), _EMPTY_REQ_CTX) is False
    # Verify [2] was re-added with count=1 (not eligible yet)
    assert manager.counts.get(to_keys([2])[0]) == 1

    # prepare store [2] -> should still be filtered out since count was reset
    prepare_store_output = manager.prepare_store(to_keys([2]), _EMPTY_REQ_CTX)
    assert prepare_store_output is not None
    assert prepare_store_output.keys_to_store == []

    manager.complete_store(to_keys([1]), _EMPTY_REQ_CTX)
