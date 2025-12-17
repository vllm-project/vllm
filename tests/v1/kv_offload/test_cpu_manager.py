# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec,
    OffloadingEvent,
    PrepareStoreOutput,
)
from vllm.v1.kv_offload.arc_manager import ARCOffloadingManager
from vllm.v1.kv_offload.backends.cpu import CPUBackend
from vllm.v1.kv_offload.lru_manager import LRUOffloadingManager
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec


@dataclass
class ExpectedPrepareStoreOutput:
    block_hashes_to_store: list[int]
    store_block_ids: list[int]
    block_hashes_evicted: list[int]


def to_hashes(int_hashes: list[int]) -> list[BlockHash]:
    return [BlockHash(str(i).encode()) for i in int_hashes]


def verify_store_output(
    prepare_store_output: PrepareStoreOutput | None,
    expected_prepare_store_output: ExpectedPrepareStoreOutput,
):
    assert prepare_store_output is not None
    assert prepare_store_output.block_hashes_to_store == to_hashes(
        expected_prepare_store_output.block_hashes_to_store
    )
    assert prepare_store_output.block_hashes_evicted == to_hashes(
        expected_prepare_store_output.block_hashes_evicted
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
    block_size: int,
    expected_stores: tuple[set[int], ...] = (),
    expected_evictions: tuple[set[int], ...] = (),
):
    stores: list[set[BlockHash]] = []
    evictions: list[set[BlockHash]] = []
    for event in events:
        assert event.medium == CPULoadStoreSpec.medium()
        assert event.block_size == block_size
        if event.removed:
            evictions.append(set(event.block_hashes))
        else:
            stores.append(set(event.block_hashes))

    def to_hash_sets(int_sets: tuple[set[int], ...]) -> tuple[set[BlockHash], ...]:
        return tuple([set(to_hashes(list(int_set))) for int_set in int_sets])

    assert tuple(evictions) == to_hash_sets(expected_evictions)
    assert tuple(stores) == to_hash_sets(expected_stores)


def test_cpu_manager():
    """
    Tests LRUOffloadingManager with a CPUBackend.
    """
    # initialize a CPU backend with a capacity of 4 blocks
    block_size = 256
    cpu_backend = CPUBackend(block_size=block_size, num_blocks=4)
    cpu_manager = LRUOffloadingManager(cpu_backend, enable_events=True)

    # prepare store [1, 2]
    prepare_store_output = cpu_manager.prepare_store(to_hashes([1, 2]))
    verify_store_output(
        prepare_store_output,
        ExpectedPrepareStoreOutput(
            block_hashes_to_store=[1, 2],
            store_block_ids=[0, 1],
            block_hashes_evicted=[],
        ),
    )

    # lookup [1, 2] -> not ready
    assert cpu_manager.lookup(to_hashes([1, 2])) == 0

    # no events so far
    assert list(cpu_manager.take_events()) == []

    # complete store [1, 2]
    cpu_manager.complete_store(to_hashes([1, 2]))
    verify_events(
        cpu_manager.take_events(), block_size=block_size, expected_stores=({1, 2},)
    )

    # lookup [1, 2]
    assert cpu_manager.lookup(to_hashes([1])) == 1
    assert cpu_manager.lookup(to_hashes([1, 2])) == 2
    assert cpu_manager.lookup(to_hashes([1, 2, 3])) == 2

    # prepare store [2, 3, 4, 5] -> evicts [1]
    prepare_store_output = cpu_manager.prepare_store(to_hashes([2, 3, 4, 5]))
    verify_store_output(
        prepare_store_output,
        ExpectedPrepareStoreOutput(
            block_hashes_to_store=[3, 4, 5],
            store_block_ids=[2, 3, 0],
            block_hashes_evicted=[1],
        ),
    )

    # verify eviction event
    verify_events(
        cpu_manager.take_events(), block_size=block_size, expected_evictions=({1},)
    )

    # prepare store with no space
    assert cpu_manager.prepare_store(to_hashes([1, 6])) is None

    # complete store [2, 3, 4, 5]
    cpu_manager.complete_store(to_hashes([2, 3, 4, 5]))

    # prepare load [2, 3]
    prepare_load_output = cpu_manager.prepare_load(to_hashes([2, 3]))
    verify_load_output(prepare_load_output, [1, 2])

    # prepare store with no space ([2, 3] is being loaded)
    assert cpu_manager.prepare_store(to_hashes([6, 7, 8])) is None

    # complete load [2, 3]
    cpu_manager.complete_load(to_hashes([2, 3]))

    # prepare store [6, 7, 8] -> evicts [2, 3, 4] (oldest)
    prepare_store_output = cpu_manager.prepare_store(to_hashes([6, 7, 8]))
    verify_store_output(
        prepare_store_output,
        ExpectedPrepareStoreOutput(
            block_hashes_to_store=[6, 7, 8],
            store_block_ids=[3, 2, 1],
            block_hashes_evicted=[2, 3, 4],
        ),
    )

    # complete store [6, 7, 8]
    cpu_manager.complete_store(to_hashes([6, 7, 8]))

    # touch [5, 6, 7] (move to end of LRU order)
    cpu_manager.touch(to_hashes([5, 6, 7]))

    # prepare store [7, 9] -> evicts [8] (oldest following previous touch)
    prepare_store_output = cpu_manager.prepare_store(to_hashes([9]))
    verify_store_output(
        prepare_store_output,
        ExpectedPrepareStoreOutput(
            block_hashes_to_store=[9],
            store_block_ids=[1],
            block_hashes_evicted=[8],
        ),
    )

    # complete store [7, 9] with failure
    cpu_manager.complete_store(to_hashes([7, 9]), success=False)

    # assert [7] is still stored, but [9] is not
    assert cpu_manager.lookup(to_hashes([7])) == 1
    assert cpu_manager.lookup(to_hashes([9])) == 0

    verify_events(
        cpu_manager.take_events(),
        block_size=block_size,
        expected_stores=({3, 4, 5}, {6, 7, 8}),
        expected_evictions=({2, 3, 4}, {8}),
    )


def test_arc_manager_basic():
    """
    Tests ARCOffloadingManager basic operations with a CPUBackend.
    Verifies that ARC handles store, load, and lookup operations correctly.
    """
    # initialize a CPU backend with a capacity of 4 blocks
    block_size = 256
    cpu_backend = CPUBackend(block_size=block_size, num_blocks=4)
    arc_manager = ARCOffloadingManager(cpu_backend, enable_events=True)

    # prepare store [1, 2]
    prepare_store_output = arc_manager.prepare_store(to_hashes([1, 2]))
    verify_store_output(
        prepare_store_output,
        ExpectedPrepareStoreOutput(
            block_hashes_to_store=[1, 2],
            store_block_ids=[0, 1],
            block_hashes_evicted=[],
        ),
    )

    # lookup [1, 2] -> not ready
    assert arc_manager.lookup(to_hashes([1, 2])) == 0

    # no events so far
    assert list(arc_manager.take_events()) == []

    # complete store [1, 2]
    arc_manager.complete_store(to_hashes([1, 2]))
    verify_events(
        arc_manager.take_events(), block_size=block_size, expected_stores=({1, 2},)
    )

    # lookup [1, 2]
    assert arc_manager.lookup(to_hashes([1])) == 1
    assert arc_manager.lookup(to_hashes([1, 2])) == 2
    assert arc_manager.lookup(to_hashes([1, 2, 3])) == 2

    # blocks should be in T1 (recent)
    assert len(arc_manager.t1) == 2
    assert len(arc_manager.t2) == 0


def test_arc_manager_t1_to_t2_promotion():
    """
    Tests that accessing a block in T1 promotes it to T2 (frequent).
    This is a key feature of ARC's adaptive behavior.
    """
    block_size = 256
    cpu_backend = CPUBackend(block_size=block_size, num_blocks=4)
    arc_manager = ARCOffloadingManager(cpu_backend, enable_events=False)

    # store and complete block 1
    arc_manager.prepare_store(to_hashes([1]))
    arc_manager.complete_store(to_hashes([1]))

    # block 1 starts in T1 (recent)
    assert to_hashes([1])[0] in arc_manager.t1
    assert to_hashes([1])[0] not in arc_manager.t2

    # touch block 1 (simulate second access)
    arc_manager.touch(to_hashes([1]))

    # block 1 should now be in T2 (frequent)
    assert to_hashes([1])[0] not in arc_manager.t1
    assert to_hashes([1])[0] in arc_manager.t2


def test_arc_manager_eviction_with_load():
    """
    Tests ARC eviction behavior similar to LRU test.
    Verifies that blocks being loaded (ref_cnt > 0) cannot be evicted.
    """
    block_size = 256
    cpu_backend = CPUBackend(block_size=block_size, num_blocks=4)
    arc_manager = ARCOffloadingManager(cpu_backend, enable_events=True)

    # prepare and complete store [1, 2, 3, 4]
    prepare_store_output = arc_manager.prepare_store(to_hashes([1, 2, 3, 4]))
    verify_store_output(
        prepare_store_output,
        ExpectedPrepareStoreOutput(
            block_hashes_to_store=[1, 2, 3, 4],
            store_block_ids=[0, 1, 2, 3],
            block_hashes_evicted=[],
        ),
    )
    arc_manager.complete_store(to_hashes([1, 2, 3, 4]))

    # prepare load [2, 3] (increases ref_cnt)
    prepare_load_output = arc_manager.prepare_load(to_hashes([2, 3]))
    verify_load_output(prepare_load_output, [1, 2])

    # prepare store [5, 6, 7] with [2, 3] being loaded
    # should fail because [2, 3] have ref_cnt > 0
    assert arc_manager.prepare_store(to_hashes([5, 6, 7])) is None

    # complete load [2, 3]
    arc_manager.complete_load(to_hashes([2, 3]))

    # now prepare store [5, 6, 7] should succeed
    # ARC will evict blocks one at a time from T1 as needed
    prepare_store_output = arc_manager.prepare_store(to_hashes([5, 6, 7]))
    assert prepare_store_output is not None
    # Should successfully evict enough blocks to make room (at least 1)
    assert len(prepare_store_output.block_hashes_evicted) >= 1


def test_arc_manager_adaptive_target():
    """
    Tests ARC's adaptive target adjustment via ghost lists.
    When a block in B1 (ghost list) is accessed, target_t1_size increases.
    When a block in B2 is accessed, target_t1_size decreases.
    """
    block_size = 256
    cpu_backend = CPUBackend(block_size=block_size, num_blocks=2)
    arc_manager = ARCOffloadingManager(cpu_backend, enable_events=False)

    # store blocks 1, 2 (fills cache)
    arc_manager.prepare_store(to_hashes([1, 2]))
    arc_manager.complete_store(to_hashes([1, 2]))

    initial_target = arc_manager.target_t1_size

    # store block 3, evicting block 1 (moves to B1 ghost list)
    arc_manager.prepare_store(to_hashes([3]))
    arc_manager.complete_store(to_hashes([3]))

    # block 1 should be in B1 (ghost list)
    assert to_hashes([1])[0] in arc_manager.b1

    # touch block 1 (cache miss, but in B1)
    # this should increase target_t1_size (favor recency)
    arc_manager.touch(to_hashes([1]))

    # target should have increased
    assert arc_manager.target_t1_size > initial_target


def test_arc_manager_t1_t2_eviction_policy():
    """
    Tests that ARC evicts from T1 or T2 based on target_t1_size.
    If |T1| >= target_t1_size, evict from T1, otherwise from T2.
    """
    block_size = 256
    cpu_backend = CPUBackend(block_size=block_size, num_blocks=4)
    arc_manager = ARCOffloadingManager(cpu_backend, enable_events=False)

    # store blocks 1, 2, 3, 4
    arc_manager.prepare_store(to_hashes([1, 2, 3, 4]))
    arc_manager.complete_store(to_hashes([1, 2, 3, 4]))

    # promote blocks 3, 4 to T2 by touching them
    arc_manager.touch(to_hashes([3, 4]))

    # now: T1 = {1, 2}, T2 = {3, 4}
    assert len(arc_manager.t1) == 2
    assert len(arc_manager.t2) == 2

    # set target_t1_size to prefer evicting from T1
    # (when |T1| >= target, evict from T1)
    arc_manager.target_t1_size = 1

    # store block 5, should evict from T1 (block 1, LRU in T1)
    output = arc_manager.prepare_store(to_hashes([5]))
    assert output is not None
    assert to_hashes([1]) == output.block_hashes_evicted

    arc_manager.complete_store(to_hashes([5]))

    # block 1 should be in B1 (ghost list)
    assert to_hashes([1])[0] in arc_manager.b1
    # block 5 should be in T1
    assert to_hashes([5])[0] in arc_manager.t1


def test_arc_manager_ghost_list_bounds():
    """
    Tests that ghost lists (B1, B2) don't grow unbounded.
    They should be capped at cache_capacity.
    """
    block_size = 256
    cpu_backend = CPUBackend(block_size=block_size, num_blocks=2)
    arc_manager = ARCOffloadingManager(cpu_backend, enable_events=False)

    # fill cache with blocks 1, 2
    arc_manager.prepare_store(to_hashes([1, 2]))
    arc_manager.complete_store(to_hashes([1, 2]))

    # store many blocks to fill ghost lists
    for i in range(3, 20):
        arc_manager.prepare_store(to_hashes([i]))
        arc_manager.complete_store(to_hashes([i]))

    # ghost lists should not exceed cache_capacity
    assert len(arc_manager.b1) <= arc_manager.cache_capacity
    assert len(arc_manager.b2) <= arc_manager.cache_capacity


def test_arc_manager_touch_ordering():
    """
    Tests that touch() correctly updates access patterns.
    Similar to LRU test but verifies T1/T2 ordering.
    """
    block_size = 256
    cpu_backend = CPUBackend(block_size=block_size, num_blocks=4)
    arc_manager = ARCOffloadingManager(cpu_backend, enable_events=True)

    # store blocks 1, 2, 3, 4
    arc_manager.prepare_store(to_hashes([1, 2, 3, 4]))
    arc_manager.complete_store(to_hashes([1, 2, 3, 4]))

    # promote 3, 4 to T2
    arc_manager.touch(to_hashes([3, 4]))

    # T1 = {1, 2}, T2 = {3, 4}
    # touch [1, 3, 4] - should promote 1 to T2, and move 3,4 to end of T2
    arc_manager.touch(to_hashes([1, 3, 4]))

    # T1 = {2}, T2 = {1, 3, 4} (in that order, with 4 most recent)
    assert len(arc_manager.t1) == 1
    assert len(arc_manager.t2) == 3

    # store block 5, should evict from T1 (block 2, only one in T1)
    prepare_store_output = arc_manager.prepare_store(to_hashes([5]))
    verify_store_output(
        prepare_store_output,
        ExpectedPrepareStoreOutput(
            block_hashes_to_store=[5],
            store_block_ids=[1],  # reuses block 2's storage
            block_hashes_evicted=[2],
        ),
    )


def test_arc_manager_failed_store():
    """
    Tests that failed store operations clean up correctly.
    Similar to LRU test but for ARC.
    """
    block_size = 256
    cpu_backend = CPUBackend(block_size=block_size, num_blocks=4)
    arc_manager = ARCOffloadingManager(cpu_backend, enable_events=True)

    # store blocks 1, 2, 3, 4
    arc_manager.prepare_store(to_hashes([1, 2, 3, 4]))
    arc_manager.complete_store(to_hashes([1, 2, 3, 4]))

    # prepare store block 5 (will evict block 1)
    prepare_store_output = arc_manager.prepare_store(to_hashes([5]))
    assert prepare_store_output is not None
    assert len(prepare_store_output.block_hashes_evicted) == 1

    # complete store with failure
    arc_manager.complete_store(to_hashes([5]), success=False)

    # block 5 should not be in cache
    assert arc_manager.lookup(to_hashes([5])) == 0
    # block 5 should not be in T1 or T2
    assert to_hashes([5])[0] not in arc_manager.t1
    assert to_hashes([5])[0] not in arc_manager.t2

    # evicted block should still be gone (in B1 ghost list)
    evicted_hash = prepare_store_output.block_hashes_evicted[0]
    assert evicted_hash in arc_manager.b1


def test_arc_manager_full_scenario():
    """
    Comprehensive test covering multiple ARC operations in sequence.
    Similar to the full LRU test but adapted for ARC behavior.
    """
    block_size = 256
    cpu_backend = CPUBackend(block_size=block_size, num_blocks=4)
    arc_manager = ARCOffloadingManager(cpu_backend, enable_events=True)

    # store [1, 2]
    arc_manager.prepare_store(to_hashes([1, 2]))
    arc_manager.complete_store(to_hashes([1, 2]))

    # store [3, 4, 5] -> evicts [1]
    prepare_store_output = arc_manager.prepare_store(to_hashes([3, 4, 5]))
    assert prepare_store_output is not None
    assert len(prepare_store_output.block_hashes_evicted) == 1
    arc_manager.complete_store(to_hashes([3, 4, 5]))

    # promote some blocks to T2
    arc_manager.touch(to_hashes([2, 3]))

    # T1 has {4, 5}, T2 has {2, 3}
    assert len(arc_manager.t1) == 2
    assert len(arc_manager.t2) == 2

    # store [6] -> should evict from T1 (4 is oldest in T1)
    prepare_store_output = arc_manager.prepare_store(to_hashes([6]))
    assert prepare_store_output is not None
    arc_manager.complete_store(to_hashes([6]))

    # verify blocks 2, 3 (in T2) are still present
    assert arc_manager.lookup(to_hashes([2])) == 1
    assert arc_manager.lookup(to_hashes([3])) == 1

    # verify events
    events = list(arc_manager.take_events())
    assert len(events) > 0  # should have store and eviction events
