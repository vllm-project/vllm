# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

import numpy as np

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import (LoadStoreSpec, OffloadingEvent,
                                         PrepareStoreOutput)
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
        prepare_store_output: Optional[PrepareStoreOutput],
        expected_prepare_store_output: ExpectedPrepareStoreOutput):
    assert prepare_store_output is not None
    assert (prepare_store_output.block_hashes_to_store == to_hashes(
        expected_prepare_store_output.block_hashes_to_store))
    assert (prepare_store_output.block_hashes_evicted == to_hashes(
        expected_prepare_store_output.block_hashes_evicted))
    store_spec = prepare_store_output.store_spec
    assert isinstance(store_spec, CPULoadStoreSpec)
    expected_array = np.array(expected_prepare_store_output.store_block_ids,
                              dtype=np.int64)
    assert np.array_equal(expected_array, store_spec.block_ids)


def verify_load_output(prepare_load_output: LoadStoreSpec,
                       expected_prepare_load_output: list[int]):
    assert isinstance(prepare_load_output, CPULoadStoreSpec)
    expected_array = np.array(expected_prepare_load_output, dtype=np.int64)
    assert np.array_equal(expected_array, prepare_load_output.block_ids)


def verify_events(events: Iterable[OffloadingEvent],
                  block_size: int,
                  expected_stores: tuple[set[int], ...] = (),
                  expected_evictions: tuple[set[int], ...] = ()):
    stores: list[set[BlockHash]] = []
    evictions: list[set[BlockHash]] = []
    for event in events:
        assert event.medium == CPULoadStoreSpec.medium()
        assert event.block_size == block_size
        if event.removed:
            evictions.append(set(event.block_hashes))
        else:
            stores.append(set(event.block_hashes))

    def to_hash_sets(
            int_sets: tuple[set[int], ...]) -> tuple[set[BlockHash], ...]:
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
        ))

    # lookup [1, 2] -> not ready
    assert cpu_manager.lookup(to_hashes([1, 2])) == 0

    # no events so far
    assert list(cpu_manager.take_events()) == []

    # complete store [1, 2]
    cpu_manager.complete_store(to_hashes([1, 2]))
    verify_events(cpu_manager.take_events(),
                  block_size=block_size,
                  expected_stores=({1, 2}, ))

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
        ))

    # verify eviction event
    verify_events(cpu_manager.take_events(),
                  block_size=block_size,
                  expected_evictions=({1}, ))

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
        ))

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
        ))

    # complete store [7, 9] with failure
    cpu_manager.complete_store(to_hashes([7, 9]), success=False)

    # assert [7] is still stored, but [9] is not
    assert cpu_manager.lookup(to_hashes([7])) == 1
    assert cpu_manager.lookup(to_hashes([9])) == 0

    verify_events(cpu_manager.take_events(),
                  block_size=block_size,
                  expected_stores=({3, 4, 5}, {6, 7, 8}),
                  expected_evictions=({2, 3, 4}, {8}))
