# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from unittest.mock import MagicMock

import pytest

from tests.v1.kv_connector.unit.offloading_connector.utils import (
    generate_store_output,
    to_keys,
)
from tests.v1.kv_connector.unit.utils import EOS_TOKEN_ID
from vllm.distributed.kv_events import BlockRemoved, BlockStored
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import (
    OffloadingEvent,
    OffloadingManager,
    ReqContext,
    get_offload_block_hash,
)
from vllm.v1.request import RequestStatus


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_offloading_connector(request_runner, async_scheduling: bool):
    offloaded_block_size = 12
    gpu_block_size = 4
    num_gpu_blocks = 100
    block_size_factor = offloaded_block_size // gpu_block_size

    runner = request_runner(
        offloaded_block_size=offloaded_block_size,
        gpu_block_size=gpu_block_size,
        num_gpu_blocks=num_gpu_blocks,
        async_scheduling=async_scheduling,
    )

    # 3 blocks, store just the middle block (skip first and last)
    # blocks = [0, 1, 2], [3, 4, 5], [6, 7, 8]
    runner.new_request(token_ids=[0] * offloaded_block_size * 3)
    runner.manager.prepare_store.side_effect = (
        lambda keys, req_context: generate_store_output(list(keys)[1:2])
    )
    runner.run(decoded_tokens=[0])

    # add block missing 1 token -> no offload
    runner.run(
        decoded_tokens=[0] * (offloaded_block_size - 1),
        expected_stored_gpu_block_indexes=(3, 4, 5),
    )
    runner.manager.prepare_store.assert_not_called()

    # +1 token -> single block, fail prepare_store
    runner.manager.prepare_store.side_effect = lambda keys, req_context: None
    runner.run(decoded_tokens=[0])
    runner.manager.prepare_store.assert_called()

    # 1 more block (+ token for async scheduling)
    # now set block_hashes_to_store = []
    runner.manager.prepare_store.side_effect = (
        lambda keys, req_context: generate_store_output([])
    )
    runner.run(decoded_tokens=[0] * (offloaded_block_size + 1))

    # 1 more block (+ token for kicking off offloading)
    # now check touch was called with all 6 blocks
    runner.manager.prepare_store.side_effect = (
        lambda keys, req_context: generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[0] * (offloaded_block_size + 1),
        expected_stored_gpu_block_indexes=(15, 16, 17),
    )
    runner.manager.touch.assert_called()
    block_hashes1 = list(runner.manager.touch.call_args.args[0])
    assert len(block_hashes1) == 6

    # terminate request
    runner.run(decoded_tokens=[EOS_TOKEN_ID])

    # create a new request differing only on the last token
    runner.new_request(token_ids=[0] * (offloaded_block_size * 6 - 1) + [1])
    runner.run(decoded_tokens=[0])
    runner.manager.touch.assert_called()
    block_hashes2 = list(runner.manager.touch.call_args.args[0])
    assert len(block_hashes2) == 6

    # verify hashes are the same, except for the last block
    assert block_hashes1[:5] == block_hashes2[:5]
    assert block_hashes1[5] != block_hashes2[5]

    # terminate request
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored_gpu_block_indexes=tuple(range(6 * block_size_factor)),
    )

    # full_block_tokens - num_computed_tokens < offloaded_block_size
    runner.new_request(
        token_ids=[0] * gpu_block_size + [1] * (offloaded_block_size - gpu_block_size)
    )
    runner.manager.prepare_store.side_effect = (
        lambda keys, req_context: generate_store_output([])
    )
    runner.run(decoded_tokens=[EOS_TOKEN_ID])
    runner.manager.lookup.assert_not_called()

    # single block lookup with no hits
    runner.new_request(token_ids=[1] * offloaded_block_size)
    runner.manager.prepare_store.side_effect = (
        lambda keys, req_context: generate_store_output([])
    )
    runner.run(decoded_tokens=[EOS_TOKEN_ID])
    runner.manager.lookup.assert_called_once()

    # single block lookup with a hit
    runner.scheduler.reset_prefix_cache()
    runner.new_request(token_ids=[0] * offloaded_block_size)
    runner.manager.prepare_store.side_effect = (
        lambda keys, req_context: generate_store_output([])
    )
    runner.connector_scheduler._maximal_prefix_lookup = lambda key, req_context: 1
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID], expected_loaded_gpu_block_indexes=(0, 1, 2)
    )

    # single block lookup with a hit in a middle block
    runner.new_request(
        token_ids=[0] * offloaded_block_size * 2 + [1] * offloaded_block_size
    )
    runner.manager.prepare_store.side_effect = (
        lambda keys, req_context: generate_store_output([])
    )
    runner.connector_scheduler._maximal_prefix_lookup = lambda key, req_context: 1
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID], expected_loaded_gpu_block_indexes=(3, 4, 5)
    )

    # test take_events
    def to_hashes(int_hashes: list[int]) -> list[BlockHash]:
        return [BlockHash(str(i).encode()) for i in int_hashes]

    def take_events() -> Iterable[OffloadingEvent]:
        yield OffloadingEvent(keys=to_keys([1, 2, 3]), medium="A", removed=False)
        yield OffloadingEvent(keys=to_keys([4, 5, 6]), medium="B", removed=True)

    runner.manager.take_events.side_effect = take_events
    events = list(runner.scheduler_connector.take_events())
    assert len(events) == 2
    event = events[0]
    assert isinstance(event, BlockStored)
    assert event.block_hashes == to_hashes([1, 2, 3])
    assert event.block_size == 0
    assert event.medium == "A"
    assert event.token_ids == []
    assert event.parent_block_hash is None
    assert event.lora_id is None
    assert event.lora_name is None
    event = events[1]
    assert isinstance(event, BlockRemoved)
    assert event.block_hashes == to_hashes([4, 5, 6])
    assert event.medium == "B"


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_request_preemption(request_runner, async_scheduling: bool):
    offloaded_block_size = 12
    gpu_block_size = 4
    num_gpu_blocks = 100

    runner = request_runner(
        offloaded_block_size=offloaded_block_size,
        gpu_block_size=gpu_block_size,
        num_gpu_blocks=num_gpu_blocks,
        async_scheduling=async_scheduling,
    )

    free_block_queue = runner.scheduler.kv_cache_manager.block_pool.free_block_queue
    num_free_blocks_empty = free_block_queue.num_free_blocks

    # 2 blocks, store all, without flushing
    # blocks = [0, 1, 2], [3, 4, 5]
    runner.new_request(token_ids=[0] * offloaded_block_size * 2)
    runner.manager.prepare_store.side_effect = (
        lambda keys, req_context: generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[0],
        complete_transfers=False,
    )

    # decode 2 more blocks - 1 gpu block, storing [6, 7, 8] (no flush)
    runner.manager.prepare_store.side_effect = (
        lambda keys, req_context: generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[0] * (2 * offloaded_block_size - gpu_block_size),
        complete_transfers=False,
    )

    # simulate KV cache running out of space
    free_block_queue.num_free_blocks = 0

    # request should be preempted now
    runner.run(
        decoded_tokens=[],
        complete_transfers=False,
        expected_flushed_gpu_block_indexes=(0, 1, 2, 3, 4, 5, 6, 7, 8),
        expected_stored_gpu_block_indexes=(0, 1, 2, 3, 4, 5, 6, 7, 8),
    )

    # restore KV cache space and reset GPU prefix cache
    free_block_queue.num_free_blocks = num_free_blocks_empty
    runner.scheduler.reset_prefix_cache()

    # request should now return from preemption
    # re-load [0, ..., 8] from the CPU and store [9, 10, 11]
    runner.connector_scheduler._maximal_prefix_lookup = lambda key, req_context: 3
    runner.manager.prepare_store.side_effect = (
        lambda keys, req_context: generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[0] * gpu_block_size,
        expected_loaded_gpu_block_indexes=(0, 1, 2, 3, 4, 5, 6, 7, 8),
    )

    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored_gpu_block_indexes=(9, 10, 11),
    )


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_concurrent_lookups_of_the_same_prefix(request_runner, async_scheduling: bool):
    offloaded_block_size = 12
    gpu_block_size = 4
    num_gpu_blocks = 100

    runner = request_runner(
        offloaded_block_size=offloaded_block_size,
        gpu_block_size=gpu_block_size,
        num_gpu_blocks=num_gpu_blocks,
        async_scheduling=async_scheduling,
    )

    # store 1 blocks
    runner.new_request(token_ids=[0] * offloaded_block_size)
    runner.manager.prepare_store.side_effect = (
        lambda keys, req_context: generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored_gpu_block_indexes=(0, 1, 2),
    )

    # start a request to load the first block, but don't complete
    runner.scheduler.reset_prefix_cache()
    runner.new_request(token_ids=[0] * offloaded_block_size)
    runner.connector_scheduler._maximal_prefix_lookup = lambda key, req_context: 1
    runner.run(
        decoded_tokens=[],
        complete_transfers=False,
    )

    # request triggered a load
    transfer_jobs = list(runner.offloading_spec.handler.transfer_specs)
    assert transfer_jobs

    # start a new request to load the same first block
    runner.new_request(token_ids=[0] * offloaded_block_size)
    runner.connector_scheduler._maximal_prefix_lookup = lambda key, req_context: 1
    runner.run(
        decoded_tokens=[],
        complete_transfers=False,
    )

    # request did not trigger a load
    assert transfer_jobs == list(runner.offloading_spec.handler.transfer_specs)

    # complete transfers
    runner.manager.prepare_store.side_effect = (
        lambda keys, req_context: generate_store_output([])
    )
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_loaded_gpu_block_indexes=(0, 1, 2),
    )

    # second request will use the GPU prefix cache
    assert transfer_jobs == list(runner.offloading_spec.handler.transfer_specs)


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_abort_loading_requests(request_runner, async_scheduling: bool):
    offloaded_block_size = 12
    gpu_block_size = 4
    num_gpu_blocks = 100

    runner = request_runner(
        offloaded_block_size=offloaded_block_size,
        gpu_block_size=gpu_block_size,
        num_gpu_blocks=num_gpu_blocks,
        async_scheduling=async_scheduling,
    )

    # store 1 blocks
    runner.new_request(token_ids=[0] * offloaded_block_size)
    runner.manager.prepare_store.side_effect = (
        lambda keys, req_context: generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored_gpu_block_indexes=(0, 1, 2),
    )

    # start a request to load the first block, but don't complete
    runner.scheduler.reset_prefix_cache()
    runner.new_request(token_ids=[0] * offloaded_block_size)
    runner.connector_scheduler._maximal_prefix_lookup = lambda key, req_context: 1
    runner.run(
        decoded_tokens=[],
        complete_transfers=False,
    )

    # request triggered a load
    transfer_jobs = list(runner.offloading_spec.handler.transfer_specs)
    assert transfer_jobs

    # abort request
    req_id = str(runner.req_id)
    runner.scheduler.finish_requests((req_id,), RequestStatus.FINISHED_ABORTED)

    # verify request is not deleted
    assert req_id in runner.scheduler.requests

    # complete loading request
    runner.run(
        decoded_tokens=[],
        expected_loaded_gpu_block_indexes=(0, 1, 2),
    )

    # assert request is deleted
    assert req_id not in runner.scheduler.requests


# ---------------------------------------------------------------------------
# Unit tests for _maximal_prefix_lookup / _sliding_window_lookup
# ---------------------------------------------------------------------------


def _make_scheduler_with_lookup(
    lookup_results: dict[int, bool | None],
) -> OffloadingConnectorScheduler:
    """Create an OffloadingConnectorScheduler with a mocked manager.lookup."""
    manager = MagicMock(spec=OffloadingManager)
    manager.lookup.side_effect = lambda key, req_context: lookup_results.get(
        int(get_offload_block_hash(key).decode()), False
    )

    scheduler = object.__new__(OffloadingConnectorScheduler)
    scheduler.manager = manager
    return scheduler


_EMPTY_REQ_CTX = ReqContext()


class TestMaximalPrefixLookup:
    def test_all_hit(self):
        sched = _make_scheduler_with_lookup({1: True, 2: True})
        assert sched._maximal_prefix_lookup(to_keys([1, 2]), _EMPTY_REQ_CTX) == 2

    def test_all_miss(self):
        sched = _make_scheduler_with_lookup({})
        assert sched._maximal_prefix_lookup(to_keys([1, 2]), _EMPTY_REQ_CTX) == 0

    def test_partial_prefix(self):
        sched = _make_scheduler_with_lookup({1: True, 2: True})
        assert sched._maximal_prefix_lookup(to_keys([1, 2, 3]), _EMPTY_REQ_CTX) == 2

    def test_miss_then_hit(self):
        sched = _make_scheduler_with_lookup({2: True})
        assert sched._maximal_prefix_lookup(to_keys([1, 2]), _EMPTY_REQ_CTX) == 0

    def test_single_hit(self):
        sched = _make_scheduler_with_lookup({1: True})
        assert sched._maximal_prefix_lookup(to_keys([1]), _EMPTY_REQ_CTX) == 1

    def test_empty(self):
        sched = _make_scheduler_with_lookup({})
        assert sched._maximal_prefix_lookup([], _EMPTY_REQ_CTX) == 0

    def test_none_defers(self):
        sched = _make_scheduler_with_lookup({1: None, 2: True})
        assert sched._maximal_prefix_lookup(to_keys([1, 2]), _EMPTY_REQ_CTX) is None

    def test_none_after_hit_defers(self):
        sched = _make_scheduler_with_lookup({1: True, 2: None})
        assert sched._maximal_prefix_lookup(to_keys([1, 2]), _EMPTY_REQ_CTX) is None

    def test_none_stops_at_miss(self):
        """None is treated as hit for iteration, but miss stops the scan."""
        sched = _make_scheduler_with_lookup({1: None, 2: False, 3: True})
        assert sched._maximal_prefix_lookup(to_keys([1, 2, 3]), _EMPTY_REQ_CTX) is None
        # lookup should have been called for blocks 1 and 2 (stops at miss)
        assert sched.manager.lookup.call_count == 2


class TestSlidingWindowLookup:
    def test_all_hit_exact_window(self):
        sched = _make_scheduler_with_lookup({1: True, 2: True})
        assert sched._sliding_window_lookup(to_keys([1, 2]), 2, _EMPTY_REQ_CTX) == 2

    def test_all_miss(self):
        sched = _make_scheduler_with_lookup({})
        assert sched._sliding_window_lookup(to_keys([1, 2, 3]), 1, _EMPTY_REQ_CTX) == 0

    def test_window_at_end(self):
        sched = _make_scheduler_with_lookup({2: True, 3: True})
        assert sched._sliding_window_lookup(to_keys([1, 2, 3]), 2, _EMPTY_REQ_CTX) == 3

    def test_window_in_middle(self):
        sched = _make_scheduler_with_lookup({2: True, 3: True})
        assert (
            sched._sliding_window_lookup(to_keys([1, 2, 3, 4]), 2, _EMPTY_REQ_CTX) == 3
        )

    def test_no_full_window_falls_back_to_prefix(self):
        sched = _make_scheduler_with_lookup({1: True, 2: True})
        assert sched._sliding_window_lookup(to_keys([1, 2, 3]), 3, _EMPTY_REQ_CTX) == 2

    def test_single_block_window(self):
        sched = _make_scheduler_with_lookup({2: True, 3: True})
        assert sched._sliding_window_lookup(to_keys([1, 2, 3]), 1, _EMPTY_REQ_CTX) == 3

    def test_gap_resets_consecutive(self):
        sched = _make_scheduler_with_lookup({2: True, 3: True, 4: True})
        # [1, 2, 3, 0, 4] — gap at 0 resets, window of 2 found at [2,3]
        assert (
            sched._sliding_window_lookup(to_keys([1, 2, 3, 0, 4]), 2, _EMPTY_REQ_CTX)
            == 3
        )

    def test_window_prefers_rightmost(self):
        sched = _make_scheduler_with_lookup({1: True, 2: True, 4: True, 5: True})
        # two valid windows: [1,2] at positions 0-1 and [4,5] at positions 3-4
        # scans right-to-left, finds [4,5] first
        assert (
            sched._sliding_window_lookup(to_keys([1, 2, 3, 4, 5]), 2, _EMPTY_REQ_CTX)
            == 5
        )

    def test_prefix_fallback_with_gap(self):
        sched = _make_scheduler_with_lookup({2: True, 3: True, 4: True, 5: True})
        # window of 4 not found contiguously (gap at 1)
        assert (
            sched._sliding_window_lookup(to_keys([2, 1, 3, 4, 5]), 4, _EMPTY_REQ_CTX)
            == 1
        )

    def test_empty(self):
        sched = _make_scheduler_with_lookup({})
        assert sched._sliding_window_lookup([], 1, _EMPTY_REQ_CTX) == 0

    def test_none_defers(self):
        sched = _make_scheduler_with_lookup({1: True, 2: None})
        assert sched._sliding_window_lookup(to_keys([1, 2]), 2, _EMPTY_REQ_CTX) is None

    def test_none_with_full_window_still_defers(self):
        """Even if a real window is found after a None, result is deferred."""
        # Scan right-to-left: 4(True), 3(None) resets, 2(True), 1(True) = window
        # but block 3 was None so defer_lookup is set
        sched = _make_scheduler_with_lookup({1: True, 2: True, 3: None, 4: True})
        assert (
            sched._sliding_window_lookup(to_keys([1, 2, 3, 4]), 2, _EMPTY_REQ_CTX)
            is None
        )
