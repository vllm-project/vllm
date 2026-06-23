# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tests.v1.kv_connector.unit.offloading_connector.utils import (
    generate_store_output,
    to_keys,
)
from tests.v1.kv_connector.unit.utils import EOS_TOKEN_ID
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
    RequestOffloadState,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_offload.base import (
    OffloadingManager,
    OffloadPolicy,
    ReqContext,
    RequestOffloadingContext,
    get_offload_block_hash,
    make_offload_key,
)
from vllm.v1.request import RequestStatus


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_offloading_connector(request_runner, async_scheduling: bool):
    block_size = 4
    block_size_factor = 3
    offloaded_block_size = block_size * block_size_factor
    num_gpu_blocks = 100

    runner = request_runner(
        block_size=block_size,
        num_gpu_blocks=num_gpu_blocks,
        async_scheduling=async_scheduling,
        block_size_factor=block_size_factor,
    )

    # 3 blocks, store just the middle block (skip first and last)
    # blocks = [0, 1, 2], [3, 4, 5], [6, 7, 8]
    runner.new_request(token_ids=[0] * offloaded_block_size * 3)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(list(keys)[1:2])
    )
    runner.run(decoded_tokens=[0])

    # add block missing 1 token -> no offload
    runner.run(
        decoded_tokens=[0] * (offloaded_block_size - 1),
        expected_stored=(3, 4, 5),
    )
    runner.manager.touch.assert_not_called()

    # +1 token -> single block, fail prepare_store
    runner.manager.prepare_store.side_effect = lambda keys, req_context: None
    runner.run(decoded_tokens=[0])
    runner.manager.prepare_store.assert_called()

    # 1 more block (+ token for async scheduling)
    # now set block_hashes_to_store = []
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output([])
    )
    runner.run(decoded_tokens=[0] * (offloaded_block_size + 1))

    # 1 more block (+ token for kicking off offloading)
    # now check touch was called with all 6 blocks
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[0] * (offloaded_block_size + 1),
        expected_stored=(15, 16, 17),
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
        expected_stored=tuple(range(6 * block_size_factor)),
    )

    # full_block_tokens - num_computed_tokens < offloaded_block_size
    runner.new_request(
        token_ids=[0] * block_size + [1] * (offloaded_block_size - block_size)
    )
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output([])
    )
    runner.run(decoded_tokens=[EOS_TOKEN_ID])
    runner.manager.lookup.assert_not_called()

    # single block lookup with no hits
    runner.new_request(token_ids=[1] * offloaded_block_size)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output([])
    )
    runner.run(decoded_tokens=[EOS_TOKEN_ID])
    runner.manager.lookup.assert_called_once()

    # single block lookup with a hit
    runner.scheduler.reset_prefix_cache()
    runner.new_request(token_ids=[0] * offloaded_block_size)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output([])
    )
    runner.connector_scheduler._maximal_prefix_lookup = lambda key, req_context: 1
    runner.run(decoded_tokens=[EOS_TOKEN_ID], expected_loaded=(0, 1, 2))

    # single block lookup with a hit in a middle block
    runner.new_request(
        token_ids=[0] * offloaded_block_size * 2 + [1] * offloaded_block_size
    )
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output([])
    )
    runner.connector_scheduler._maximal_prefix_lookup = lambda key, req_context: 1
    runner.run(decoded_tokens=[EOS_TOKEN_ID], expected_loaded=(3, 4, 5))


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_request_preemption(request_runner, async_scheduling: bool):
    block_size = 4
    block_size_factor = 3
    offloaded_block_size = block_size * block_size_factor
    num_gpu_blocks = 100

    runner = request_runner(
        block_size=block_size,
        num_gpu_blocks=num_gpu_blocks,
        async_scheduling=async_scheduling,
        block_size_factor=block_size_factor,
    )

    free_block_queue = runner.scheduler.kv_cache_manager.block_pool.free_block_queue
    num_free_blocks_empty = free_block_queue.num_free_blocks

    # 2 blocks, store all, without flushing
    # blocks = [0, 1, 2], [3, 4, 5]
    runner.new_request(token_ids=[0] * offloaded_block_size * 2)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[0],
        complete_transfers=False,
    )

    # decode 2 more blocks - 1 gpu block, storing [6, 7, 8] (no flush)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[0] * (2 * offloaded_block_size - block_size),
        complete_transfers=False,
    )

    # simulate KV cache running out of space
    free_block_queue.num_free_blocks = 0

    # request should be preempted now
    runner.run(
        decoded_tokens=[],
        complete_transfers=False,
        expected_flushed=(0, 1, 2, 3, 4, 5, 6, 7, 8),
        expected_stored=(0, 1, 2, 3, 4, 5, 6, 7, 8),
    )

    # restore KV cache space and reset GPU prefix cache
    free_block_queue.num_free_blocks = num_free_blocks_empty
    runner.scheduler.reset_prefix_cache()

    # request should now return from preemption
    # re-load [0, ..., 8] from the CPU and store [9, 10, 11]
    runner.connector_scheduler._maximal_prefix_lookup = lambda key, req_context: 3
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[0] * block_size,
        expected_loaded=(0, 1, 2, 3, 4, 5, 6, 7, 8),
    )

    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=(9, 10, 11),
    )

    # All stores completed before request_finished -> fence index empty.
    assert runner.connector_scheduler._block_id_to_pending_jobs == {}


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_no_offload_call_after_on_request_finished(
    request_runner, async_scheduling: bool
):
    """on_request_finished is not issued before a per-request offload
    call.

    A request can finish while its GPU->primary store is still in flight; the
    later worker completion then drives complete_store. The scheduler defers
    on_request_finished until the request is finished AND has no in-flight
    transfer jobs, so complete_store is observed BEFORE on_request_finished,
    and it is called exactly once.
    """
    block_size = 4
    block_size_factor = 3
    offloaded_block_size = block_size * block_size_factor

    runner = request_runner(
        block_size=block_size,
        num_gpu_blocks=100,
        async_scheduling=async_scheduling,
        block_size_factor=block_size_factor,
    )

    # Record the order of per-request connector calls on the (mocked) manager.
    # The external list survives manager.reset_mock() between run() calls.
    calls: list[tuple[str, str]] = []
    runner.manager.on_request_finished.side_effect = lambda req_context: calls.append(
        ("on_request_finished", req_context.req_id)
    )
    runner.manager.complete_store.side_effect = (
        lambda keys, req_context, *args, **kwargs: calls.append(
            ("complete_store", req_context.req_id)
        )
    )
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )

    # Decode a couple of blocks, keeping every transfer in flight
    # (complete_transfers=False) so no store completes while the request runs.
    runner.new_request(token_ids=[0] * offloaded_block_size * 2)
    runner.run(decoded_tokens=[0], complete_transfers=False)
    runner.run(
        decoded_tokens=[0] * (2 * offloaded_block_size),
        complete_transfers=False,
    )

    # Finish the request, completing its pending stores. on_request_finished is
    # deferred until the stores drain, so it lands after the last complete_store.
    # 4 offloaded blocks are stored (2 prompt + 2 decode) -> 4 * block_size_factor
    # GPU blocks.
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=tuple(range(4 * block_size_factor)),
    )

    req_id = str(runner.req_id)

    # on_request_finished is issued exactly once.
    assert calls.count(("on_request_finished", req_id)) == 1, calls

    finished_idx = calls.index(("on_request_finished", req_id))
    store_indices = [i for i, c in enumerate(calls) if c == ("complete_store", req_id)]

    # All of the request's complete_store calls must precede its single
    # on_request_finished.
    assert store_indices, calls
    assert max(store_indices) < finished_idx, calls


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_concurrent_lookups_of_the_same_prefix(request_runner, async_scheduling: bool):
    block_size = 4
    block_size_factor = 3
    offloaded_block_size = block_size * block_size_factor
    num_gpu_blocks = 100

    runner = request_runner(
        block_size=block_size,
        num_gpu_blocks=num_gpu_blocks,
        async_scheduling=async_scheduling,
        block_size_factor=block_size_factor,
    )

    # store 1 blocks
    runner.new_request(token_ids=[0] * offloaded_block_size)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=(0, 1, 2),
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
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output([])
    )
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_loaded=(0, 1, 2),
    )

    # second request will use the GPU prefix cache
    assert transfer_jobs == list(runner.offloading_spec.handler.transfer_specs)

    # Fence index drained: stores completed before request_finished ran.
    assert runner.connector_scheduler._block_id_to_pending_jobs == {}


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_abort_loading_requests(request_runner, async_scheduling: bool):
    block_size = 4
    block_size_factor = 3
    offloaded_block_size = block_size * block_size_factor
    num_gpu_blocks = 100

    runner = request_runner(
        block_size=block_size,
        num_gpu_blocks=num_gpu_blocks,
        async_scheduling=async_scheduling,
        block_size_factor=block_size_factor,
    )

    # store 1 blocks
    runner.new_request(token_ids=[0] * offloaded_block_size)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=(0, 1, 2),
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
        expected_loaded=(0, 1, 2),
    )

    # assert request is deleted
    assert req_id not in runner.scheduler.requests


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_two_groups_full_and_sliding_window(request_runner, async_scheduling: bool):
    block_size = 4
    num_gpu_blocks = 100
    # sliding_window=8 -> 2 offloaded blocks (block_size_factor=1)
    sliding_window = 8

    kv_cache_groups = [
        KVCacheGroupSpec(
            ["layer0"],
            FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
            ),
        ),
        KVCacheGroupSpec(
            ["layer1"],
            SlidingWindowSpec(
                block_size=block_size,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
                sliding_window=sliding_window,
            ),
        ),
    ]

    runner = request_runner(
        block_size=block_size,
        num_gpu_blocks=num_gpu_blocks,
        async_scheduling=async_scheduling,
        kv_cache_groups=kv_cache_groups,
    )

    # Verify group configs: group 0 = full attention, group 1 = sliding window
    kv_group_configs = runner.connector_scheduler.config.kv_group_configs
    assert len(kv_group_configs) == 2
    assert kv_group_configs[0].sliding_window_size_in_blocks is None
    assert kv_group_configs[1].sliding_window_size_in_blocks == 2

    # Blocks [0, 1, 2] miss
    runner.new_request(token_ids=[0] * block_size * 3)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(decoded_tokens=[0])
    # _touch called from get_num_new_matched_tokens (2 groups) and
    # _get_reqs_to_store (2 groups) → 4 touch calls total.
    touch_calls = runner.manager.touch.call_args_list
    assert len(touch_calls) == 4
    assert len(touch_calls[0].args[0]) == 3
    assert len(touch_calls[1].args[0]) == 3
    assert len(touch_calls[2].args[0]) == 3
    assert len(touch_calls[3].args[0]) == 3

    # store 3 more block
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[0] * (block_size * 3 + 2),
        expected_stored=(0, 1, 2, 3, 4, 5),
    )

    # touch called from _get_reqs_to_store * 3 blocks, once for each group
    touch_calls = runner.manager.touch.call_args_list
    assert len(touch_calls) == 6

    runner.run(decoded_tokens=[EOS_TOKEN_ID])

    runner.scheduler.reset_prefix_cache()

    # full 3 blocks hit [0, 1, 2]
    runner.new_request(token_ids=[0] * (block_size * 3 + 1))
    runner.manager.lookup.return_value = True
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        # Group 0 (full attn): prefix lookup hits 3 → loads blocks 0,1,2
        # Group 1 (sliding window, window=2): only the last 2 blocks
        #   are within the window → loads blocks 1,2
        expected_loaded=((0, 0), (0, 1), (0, 2), (1, 1), (1, 2)),
    )

    # one touch in get_num_new_matched_tokens x 2 groups
    touch_calls = runner.manager.touch.call_args_list
    assert len(touch_calls) == 2
    # full attention group touched all 3 blocks
    assert len(touch_calls[0].args[0]) == 3
    # sliding window group touched just the last 2 blocks
    assert len(touch_calls[1].args[0]) == 2

    # 3 blocks are hit on GPU [0, 1, 2]
    # 1 block loaded [3,]
    runner.new_request(token_ids=[0] * (block_size * 4 + 1))
    runner.manager.lookup.return_value = True
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        # Group 0 (full attn): prefix lookup hits 3 → loads blocks 0,1,2
        # Group 1 (sliding window, window=2): only the last 2 blocks
        #   are within the window → loads blocks 1,2
        expected_loaded=((0, 3), (1, 3)),
    )


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_two_groups_different_block_sizes(request_runner, async_scheduling: bool):
    hash_block_size = 4
    num_gpu_blocks = 100

    # Group 0: block_size=12 (offloaded_block_size=12)
    # Group 1: block_size=16 (offloaded_block_size=16)
    kv_cache_groups = [
        KVCacheGroupSpec(
            ["layer0"],
            FullAttentionSpec(
                block_size=hash_block_size * 3,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
            ),
        ),
        KVCacheGroupSpec(
            ["layer1"],
            FullAttentionSpec(
                block_size=hash_block_size * 4,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
            ),
        ),
    ]

    runner = request_runner(
        block_size=hash_block_size,
        num_gpu_blocks=num_gpu_blocks,
        async_scheduling=async_scheduling,
        kv_cache_groups=kv_cache_groups,
    )

    # Verify group configs
    kv_group_configs = runner.connector_scheduler.config.kv_group_configs
    assert len(kv_group_configs) == 2
    assert kv_group_configs[0].gpu_block_size == 12
    assert kv_group_configs[0].offloaded_block_size == 12
    assert kv_group_configs[1].gpu_block_size == 16
    assert kv_group_configs[1].offloaded_block_size == 16

    # Prompt: 25 tokens, unaligned to both block sizes.
    # Group 0 blocks: [0, 1], ending_token_offset = 24
    # Group 1 blocks: [0,], ending_token_offset = 16
    runner.new_request(token_ids=[0] * 25)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(decoded_tokens=[0])
    # _touch called from get_num_new_matched_tokens (2 groups) and
    # _get_reqs_to_store (2 groups) → 4 touch calls total.
    # Group 0 has 2 offload keys, group 1 has 1.
    touch_calls = runner.manager.touch.call_args_list
    assert len(touch_calls) == 4
    assert len(touch_calls[0].args[0]) == 2
    assert len(touch_calls[1].args[0]) == 1
    assert len(touch_calls[2].args[0]) == 2
    assert len(touch_calls[3].args[0]) == 1

    # Get to 31 tokens
    # No further blocks offloaded
    runner.run(decoded_tokens=[0] * 6, expected_stored=((0, 0), (0, 1), (1, 0)))

    # Get to 32 tokens
    # Group 0 blocks: [0, 1], ending_token_offset = 24
    # Group 1 blocks: [0, 1], ending_token_offset = 32
    runner.run(decoded_tokens=[0])
    # _get_reqs_to_store touch: only group 1 has a new block to store
    touch_calls = runner.manager.touch.call_args_list
    assert len(touch_calls) == 2
    assert len(touch_calls[0].args[0]) == 2
    assert len(touch_calls[1].args[0]) == 2

    # Get to 35 tokens
    # No further blocks offloaded
    runner.run(decoded_tokens=[0] * 3, expected_stored=((1, 1),))

    # Get to 36 tokens
    # Group 0 blocks: [0, 1, 2], ending_token_offset = 36
    # Group 1 blocks: [0, 1], ending_token_offset = 32
    runner.run(decoded_tokens=[0])
    # _get_reqs_to_store touch: only group 0 has a new block to store
    touch_calls = runner.manager.touch.call_args_list
    assert len(touch_calls) == 2
    assert len(touch_calls[0].args[0]) == 3
    assert len(touch_calls[1].args[0]) == 2

    # Get to 47 tokens
    # No further blocks offloaded
    runner.run(decoded_tokens=[0] * 11, expected_stored=((0, 2),))

    # Get to 48 tokens
    # Group 0 blocks: [0, 1, 2, 3], ending_token_offset = 4
    # Group 1 blocks: [0, 1, 2], ending_token_offset = 48
    runner.run(decoded_tokens=[0])
    # _get_reqs_to_store touch: both groups have a new block, each with 1 key
    touch_calls = runner.manager.touch.call_args_list
    assert len(touch_calls) == 2
    assert len(touch_calls[0].args[0]) == 4
    assert len(touch_calls[1].args[0]) == 3

    runner.run(decoded_tokens=[0], expected_stored=((0, 3), (1, 2)))

    # Get to 96 tokens
    runner.run(
        decoded_tokens=[0] * 47 + [EOS_TOKEN_ID],
        expected_stored=((0, 4), (0, 5), (0, 6), (0, 7), (1, 3), (1, 4), (1, 5)),
    )

    runner.scheduler.reset_prefix_cache()

    # Request with 48 matching tokens
    # will match 48 tokens (4 block) from the first group
    # 48 tokens (3 block) from the second group
    # Total 48 tokens can be loaded
    runner.new_request(token_ids=[0] * 48)
    runner.manager.lookup.return_value = True
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output([])
    )
    runner.run(
        decoded_tokens=[0],
        expected_loaded=((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)),
    )
    runner.run(decoded_tokens=[EOS_TOKEN_ID])

    # Request with 48+37 matching tokens
    # 48 tokens will be hit on GPU
    # extra 32 tokens will be loaded
    # extra tokens [0, 36] (blocks [4, 5, 6]) from the first group
    # extra tokens [0, 32] (block [3, 4]) from the second group
    runner.new_request(token_ids=[0] * (48 + 37))
    runner.manager.lookup.return_value = True
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output([])
    )
    runner.run(
        decoded_tokens=[0],
        expected_loaded=((0, 4), (0, 5), (0, 6), (1, 3), (1, 4)),
    )
    runner.run(decoded_tokens=[EOS_TOKEN_ID])


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


_EMPTY_REQ_CTX = ReqContext(req_id="")


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


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_request_level_policy_stores_all_blocks(request_runner, async_scheduling: bool):
    """With REQUEST_LEVEL policy, all blocks are stored — including prefix hits."""
    gpu_block_size = 4
    block_size_factor = 3
    offloaded_block_size = gpu_block_size * block_size_factor
    num_gpu_blocks = 100

    runner = request_runner(
        block_size_factor=block_size_factor,
        block_size=gpu_block_size,
        num_gpu_blocks=num_gpu_blocks,
        async_scheduling=async_scheduling,
    )

    # Store 1 offloaded block (3 GPU blocks) via a normal request.
    runner.new_request(token_ids=[0] * offloaded_block_size)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=(0, 1, 2),
    )

    # Reset GPU prefix cache so the next request must load from CPU.
    runner.scheduler.reset_prefix_cache()

    # Manager returns REQUEST_LEVEL for the next request.
    runner.manager.on_new_request.return_value = RequestOffloadingContext(
        policy=OffloadPolicy.REQUEST_LEVEL
    )

    # New request with 2 offloaded blocks; first matches what's in CPU.
    runner.new_request(token_ids=[0] * offloaded_block_size * 2)
    runner.connector_scheduler._maximal_prefix_lookup = lambda key, req_context: 1
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )

    # Load the first offloaded block from CPU.
    runner.run(decoded_tokens=[0], expected_loaded=(0, 1, 2))

    # Store must include ALL 6 GPU blocks (both the loaded prefix and
    # the newly computed block), not just the 3 new ones.
    runner.run(decoded_tokens=[EOS_TOKEN_ID], expected_stored=(0, 1, 2, 3, 4, 5))

    # All stores completed before request_finished -> fence index empty.
    assert runner.connector_scheduler._block_id_to_pending_jobs == {}


# ---------------------------------------------------------------------------
# Tests for the per-job-store-completion design and fence invariants.
# ---------------------------------------------------------------------------


def test_loads_do_not_populate_fence_index(request_runner):
    """Loads don't populate _block_id_to_pending_jobs (protected by
    delay_free_blocks while in flight)."""
    runner = request_runner(
        block_size_factor=3,
        block_size=4,
        num_gpu_blocks=100,
        async_scheduling=False,
    )
    runner.new_request(token_ids=[0] * 12)
    runner.connector_scheduler._maximal_prefix_lookup = lambda key, req_context: 1
    runner.run(decoded_tokens=[], complete_transfers=False)
    assert runner.connector_scheduler._block_id_to_pending_jobs == {}


def test_fence_at_update_state_after_alloc(request_runner):
    """A load reusing a finished request's pending-store block triggers
    a flush via update_state_after_alloc's fence.

    num_gpu_blocks=2 forces the BlockPool to give req2 the same block
    req1 just freed.
    """
    runner = request_runner(
        block_size_factor=1,
        block_size=4,
        num_gpu_blocks=2,
        async_scheduling=False,
    )

    runner.new_request(token_ids=[0] * 4)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )

    # Capture fence snapshots to verify block 0 is registered.
    fence_snapshots: list[dict] = []

    def capture_fence():
        fence_snapshots.append(
            dict(runner.connector_scheduler._block_id_to_pending_jobs)
        )

    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        complete_transfers=False,
        post_step_fn=capture_fence,
    )
    assert runner.connector_scheduler._block_id_to_pending_jobs

    # Verify fence was populated with the store job's block IDs.
    populated_fence = next((f for f in fence_snapshots if f), None)
    assert populated_fence is not None, "Fence was never populated"
    assert len(populated_fence) > 0, "Fence is empty"

    runner.scheduler.reset_prefix_cache()
    runner.new_request(token_ids=[0] * 4)
    runner.connector_scheduler._maximal_prefix_lookup = lambda key, req_context: 1
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output([])
    )
    runner.run(
        decoded_tokens=[],
        complete_transfers=False,
        expected_stored=(0,),
        expected_flushed=(0,),
    )
    assert runner.connector_scheduler._block_id_to_pending_jobs == {}


def test_fence_at_build_store_jobs(request_runner):
    """A new prefill (no load -> update_state_after_alloc returns early)
    reusing a finished request's pending-store block is flushed by
    _build_store_jobs's fence."""
    runner = request_runner(
        block_size_factor=1,
        block_size=4,
        num_gpu_blocks=2,
        async_scheduling=False,
    )

    runner.new_request(token_ids=[0] * 4)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )

    # Capture fence snapshots to verify block 0 is registered.
    fence_snapshots: list[dict] = []

    def capture_fence():
        fence_snapshots.append(
            dict(runner.connector_scheduler._block_id_to_pending_jobs)
        )

    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        complete_transfers=False,
        post_step_fn=capture_fence,
    )
    assert runner.connector_scheduler._block_id_to_pending_jobs

    # Verify fence was populated with the store job's block IDs.
    populated_fence = next((f for f in fence_snapshots if f), None)
    assert populated_fence is not None, "Fence was never populated"
    assert len(populated_fence) > 0, "Fence is empty"

    runner.scheduler.reset_prefix_cache()
    runner.new_request(token_ids=[1] * 4)
    runner.connector_scheduler._maximal_prefix_lookup = lambda key, req_context: 0
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output([])
    )
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=(0,),
        expected_flushed=(0,),
    )
    assert runner.connector_scheduler._block_id_to_pending_jobs == {}


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_complete_store_called_per_job(request_runner, async_scheduling: bool):
    """complete_store fires per-job, not deferred to request finish.
    Each call carries only that store's keys."""
    gpu_block_size = 4
    block_size_factor = 3
    offloaded_block_size = gpu_block_size * block_size_factor
    runner = request_runner(
        block_size_factor=block_size_factor,
        block_size=gpu_block_size,
        num_gpu_blocks=100,
        async_scheduling=async_scheduling,
    )
    runner.new_request(token_ids=[0] * offloaded_block_size)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )

    # First store: fires when block 0 is fully populated.
    runner.run(decoded_tokens=[0, 0], expected_stored=(0, 1, 2))
    assert runner.manager.complete_store.call_count == 1
    first_call_keys = set(runner.manager.complete_store.call_args.args[0])
    assert len(first_call_keys) == 1
    runner.manager.complete_store.reset_mock()

    # Second store: fires when block 1 is fully populated, with different keys.
    runner.run(
        decoded_tokens=[0] * (offloaded_block_size + 1),
        expected_stored=(3, 4, 5),
    )
    assert runner.manager.complete_store.call_count == 1
    second_call_keys = set(runner.manager.complete_store.call_args.args[0])
    assert first_call_keys != second_call_keys
    runner.manager.complete_store.reset_mock()

    # Finish: no store pending -> no further call.
    runner.run(decoded_tokens=[EOS_TOKEN_ID])
    assert runner.manager.complete_store.call_count == 0


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_max_offload_tokens_validation(request_runner, async_scheduling: bool):
    """Validates max_offload_tokens: type coercion, boundary values, and capping.

    Setup: 3 offloaded blocks × 3 GPU blocks each = 9 GPU block offsets (0–8).
    """
    gpu_block_size = 4
    block_size_factor = 3
    offloaded_block_size = gpu_block_size * block_size_factor  # 12
    num_gpu_blocks = 100
    all_offsets = (0, 1, 2, 3, 4, 5, 6, 7, 8)

    def make_runner():
        return request_runner(
            block_size=gpu_block_size,
            num_gpu_blocks=num_gpu_blocks,
            async_scheduling=async_scheduling,
            block_size_factor=block_size_factor,
        )

    def setup(r, max_offload_tokens):
        r.new_request(
            token_ids=[0] * offloaded_block_size * 3,
            kv_transfer_params={"max_offload_tokens": max_offload_tokens},
        )
        r.manager.prepare_store.side_effect = lambda keys, req_context: (
            generate_store_output(keys)
        )

    # Pending offloads drain via non-blocking stepping, not a flush, so no
    # blocks are flushed when the request finishes.
    flushed_all: tuple[int, ...] = ()
    flushed_two: tuple[int, ...] = ()

    # None -> no cap, all 9 offsets stored
    r = make_runner()
    setup(r, None)
    r.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=all_offsets,
        expected_flushed=flushed_all,
    )

    # str -> warn and fall back to no cap
    r = make_runner()
    setup(r, "24")
    r.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=all_offsets,
        expected_flushed=flushed_all,
    )

    # float -> warn and fall back to no cap
    r = make_runner()
    setup(r, 24.5)
    r.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=all_offsets,
        expected_flushed=flushed_all,
    )

    # negative -> warn and fall back to no cap
    r = make_runner()
    setup(r, -1)
    r.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=all_offsets,
        expected_flushed=flushed_all,
    )

    # bool -> rejected (type(True) is bool, not int), falls back to no cap
    r = make_runner()
    setup(r, True)
    r.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=all_offsets,
        expected_flushed=flushed_all,
    )

    # zero -> valid, no blocks offloaded
    r = make_runner()
    setup(r, 0)
    r.run(decoded_tokens=[EOS_TOKEN_ID], expected_stored=())

    # positive int cap -> limits offload to first 2 offloaded blocks (offsets 0–5)
    r = make_runner()
    setup(r, 24)  # 24 tokens = 2 offloaded blocks × 12 tokens each
    r.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=(0, 1, 2, 3, 4, 5),
        expected_flushed=flushed_two,
    )


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_offload_prompt_only(request_runner, async_scheduling: bool):
    """offload_prompt_only=True offloads prompt blocks but never decode blocks.

    Setup: a 2-offloaded-block prompt followed by enough decode tokens to fill
    4 more offloaded blocks. The flag clamps the offloadable token count to the
    prompt length, so only the prompt's blocks (GPU offsets 0-5) are ever
    eligible for store; the decode blocks (offsets >= 6) are skipped.

    The request is intentionally not terminated (no EOS): a store is only
    flushed when a request finishes (or is preempted), so without a finish
    there is nothing to flush and the assertion stays free of flush-timing
    subtleties. The decode steps are still enough for the prompt store to
    complete and show up in expected_stored.
    """
    gpu_block_size = 4
    block_size_factor = 3
    offloaded_block_size = gpu_block_size * block_size_factor  # 12
    num_prompt_blocks = 2
    num_decode_blocks = 4
    prompt_offsets = (0, 1, 2, 3, 4, 5)

    runner = request_runner(
        block_size=gpu_block_size,
        num_gpu_blocks=100,
        async_scheduling=async_scheduling,
        block_size_factor=block_size_factor,
        extra_config_overrides={"offload_prompt_only": True},
    )

    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )

    runner.new_request(token_ids=[0] * offloaded_block_size * num_prompt_blocks)
    runner.run(
        decoded_tokens=[0] * (offloaded_block_size * num_decode_blocks),
        expected_stored=prompt_offsets,
    )

    # Timing-independent guard: only the prompt's blocks were ever offered for
    # store. If decode blocks leaked through, more keys would appear here.
    offered_keys = {
        key
        for call in runner.manager.prepare_store.call_args_list
        for key in call.args[0]
    }
    assert len(offered_keys) == num_prompt_blocks


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_reset_cache(request_runner, async_scheduling: bool):
    """reset_cache flushes in-flight loads, calls manager.reset_cache(), resets
    next_stored_block_idx for active requests and clears job tracking."""
    block_size = 4
    block_size_factor = 3
    offloaded_block_size = block_size * block_size_factor
    num_gpu_blocks = 100

    runner = request_runner(
        block_size=block_size,
        num_gpu_blocks=num_gpu_blocks,
        async_scheduling=async_scheduling,
        block_size_factor=block_size_factor,
    )

    # Store 1 offloaded block (3 GPU blocks) to CPU.
    runner.new_request(token_ids=[0] * offloaded_block_size)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=(0, 1, 2),
    )

    # Reset GPU prefix cache then start a request that loads from CPU.
    # Leave the load in-flight so that reset_cache must flush it.
    runner.scheduler.reset_prefix_cache()
    runner.new_request(token_ids=[0] * offloaded_block_size)
    runner.connector_scheduler._maximal_prefix_lookup = lambda key, req_context: 1
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output([])
    )
    runner.run(decoded_tokens=[], complete_transfers=False)

    # Capture in-flight load job IDs before reset.
    load_job_ids = {
        jid
        for jid, status in runner.connector_scheduler._jobs.items()
        if not status.is_store
    }
    assert load_job_ids, "expected in-flight load jobs before reset"

    # Record job counter to verify the reset counter is set correctly.
    job_counter_before_reset = runner.connector_scheduler._job_counter

    # After update_state_after_alloc, next_stored_block_idx is advanced to
    # skip the loaded prefix; reset_cache must bring it back to 0.
    for req_status in runner.connector_scheduler._req_status.values():
        for group_state in req_status.group_states:
            assert group_state.next_stored_block_idx > 0

    # Reset the cache
    runner.connector_scheduler.reset_cache()

    # manager.reset_cache() must be called exactly once.
    runner.manager.reset_cache.assert_called_once()

    # In-flight load jobs must be queued for flushing to prevent CUDA stream
    # races between old loads and new post-reset stores.
    assert load_job_ids <= runner.connector_scheduler._current_batch_jobs_to_flush

    # All internal job tracking must be cleared.
    assert not runner.connector_scheduler._jobs
    assert not runner.connector_scheduler._block_id_to_pending_jobs
    if runner.connector_scheduler._blocks_being_loaded is not None:
        assert not runner.connector_scheduler._blocks_being_loaded

    # Job reset counter must equal the job counter so that completions for
    # pre-reset jobs arriving from workers are silently discarded.
    assert runner.connector_scheduler._stale_job_threshold == job_counter_before_reset

    # next_stored_block_idx must be reset to 0 for every active request so
    # that post-reset stores restart from block 0.
    for req_status in runner.connector_scheduler._req_status.values():
        for group_state in req_status.group_states:
            assert group_state.next_stored_block_idx == 0


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_reset_cache_finalizes_finished_request_with_pending_store(
    request_runner, async_scheduling: bool
):
    """reset_cache must finalize a finished request whose in-flight stores it
    discards: call on_request_finished and drop its _req_status entry.

    Otherwise the deferred hook (which waits for the now-discarded jobs to
    complete) never fires and the entry leaks.
    """
    block_size = 4
    block_size_factor = 3
    offloaded_block_size = block_size * block_size_factor

    runner = request_runner(
        block_size=block_size,
        num_gpu_blocks=100,
        async_scheduling=async_scheduling,
        block_size_factor=block_size_factor,
    )

    finalized: list[str] = []
    runner.manager.on_request_finished.side_effect = lambda req_context: (
        finalized.append(req_context.req_id)
    )
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )

    # Decode a couple of blocks and keep every transfer in flight, so the
    # request has pending store jobs.
    runner.new_request(token_ids=[0] * offloaded_block_size * 2)
    runner.run(decoded_tokens=[0], complete_transfers=False)
    runner.run(
        decoded_tokens=[0] * (2 * offloaded_block_size),
        complete_transfers=False,
    )

    cs = runner.connector_scheduler
    req_id = str(runner.req_id)
    req_status = cs._req_status[req_id]
    assert req_status.transfer_jobs, "expected an in-flight store before finish"
    assert any(job.is_store for job in cs._jobs.values())

    # Finish the request while its store is still in flight. request_finished
    # takes the defer branch (pending jobs), so on_request_finished is NOT
    # called yet and the entry stays tracked.
    req_status.req.status = RequestStatus.FINISHED_STOPPED
    cs.request_finished(req_status.req)
    assert finalized == []
    assert req_id in cs._req_status

    # reset_cache discards the in-flight store; it must finalize the request.
    cs.reset_cache()
    assert finalized == [req_id]
    assert req_id not in cs._req_status


def test_pending_transfer_defers_prefix_lookup():
    """A request with an in-flight store must not issue a load on re-admission.

    With async scheduling, a preempted request's store can be flushed by the
    worker before the scheduler consumes its completion. If the request is
    re-admitted in that window, the connector should defer it instead of
    looking up offloaded blocks and later asserting when a load is queued while
    the store job is still tracked.
    """
    scheduler = object.__new__(OffloadingConnectorScheduler)
    scheduler.manager = MagicMock(spec=OffloadingManager)

    request = SimpleNamespace(request_id="req-0")
    group_state = SimpleNamespace(block_ids=[1, 2, 3])
    req_status = SimpleNamespace(
        group_states=[group_state],
        transfer_jobs={123},
    )
    scheduler._req_status = {request.request_id: req_status}

    matched_tokens, is_async = scheduler.get_num_new_matched_tokens(
        request,
        num_computed_tokens=0,
    )

    assert matched_tokens is None
    assert is_async is False
    assert group_state.block_ids == []
    scheduler.manager.lookup.assert_not_called()


def test_async_preempt_readmit_before_transfer_output_is_deferred(request_runner):
    """A preempted request can be scheduled again before flush output is read.

    EngineCore.step_with_batch_queue() may schedule a new batch while a prior
    preemption batch is still queued. The store completion from jobs_to_flush is
    only cleared when that queued output reaches update_from_output(), so the
    re-admission path must defer while the scheduler still tracks the store.
    """
    block_size = 4
    block_size_factor = 3
    offloaded_block_size = block_size * block_size_factor

    runner = request_runner(
        block_size=block_size,
        num_gpu_blocks=100,
        async_scheduling=True,
        block_size_factor=block_size_factor,
    )
    free_block_queue = runner.scheduler.kv_cache_manager.block_pool.free_block_queue
    num_free_blocks_empty = free_block_queue.num_free_blocks

    req_id = "0"
    runner.new_request(token_ids=[0] * offloaded_block_size * 2)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )

    runner.run(decoded_tokens=[0], complete_transfers=False)
    runner.run(
        decoded_tokens=[0] * (2 * offloaded_block_size - block_size),
        complete_transfers=False,
    )

    req_status = runner.connector_scheduler._req_status[req_id]
    pending_store_jobs = set(req_status.transfer_jobs)
    assert pending_store_jobs
    assert all(
        runner.connector_scheduler._jobs[jid].is_store for jid in pending_store_jobs
    )

    free_block_queue.num_free_blocks = 0
    preempt_output = runner.scheduler.schedule()
    assert preempt_output.preempted_req_ids == {req_id}
    assert preempt_output.kv_connector_metadata is not None
    assert pending_store_jobs <= preempt_output.kv_connector_metadata.jobs_to_flush
    assert req_status.transfer_jobs == pending_store_jobs

    # Simulate the async batch-queue window: schedule again before the
    # preemption batch's ModelRunnerOutput is consumed by update_from_output().
    free_block_queue.num_free_blocks = num_free_blocks_empty
    assert runner.scheduler.reset_prefix_cache()
    runner.connector_scheduler._maximal_prefix_lookup = lambda key, req_context: len(
        key
    )

    readmit_output = runner.scheduler.schedule()

    assert readmit_output.num_scheduled_tokens == {}
    assert readmit_output.kv_connector_metadata is not None
    assert readmit_output.kv_connector_metadata.load_jobs == {}
    assert req_status.transfer_jobs == pending_store_jobs


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_swa_alignment_skip(request_runner, async_scheduling: bool):
    """SWA blocks unreachable by the load path are skipped during store.

    Simulates a DeepSeek V4-like hybrid architecture where SWA groups have
    much smaller block sizes than the full-attention (MLA) group, causing
    most SWA blocks to be unreachable by the alignment-based load path.

    Setup:
      - Group 0: full attention (MLA-like), block_size=16
      - Group 1: SWA, block_size=4, sliding_window=8

    alignment_block_count = 16 / 4 = 4 SWA blocks per alignment segment.
    sliding_window_size_in_blocks = ceil(8 / 4) = 2.
    Within each segment of 4 SWA blocks, only the trailing 2 are stored.

    With 32 tokens (2 full-attn blocks, 8 SWA blocks):
      - Group 0 stores: blocks 0, 1  (all full-attn blocks)
      - Group 1 stores: blocks 2, 3, 6, 7  (skip 0,1,4,5)

    For real DeepSeek V4 (100K tokens), this reduces SWA stores by ~78%.
    """
    full_attn_block_size = 16
    swa_block_size = 4
    sliding_window = 8
    num_gpu_blocks = 200

    kv_cache_groups = [
        KVCacheGroupSpec(
            ["layer0"],
            FullAttentionSpec(
                block_size=full_attn_block_size,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
            ),
        ),
        KVCacheGroupSpec(
            ["layer1"],
            SlidingWindowSpec(
                block_size=swa_block_size,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
                sliding_window=sliding_window,
            ),
        ),
    ]

    runner = request_runner(
        block_size=swa_block_size,
        num_gpu_blocks=num_gpu_blocks,
        async_scheduling=async_scheduling,
        kv_cache_groups=kv_cache_groups,
    )

    # Verify config: alignment_block_count computed correctly
    kv_group_configs = runner.connector_scheduler.config.kv_group_configs
    assert len(kv_group_configs) == 2
    # Group 0: full attention -> no alignment skip
    assert kv_group_configs[0].alignment_block_count is None
    assert kv_group_configs[0].sliding_window_size_in_blocks is None
    assert kv_group_configs[0].offloaded_block_size == full_attn_block_size
    # Group 1: SWA -> alignment_block_count = 16/4 = 4, tail = 2
    assert kv_group_configs[1].alignment_block_count == 4
    assert kv_group_configs[1].sliding_window_size_in_blocks == 2
    assert kv_group_configs[1].offloaded_block_size == swa_block_size

    # Send 32 tokens = 2 full-attn blocks (block_size=16) = 8 SWA blocks
    # (block_size=4). Decode 1 token to kick off processing (stores are
    # deferred to next step).
    num_tokens = 32
    runner.new_request(token_ids=[0] * num_tokens)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(decoded_tokens=[0])

    # Decode 1 more token to complete the deferred stores from above.
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        # Group 0 (full attn, block_size=16): 2 offloaded blocks
        #   -> GPU blocks (0, 0) and (0, 1)
        # Group 1 (SWA, block_size=4): 8 offloaded blocks, skip first 2
        #   per segment of 4:
        #   Segment 0 (blocks 0-3): skip 0,1 -> store (1, 2), (1, 3)
        #   Segment 1 (blocks 4-7): skip 4,5 -> store (1, 6), (1, 7)
        expected_stored=(
            (0, 0),
            (0, 1),
            (1, 2),
            (1, 3),
            (1, 6),
            (1, 7),
        ),
    )

    # Verify that loads still work correctly for the stored SWA blocks.
    runner.scheduler.reset_prefix_cache()
    runner.new_request(token_ids=[0] * num_tokens + [1])
    runner.manager.lookup.return_value = True
    runner.connector_scheduler._maximal_prefix_lookup = lambda key, req_context: 2
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        # Group 0: full prefix lookup hits 2 offloaded blocks
        #   -> loads GPU blocks (0, 0), (0, 1)
        # Group 1: sliding window lookup finds trailing 2 from last segment
        #   (blocks 6, 7 which were stored)
        #   -> loads GPU blocks (1, 6), (1, 7)
        expected_loaded=(
            (0, 0),
            (0, 1),
            (1, 6),
            (1, 7),
        ),
    )


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_stale_sliding_window_block_after_prepare_store_failure(
    request_runner, async_scheduling: bool
):
    """Regression test: when prepare_store fails (returns None), offloading is
    delayed. Meanwhile, sliding window blocks get freed and reallocated to the
    same request. On retry, the stale block_id must be detected and skipped.

    Without the fix, the stale block_id would either:
    - Cause a KeyError in _remove_pending_job (duplicate in
      _block_id_to_pending_jobs)
    - Silently offload wrong data under a wrong key
    """
    block_size = 4
    # sliding_window = 8 -> window of 2 blocks
    sliding_window = 8
    # Use a tight GPU block budget so freed sliding window blocks are
    # immediately reused by the same request's new allocations.
    num_gpu_blocks = 4

    kv_cache_groups = [
        KVCacheGroupSpec(
            ["layer0"],
            SlidingWindowSpec(
                block_size=block_size,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
                sliding_window=sliding_window,
            ),
        ),
    ]

    runner = request_runner(
        block_size=block_size,
        num_gpu_blocks=num_gpu_blocks,
        async_scheduling=async_scheduling,
        kv_cache_groups=kv_cache_groups,
    )

    # Request with 3 blocks of prompt. Window = 2 blocks, so block 0 is
    # outside the window but won't be freed until the next allocate_slots.
    runner.new_request(token_ids=[0] * block_size * 3)

    # First step: prepare_store FAILS -> offloading delayed.
    # next_stored_block_idx stays at 0, block_ids[0] still holds the
    # original block_id for position 0.
    runner.manager.prepare_store.side_effect = lambda keys, req_context: None
    runner.run(decoded_tokens=[0])
    runner.manager.prepare_store.assert_called()

    # Second step: decode more tokens -> block 3 allocated.
    # allocate_slots calls remove_skipped_blocks which frees block 0
    # (it's now outside the sliding window). With num_gpu_blocks=4,
    # the freed block is immediately reused for the new allocation.
    # prepare_store still fails so offloading is still delayed.
    runner.manager.prepare_store.side_effect = lambda keys, req_context: None
    runner.run(decoded_tokens=[0] * block_size)

    # Now prepare_store succeeds.
    # Without the fix, the request would try to offload the stale block_id
    # at position 0 (now reused at position 3), causing a duplicate in
    # sliding_window_block_ids and eventually a KeyError.
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    # block_ids=[0, ?, 3, 1]: positions 0 and 1 are zeroed (stale blocks that
    # were freed by the sliding window and reallocated). Only blocks at
    # positions 2 and 3 (request offsets 2, 3) are stored.
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=(2, 3),
    )


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_skip_reading_prefix_cache(request_runner, async_scheduling: bool):
    """When skip_reading_prefix_cache=True, the offloading connector must not
    load any blocks from CPU even if a matching prefix is cached there."""
    block_size = 4
    block_size_factor = 3
    offloaded_block_size = block_size * block_size_factor
    num_gpu_blocks = 100

    runner = request_runner(
        block_size=block_size,
        num_gpu_blocks=num_gpu_blocks,
        async_scheduling=async_scheduling,
        block_size_factor=block_size_factor,
    )

    # Populate the CPU offload cache with one block.
    runner.new_request(token_ids=[0] * offloaded_block_size)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=(0, 1, 2),
    )

    # Reset GPU prefix cache so the next request cannot hit locally.
    runner.scheduler.reset_prefix_cache()

    # New request with identical tokens but skip_reading_prefix_cache=True.
    # The offloading connector must not load anything from CPU, but must
    # still offload the freshly computed blocks (state management intact).
    runner.new_request(
        token_ids=[0] * offloaded_block_size,
        skip_reading_prefix_cache=True,
    )
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_loaded=(),  # no CPU loads must happen
        expected_stored=(0, 1, 2),  # tokens still offloaded to CPU
    )

    # The external lookup must have been completely skipped.
    runner.manager.lookup.assert_not_called()


# ---------------------------------------------------------------------------
# Eagle/MTP test class
# ---------------------------------------------------------------------------


class TestEagle:
    """Tests for Eagle/MTP speculative decoding support in the offloading
    connector scheduler — both _lookup() unit tests and integration tests."""

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _group_keys(group_idx: int, int_hashes: list[int]) -> list:
        return [make_offload_key(str(h).encode(), group_idx) for h in int_hashes]

    @staticmethod
    def _make_req_status(
        scheduler: OffloadingConnectorScheduler,
        *,
        num_tokens: int,
        num_computed_tokens: int = 0,
        offload_keys_per_group: list[list[int]],
    ) -> RequestOffloadState:
        """Build RequestOffloadState with synthetic offload keys."""
        req = MagicMock()
        req.request_id = "test-req"
        req.num_tokens = num_tokens
        req.kv_transfer_params = None

        state = RequestOffloadState(
            config=scheduler.config,
            req=req,
            req_context=ReqContext(req_id="test-req"),
            offloading_context=RequestOffloadingContext(
                policy=OffloadPolicy.BLOCK_LEVEL
            ),
            num_locally_computed_tokens=num_computed_tokens,
        )
        for idx, (gs, hashes) in enumerate(
            zip(state.group_states, offload_keys_per_group)
        ):
            gs.offload_keys = TestEagle._group_keys(
                scheduler.config.kv_group_configs[idx].group_idx, hashes
            )
        return state

    # -------------------------------------------------------------------
    # Lookup unit tests: call _lookup() directly via request_runner
    # -------------------------------------------------------------------

    def test_full_attn_lookup_pops_one_block(self, request_runner):
        """Full-attn eagle group with 3 blocks all hit → pop to 2 blocks."""
        block_size = 4
        groups = [
            KVCacheGroupSpec(
                ["layer0"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
                is_eagle_group=True,
            ),
        ]
        runner = request_runner(
            block_size=block_size,
            num_gpu_blocks=100,
            async_scheduling=False,
            kv_cache_groups=groups,
        )
        runner.manager.lookup.side_effect = lambda key, req_context: (
            int(get_offload_block_hash(key).decode()) in {1, 2, 3}
        )
        sched = runner.connector_scheduler
        req_status = self._make_req_status(
            sched, num_tokens=12, offload_keys_per_group=[[1, 2, 3]]
        )
        # 3 hits, pop to 2 → 2 * block_size = 8 tokens loadable
        assert sched._lookup(req_status) == 8

    def test_full_attn_lookup_single_block_returns_zero(self, request_runner):
        """Full-attn eagle group with 1 block hit → pop to 0 → returns 0."""
        block_size = 4
        groups = [
            KVCacheGroupSpec(
                ["layer0"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
                is_eagle_group=True,
            ),
        ]
        runner = request_runner(
            block_size=block_size,
            num_gpu_blocks=100,
            async_scheduling=False,
            kv_cache_groups=groups,
        )
        runner.manager.lookup.side_effect = lambda key, req_context: (
            int(get_offload_block_hash(key).decode()) in {1}
        )
        sched = runner.connector_scheduler
        req_status = self._make_req_status(
            sched, num_tokens=4, offload_keys_per_group=[[1]]
        )
        # 1 hit, pop to 0 → new_num_hit_tokens < block_size → return 0
        assert sched._lookup(req_status) == 0

    def test_full_attn_lookup_no_hits_returns_zero(self, request_runner):
        """Full-attn eagle group with 0 hits returns 0 before pop."""
        block_size = 4
        groups = [
            KVCacheGroupSpec(
                ["layer0"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
                is_eagle_group=True,
            ),
        ]
        runner = request_runner(
            block_size=block_size,
            num_gpu_blocks=100,
            async_scheduling=False,
            kv_cache_groups=groups,
        )
        runner.manager.lookup.return_value = False
        sched = runner.connector_scheduler
        req_status = self._make_req_status(
            sched, num_tokens=8, offload_keys_per_group=[[1, 2]]
        )
        assert sched._lookup(req_status) == 0

    def test_sw_lookup_inflates_query_max(self, request_runner):
        """SW eagle group inflates query_max so _sliding_window_lookup gets
        one extra key beyond what max_hit_size_tokens alone would yield.

        With block_size=4, W=2, eagle, num_tokens=13, 4 keys all hitting:
        - max_hit = 13-1 = 12 (SW reduction)
        - Without inflation: num_blocks = cdiv(12,4) = 3 → only 3 keys
        - With inflation: query_max = min(12+4, 4*4=16) = 16,
          num_blocks = cdiv(16,4) = 4 → 4 keys passed to SW
        - SW finds window of 3 (required=W+1=3) at idx 1 → returns 4
        - Pop: 4-1=3 → max_hit = min(12, 12) = 12. Result: 12.
        """
        block_size = 4
        sw_blocks = 2
        groups = [
            KVCacheGroupSpec(
                ["layer0"],
                SlidingWindowSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=sw_blocks * block_size,
                ),
                is_eagle_group=True,
            ),
        ]
        runner = request_runner(
            block_size=block_size,
            num_gpu_blocks=100,
            async_scheduling=False,
            kv_cache_groups=groups,
        )
        runner.manager.lookup.side_effect = lambda key, req_context: (
            int(get_offload_block_hash(key).decode()) in {1, 2, 3, 4}
        )
        sched = runner.connector_scheduler

        captured_keys: list = []
        orig_sw_lookup = type(sched)._sliding_window_lookup

        def capturing_sw_lookup(self_arg, keys, window, req_context):
            captured_keys.append(list(keys))
            return orig_sw_lookup(self_arg, keys, window, req_context)

        sched._sliding_window_lookup = lambda keys, window, req_ctx: (
            capturing_sw_lookup(sched, keys, window, req_ctx)
        )

        req_status = self._make_req_status(
            sched, num_tokens=13, offload_keys_per_group=[[1, 2, 3, 4]]
        )
        result = sched._lookup(req_status)
        assert len(captured_keys) == 1
        # Inflation bumped from 3 keys (cdiv(12,4)) to 4 keys (cdiv(16,4))
        assert len(captured_keys[0]) == 4
        # SW finds window of 3 → returns 4, pop to 3 → 3*4=12
        assert result == 12

    def test_sw_lookup_requires_extra_window_block(self, request_runner):
        """SW eagle with W=2 and only 2 keys (both hit) uses prefix fallback.

        Since required_window = W+1 = 3 but only 2 keys are available
        (inflation is capped by len(offload_keys)), _sliding_window_lookup
        can never find a window of 3. It falls back to prefix count (2).
        Pop: 2-1=1 → max_hit = 4. Result: 4 tokens (degraded from full hit).
        """
        block_size = 4
        sw_blocks = 2
        groups = [
            KVCacheGroupSpec(
                ["layer0"],
                SlidingWindowSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=sw_blocks * block_size,
                ),
                is_eagle_group=True,
            ),
        ]
        runner = request_runner(
            block_size=block_size,
            num_gpu_blocks=100,
            async_scheduling=False,
            kv_cache_groups=groups,
        )
        runner.manager.lookup.side_effect = lambda key, req_context: (
            int(get_offload_block_hash(key).decode()) in {1, 2}
        )
        sched = runner.connector_scheduler
        req_status = self._make_req_status(
            sched, num_tokens=9, offload_keys_per_group=[[1, 2]]
        )
        # Prefix fallback returns 2, pop to 1 → 1*4 = 4 tokens
        assert sched._lookup(req_status) == 4

    def test_sw_lookup_w_plus_one_hits_returns_w_blocks(self, request_runner):
        """SW eagle with W=2, 3 contiguous hits → pop to 2 → returns 2*bs."""
        block_size = 4
        sw_blocks = 2
        groups = [
            KVCacheGroupSpec(
                ["layer0"],
                SlidingWindowSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=sw_blocks * block_size,
                ),
                is_eagle_group=True,
            ),
        ]
        runner = request_runner(
            block_size=block_size,
            num_gpu_blocks=100,
            async_scheduling=False,
            kv_cache_groups=groups,
        )
        runner.manager.lookup.side_effect = lambda key, req_context: (
            int(get_offload_block_hash(key).decode()) in {1, 2, 3}
        )
        sched = runner.connector_scheduler
        # num_tokens=13 → max_hit=13-1=12, query_max=min(12+4,12)=12
        # num_blocks=cdiv(12,4)=3, keys=[1,2,3], required_window=3
        # SW finds window of 3, pop to 2 → 2*4=8
        req_status = self._make_req_status(
            sched, num_tokens=13, offload_keys_per_group=[[1, 2, 3]]
        )
        assert sched._lookup(req_status) == 8

    def test_eagle_verified_prevents_double_pop(self, request_runner):
        """Once an eagle group has popped, it doesn't pop again on re-iteration.

        Setup: group 0 = non-eagle full-attn (3 blocks), group 1 = eagle
        full-attn (3 blocks). Both see all hits. Eagle pops to 2 and tightens
        max_hit to 8. Group 0 re-runs (convergence) but since eagle_verified
        contains group 1, it won't pop again — result stays at 8 tokens.
        """
        block_size = 4
        groups = [
            KVCacheGroupSpec(
                ["layer0"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
                is_eagle_group=False,
            ),
            KVCacheGroupSpec(
                ["layer1"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=2,
                    head_size=1,
                    dtype=torch.float32,
                ),
                is_eagle_group=True,
            ),
        ]
        runner = request_runner(
            block_size=block_size,
            num_gpu_blocks=100,
            async_scheduling=False,
            kv_cache_groups=groups,
        )
        runner.manager.lookup.side_effect = lambda key, req_context: (
            int(get_offload_block_hash(key).decode()) in {1, 2, 3}
        )
        sched = runner.connector_scheduler
        req_status = self._make_req_status(
            sched,
            num_tokens=12,
            offload_keys_per_group=[[1, 2, 3], [1, 2, 3]],
        )
        # Group 0: prefix finds 3 → max_hit=12, num_hit=12
        # Group 1 (eagle): prefix finds 3, pop to 2 → max_hit=8, num_hit=8
        # num_hit(8) < prev num_hit(12) AND group IS eagle → no clear
        # No re-iteration triggered (eagle shrink doesn't trigger re-loop)
        # Final: 8 tokens
        assert sched._lookup(req_status) == 8

    def test_non_eagle_tighten_clears_eagle_verified(self, request_runner):
        """Non-eagle group tightening clears eagle_verified → eagle re-pops.

        Groups: 0=non-eagle full-attn, 1=eagle full-attn.
        Group 0 has only 1 hit (out of 3 keys) → max_hit tightens to 4.
        This clears eagle_verified. Group 1 runs with max_hit=4 → only 1
        key queried, 1 hit, pop to 0 → returns 0.
        """
        block_size = 4
        groups = [
            KVCacheGroupSpec(
                ["layer0"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
                is_eagle_group=False,
            ),
            KVCacheGroupSpec(
                ["layer1"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=2,
                    head_size=1,
                    dtype=torch.float32,
                ),
                is_eagle_group=True,
            ),
        ]
        runner = request_runner(
            block_size=block_size,
            num_gpu_blocks=100,
            async_scheduling=False,
            kv_cache_groups=groups,
        )
        # Group 0 keys [10,11,12]: only 10 hits.
        # Group 1 keys [1,2,3]: all hit.
        runner.manager.lookup.side_effect = lambda key, req_context: (
            int(get_offload_block_hash(key).decode()) in {10, 1, 2, 3}
        )
        sched = runner.connector_scheduler
        req_status = self._make_req_status(
            sched,
            num_tokens=12,
            offload_keys_per_group=[[10, 11, 12], [1, 2, 3]],
        )
        # Group 0 (non-eagle FA): prefix finds 1 hit → max_hit=4, num_hit=4
        # Group 1 (eagle FA): max_hit=4 → num_blocks=1, keys=[1].
        #   Finds 1 hit, pop to 0 → new_num_hit = 0 < block_size → return 0
        assert sched._lookup(req_status) == 0

    def test_eagle_verified_survives_eagle_tighten(self, request_runner):
        """Eagle group tightening does NOT clear eagle_verified.

        Groups: 0=non-eagle full-attn, 1=eagle full-attn.
        Group 0 finds 3 hits (max_hit=12). Group 1 finds 3 hits, pops to 2
        (max_hit=8). Since group 1 IS eagle, eagle_verified is NOT cleared.
        Result: 8 tokens (eagle only pops once).
        """
        block_size = 4
        groups = [
            KVCacheGroupSpec(
                ["layer0"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
                is_eagle_group=False,
            ),
            KVCacheGroupSpec(
                ["layer1"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=2,
                    head_size=1,
                    dtype=torch.float32,
                ),
                is_eagle_group=True,
            ),
        ]
        runner = request_runner(
            block_size=block_size,
            num_gpu_blocks=100,
            async_scheduling=False,
            kv_cache_groups=groups,
        )
        runner.manager.lookup.side_effect = lambda key, req_context: (
            int(get_offload_block_hash(key).decode()) in {1, 2, 3}
        )
        sched = runner.connector_scheduler
        req_status = self._make_req_status(
            sched,
            num_tokens=12,
            offload_keys_per_group=[[1, 2, 3], [1, 2, 3]],
        )
        # Group 0: 3 hits → max_hit=12, num_hit=12
        # Group 1 (eagle): 3 hits, pop to 2 → max_hit=8, num_hit=8
        # Tightened but IS eagle → no clear. No re-iteration.
        assert sched._lookup(req_status) == 8

    # -------------------------------------------------------------------
    # Integration tests: store and load via request_runner
    # -------------------------------------------------------------------

    @pytest.mark.parametrize("async_scheduling", [True, False])
    def test_full_attn_store_excludes_trailing_block(
        self, request_runner, async_scheduling: bool
    ):
        """Eagle full-attention group stores all blocks except the trailing
        one.

        Setup: 2 groups — group 0 is normal full-attention, group 1 is
        eagle full-attention. With a 3-block prompt, group 1 should store
        only blocks 0 and 1, skipping block 2 (the volatile tail).
        """
        block_size = 4
        block_size_factor = 1
        offloaded_block_size = block_size * block_size_factor
        num_gpu_blocks = 100

        kv_cache_groups = [
            KVCacheGroupSpec(
                ["layer0"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["layer1"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=2,
                    head_size=1,
                    dtype=torch.float32,
                ),
                is_eagle_group=True,
            ),
        ]

        runner = request_runner(
            block_size=block_size,
            num_gpu_blocks=num_gpu_blocks,
            async_scheduling=async_scheduling,
            kv_cache_groups=kv_cache_groups,
            block_size_factor=block_size_factor,
        )

        kv_group_configs = runner.connector_scheduler.config.kv_group_configs
        assert len(kv_group_configs) == 2
        assert not kv_group_configs[0].is_eagle_group
        assert kv_group_configs[1].is_eagle_group

        runner.new_request(token_ids=[0] * offloaded_block_size * 3)
        runner.manager.prepare_store.side_effect = lambda keys, req_context: (
            generate_store_output(keys)
        )
        runner.run(
            decoded_tokens=[EOS_TOKEN_ID],
            expected_stored=(
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (1, 1),
            ),
        )

    @pytest.mark.parametrize("async_scheduling", [True, False])
    def test_sw_store_excludes_trailing_block(
        self, request_runner, async_scheduling: bool
    ):
        """Eagle sliding-window group stores all blocks except the trailing
        one."""
        block_size = 4
        sliding_window = 8
        num_gpu_blocks = 100

        kv_cache_groups = [
            KVCacheGroupSpec(
                ["layer0"],
                SlidingWindowSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=sliding_window,
                ),
                is_eagle_group=True,
            ),
        ]

        runner = request_runner(
            block_size=block_size,
            num_gpu_blocks=num_gpu_blocks,
            async_scheduling=async_scheduling,
            kv_cache_groups=kv_cache_groups,
        )

        kv_group_configs = runner.connector_scheduler.config.kv_group_configs
        assert len(kv_group_configs) == 1
        assert kv_group_configs[0].is_eagle_group
        assert kv_group_configs[0].sliding_window_size_in_blocks == 2

        runner.new_request(token_ids=[0] * block_size * 3)
        runner.manager.prepare_store.side_effect = lambda keys, req_context: (
            generate_store_output(keys)
        )
        runner.run(
            decoded_tokens=[EOS_TOKEN_ID],
            expected_stored=((0, 0), (0, 1)),
        )

    @pytest.mark.parametrize("async_scheduling", [True, False])
    def test_single_block_nothing_stored(self, request_runner, async_scheduling: bool):
        """An eagle group with only one block stores nothing: that block is
        the tail."""
        block_size = 4
        block_size_factor = 1
        offloaded_block_size = block_size * block_size_factor
        num_gpu_blocks = 100

        kv_cache_groups = [
            KVCacheGroupSpec(
                ["layer0"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
                is_eagle_group=True,
            ),
        ]

        runner = request_runner(
            block_size=block_size,
            num_gpu_blocks=num_gpu_blocks,
            async_scheduling=async_scheduling,
            kv_cache_groups=kv_cache_groups,
            block_size_factor=block_size_factor,
        )

        runner.new_request(token_ids=[0] * offloaded_block_size)
        runner.manager.prepare_store.side_effect = lambda keys, req_context: (
            generate_store_output(keys)
        )
        runner.run(decoded_tokens=[EOS_TOKEN_ID], expected_stored=())
        runner.manager.prepare_store.assert_not_called()

    @pytest.mark.parametrize("async_scheduling", [True, False])
    def test_full_attn_store_then_load(self, request_runner, async_scheduling: bool):
        """Eagle group constrains load: convergence tightens both groups.

        Store 3 offloaded blocks per group (eagle group skips tail → stores
        2). Then a new request loads from CPU. The eagle group's post-pop hit
        (2) does not tighten below group 0's hit (3), so both groups load
        normally.
        """
        block_size = 4
        block_size_factor = 1
        offloaded_block_size = block_size * block_size_factor
        num_gpu_blocks = 100

        kv_cache_groups = [
            KVCacheGroupSpec(
                ["layer0"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["layer1"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=2,
                    head_size=1,
                    dtype=torch.float32,
                ),
                is_eagle_group=True,
            ),
        ]

        runner = request_runner(
            block_size=block_size,
            num_gpu_blocks=num_gpu_blocks,
            async_scheduling=async_scheduling,
            kv_cache_groups=kv_cache_groups,
            block_size_factor=block_size_factor,
        )

        runner.new_request(token_ids=[0] * offloaded_block_size * 3)
        runner.manager.prepare_store.side_effect = lambda keys, req_context: (
            generate_store_output(keys)
        )
        runner.run(
            decoded_tokens=[EOS_TOKEN_ID],
            expected_stored=(
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (1, 1),
            ),
        )

        runner.scheduler.reset_prefix_cache()

        runner.new_request(token_ids=[0] * offloaded_block_size * 3 + [1])
        runner.manager.lookup.return_value = True
        runner.manager.prepare_store.side_effect = lambda keys, req_context: (
            generate_store_output([])
        )
        runner.run(
            decoded_tokens=[EOS_TOKEN_ID],
            expected_loaded=(
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
            ),
        )


# ---------------------------------------------------------------------------
# Tests for request_finished fence population with in-flight pending stores.
# ---------------------------------------------------------------------------


def test_request_finished_with_pending_stores_populates_fence(request_runner):
    """When a request finishes with in-flight store jobs, the fence index
    (_block_id_to_pending_jobs) is correctly populated with the store jobs'
    non_sliding_window_block_ids.

    This prevents data corruption when a subsequent request reuses the same
    GPU blocks before the store completes.
    """
    block_size = 4
    block_size_factor = 1
    offloaded_block_size = block_size * block_size_factor

    # Use 2 GPU blocks so the second run reuses the same blocks,
    # triggering a fence-based flush of the in-flight job from run 1.
    runner = request_runner(
        block_size=block_size,
        num_gpu_blocks=2,
        async_scheduling=False,
        block_size_factor=block_size_factor,
    )

    # 4 prompt tokens → 1 GPU block (block 0)
    runner.new_request(token_ids=[0] * offloaded_block_size)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )

    # Capture fence state at each step to verify it was populated.
    fence_snapshots: list[dict] = []
    job_block_ids: set[int] = set()

    def capture_fence():
        fence_snapshots.append(
            dict(runner.connector_scheduler._block_id_to_pending_jobs)
        )
        for js in runner.connector_scheduler._jobs.values():
            if js.is_store:
                job_block_ids.update(js.non_sliding_window_block_ids or [])

    # Run 1: create store job, finish request, populate fence.
    # With non-blocking drain (#45595), the job stays in-flight.
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        complete_transfers=False,
        post_step_fn=capture_fence,
    )

    # Verify fence was populated at some point during the run.
    assert len(job_block_ids) > 0, "No store job was created"
    populated_fence = next((f for f in fence_snapshots if len(f) > 0), None)
    assert populated_fence is not None, "Fence was never populated"

    # Verify fence contained the job's non-SW block IDs.
    for bid in job_block_ids:
        assert bid in populated_fence, f"Block {bid} not in fence: {populated_fence}"

    # Run 2: block reuse triggers fence-based flush → cleanup.
    runner.scheduler.reset_prefix_cache()
    runner.new_request(token_ids=[0] * offloaded_block_size)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=(0,),
        expected_flushed=(0,),
    )

    # Verify fence is empty after full lifecycle (cleanup happened).
    assert runner.connector_scheduler._block_id_to_pending_jobs == {}
    # req_status should be removed.
    req_id = str(runner.req_id)
    assert req_id not in runner.connector_scheduler._req_status


def test_multiple_in_flight_stores_all_flushed_by_fence(request_runner):
    """When a request finishes with multiple in-flight store jobs,
    ALL jobs are flushed when a new request reuses their blocks.

    Uses three runner.run() calls:
    - Run 1: decode fills a block → job_0 created
    - Run 2: decode fills another block + EOS → job_1 created, request finishes
    - Run 3: block reuse → both jobs flushed via fence
    """
    block_size = 4
    block_size_factor = 1
    offloaded_block_size = block_size * block_size_factor

    # 4 GPU blocks: block 0 is null, blocks 1-3 are usable.
    runner = request_runner(
        block_size=block_size,
        num_gpu_blocks=4,
        async_scheduling=False,
        block_size_factor=block_size_factor,
    )

    # Prompt: 4 tokens → block 1
    runner.new_request(token_ids=[0] * offloaded_block_size)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )

    # Run 1: 4 decoded tokens → block 2 full → job_0 created for block 1.
    runner.run(
        decoded_tokens=[0] * offloaded_block_size,
        complete_transfers=False,
    )
    assert len(runner.connector_scheduler._jobs) >= 1

    # Run 2: 4 more tokens + EOS → block 3 full → more jobs created.
    # Request finishes → all jobs registered in fence.
    runner.run(
        decoded_tokens=[0] * offloaded_block_size + [EOS_TOKEN_ID],
        complete_transfers=False,
    )
    num_jobs = len(runner.connector_scheduler._jobs)
    assert num_jobs >= 2, f"Expected multiple in-flight jobs, got {num_jobs}"

    # Run 3: block reuse → fence flushes both jobs.
    runner.scheduler.reset_prefix_cache()
    runner.new_request(token_ids=[0] * offloaded_block_size * 3)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=(0, 1, 2),
        expected_flushed=(0, 1, 2),
    )

    # Post-condition: fence cleaned up, all jobs gone.
    assert runner.connector_scheduler._block_id_to_pending_jobs == {}
    assert len(runner.connector_scheduler._jobs) == 0


def test_request_finished_mixed_full_attn_and_sliding_window(
    request_runner,
):
    """With both FullAttention and SlidingWindow groups, a single store job
    has both non_sliding_window_block_ids and sliding_window_block_ids.

    request_finished only registers non-SW blocks in the fence.
    SW blocks were already registered at store creation time.
    """
    block_size = 4
    sliding_window = 8  # 2 blocks

    kv_cache_groups = [
        KVCacheGroupSpec(
            ["layer0"],
            FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
            ),
        ),
        KVCacheGroupSpec(
            ["layer1"],
            SlidingWindowSpec(
                block_size=block_size,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
                sliding_window=sliding_window,
            ),
        ),
    ]

    # Use 4 GPU blocks (2 per group) so run 2 reuses the same blocks,
    # triggering a fence-based flush.
    runner = request_runner(
        block_size=block_size,
        num_gpu_blocks=4,
        async_scheduling=False,
        kv_cache_groups=kv_cache_groups,
    )

    # 1 block of prompt (4 tokens) — 1 block per group.
    runner.new_request(token_ids=[0] * block_size)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )

    # Capture fence state and job block IDs at each step.
    fence_snapshots: list[dict] = []
    sw_block_ids: set[int] = set()
    non_sw_block_ids: set[int] = set()

    def capture_fence():
        fence_snapshots.append(
            dict(runner.connector_scheduler._block_id_to_pending_jobs)
        )
        for js in runner.connector_scheduler._jobs.values():
            if js.is_store:
                sw_block_ids.update(js.sliding_window_block_ids or [])
                non_sw_block_ids.update(js.non_sliding_window_block_ids or [])

    # Run 1: create store job, finish request, populate fence.
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        complete_transfers=False,
        post_step_fn=capture_fence,
    )

    # Verify job had both SW and non-SW blocks.
    assert len(sw_block_ids) > 0, "No SW blocks in store job"
    assert len(non_sw_block_ids) > 0, "No non-SW blocks in store job"

    # Find the fence snapshot where both SW and non-SW blocks were present.
    # SW blocks should appear at creation time, non-SW at request_finished.
    populated_fence = None
    for fence in fence_snapshots:
        has_sw = all(bid in fence for bid in sw_block_ids)
        has_non_sw = all(bid in fence for bid in non_sw_block_ids)
        if has_sw and has_non_sw:
            populated_fence = fence
            break

    assert populated_fence is not None, (
        f"Fence never contained both SW {sw_block_ids} and "
        f"non-SW {non_sw_block_ids} blocks. Snapshots: {fence_snapshots}"
    )

    # Run 2: block reuse triggers fence-based flush of the old job.
    runner.scheduler.reset_prefix_cache()
    runner.new_request(token_ids=[0] * block_size)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=((0, 0), (1, 0)),
        expected_flushed=((1, 0),),
    )

    # Verify fence is empty after full lifecycle (cleanup happened).
    assert runner.connector_scheduler._block_id_to_pending_jobs == {}
    assert len(runner.connector_scheduler._jobs) == 0
