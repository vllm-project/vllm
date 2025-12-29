# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Tests for correctness in invalid block handling.

These tests verify correct behavior in three scenarios:
1. Sync recompute case: Blocks should not be freed for running requests
   that need to recompute invalid blocks
2. Sync fail case: Invalid blocks must be evicted from cache when request fails
3. Async recompute case: Invalid blocks should not be cached after transfer
"""

from collections.abc import Callable
from unittest.mock import Mock

import pytest

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import FinishReason, Request, RequestStatus

from .utils import (
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)

pytestmark = pytest.mark.cpu_test


def _make_get_num_new_matched_tokens(
    req_num_new_matched_tokens: dict[str, int],
    async_load: bool,
) -> Callable[[Request, int], tuple[int, bool]]:
    def get_num_new_matched_tokens(request: Request, _: int) -> tuple[int, bool]:
        value = req_num_new_matched_tokens.get(request.request_id, 0)
        return value, async_load

    return get_num_new_matched_tokens


@pytest.fixture
def fail_scheduler():
    """scheduler with kv_load_failure_policy='fail'"""
    vllm_config = create_vllm_config()
    vllm_config.kv_transfer_config.kv_load_failure_policy = "fail"
    return create_scheduler(vllm_config)


@pytest.fixture
def recompute_scheduler():
    """scheduler with kv_load_failure_policy='recompute'"""
    vllm_config = create_vllm_config()
    vllm_config.kv_transfer_config.kv_load_failure_policy = "recompute"
    return create_scheduler(vllm_config)


def test_sync_recompute_blocks_not_freed_for_running_requests(
    recompute_scheduler: Scheduler,
):
    """
    Test sync recompute case - blocks must not be freed for running requests.

    When a running request has invalid blocks and retry_policy is 'recompute':
    1. Request should remain in RUNNING state
    2. num_computed_tokens should be truncated to invalid block boundary
    3. Blocks should NOT be freed (request still needs them for recomputation)
    4. Request should remain in scheduler.requests and scheduler.running
    """
    num_prompt_blocks = 100
    num_external_computed_blocks = 99
    invalid_block_idx = 50

    num_prompt_tokens = num_prompt_blocks * recompute_scheduler.block_size
    num_external_computed_tokens = (
        num_external_computed_blocks * recompute_scheduler.block_size
    )

    request = create_request(num_tokens=num_prompt_tokens)
    recompute_scheduler.add_request(request=request)

    req_num_new_matched_tokens = {
        request.request_id: num_external_computed_tokens,
    }

    # mock connector indicating sync load
    recompute_scheduler.connector = Mock()
    recompute_scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(req_num_new_matched_tokens, False)
    )
    recompute_scheduler.connector.request_finished.return_value = (False, None)
    recompute_scheduler.connector.take_events.return_value = ()

    scheduler_output = recompute_scheduler.schedule()

    # request should be running with sync KV load
    assert len(recompute_scheduler.running) == 1
    assert len(scheduler_output.scheduled_new_reqs) == 1
    assert request.status == RequestStatus.RUNNING

    # get the allocated block IDs before invalid blocks are reported
    req_block_ids = scheduler_output.scheduled_new_reqs[0].block_ids[0]
    invalid_block_ids = {req_block_ids[invalid_block_idx]}

    # store original num_computed_tokens for comparison
    original_num_computed_tokens = request.num_computed_tokens

    model_runner_output = create_model_runner_output(
        [request],
        invalid_block_ids=invalid_block_ids,
        use_eos=False,  # not finished - should continue running
    )

    outputs = recompute_scheduler.update_from_output(
        scheduler_output, model_runner_output
    )

    # critical assertions for recompute case:

    # 1. request should still be RUNNING (not finished, not aborted)
    assert request.status == RequestStatus.RUNNING, (
        f"Request should remain RUNNING for recompute, got {request.status}"
    )

    # 2. num_computed_tokens should be truncated to first invalid block
    expected_truncated_tokens = invalid_block_idx * recompute_scheduler.block_size
    assert request.num_computed_tokens == expected_truncated_tokens, (
        f"num_computed_tokens should be truncated to {expected_truncated_tokens}, "
        f"got {request.num_computed_tokens}"
    )
    assert request.num_computed_tokens < original_num_computed_tokens, (
        "num_computed_tokens should be reduced after invalid block detection"
    )

    # 3. no output should be generated (request is still running)
    # the request should be skipped in the output loop
    assert len(outputs) == 0 or request.request_id not in [
        out.request_id for outs in outputs.values() for out in outs.outputs
    ], "No output should be generated for recompute requests"

    # 4. request should still be in running queue
    assert request in recompute_scheduler.running, (
        "Request should remain in running queue for recomputation"
    )

    # 5. request should still be in scheduler.requests (not deleted)
    assert request.request_id in recompute_scheduler.requests, (
        "Request should not be deleted from scheduler.requests"
    )

    # 6. blocks should NOT be freed - verify blocks are still allocated
    try:
        allocated_blocks = recompute_scheduler.kv_cache_manager.get_block_ids(
            request.request_id
        )
        assert allocated_blocks is not None
        assert len(allocated_blocks[0]) > 0, (
            "Blocks should still be allocated for recomputation"
        )
    except KeyError:
        pytest.fail(
            "Blocks were freed incorrectly! Running requests need their blocks "
            "to recompute invalid portions."
        )

    # 7. Connector prefix cache stats should not be recorded yet
    # in SchedulerStats, but are recorded on the request
    assert request.connector_prefix_cache_queries == num_prompt_tokens

    # Reflect only successfully loaded blocks
    expected_hits = invalid_block_idx * recompute_scheduler.block_size
    assert request.connector_prefix_cache_hits == expected_hits, (
        f"Request hits should be {expected_hits} (only valid blocks), "
        f"got {request.connector_prefix_cache_hits}"
    )

    # Verify request can be rescheduled in next step
    scheduler_output_2 = recompute_scheduler.schedule()

    # request should appear in the new schedule to recompute invalid blocks
    scheduled_req_ids = [
        req.request_id for req in scheduler_output_2.scheduled_new_reqs
    ]
    if scheduler_output_2.num_scheduled_tokens:
        scheduled_req_ids.extend(scheduler_output_2.num_scheduled_tokens.keys())

    assert (
        request.request_id in scheduled_req_ids or len(recompute_scheduler.running) > 0
    ), "Request should be reschedulable for recomputation"


def test_sync_fail_invalid_blocks_evicted(fail_scheduler: Scheduler):
    """
    Test sync fail case - invalid blocks must be evicted from cache.

    When a request fails with policy='fail' and has invalid blocks from sync loading:
    1. Request should be finished with FINISHED_ERROR
    2. Invalid blocks should be evicted from the KV cache
    3. Valid blocks (if shared) should remain in cache
    4. Future requests should not reuse the invalid blocks

    This test verifies that invalid blocks are properly evicted to prevent
    cache corruption and reuse of invalid data.
    """
    num_prompt_blocks = 100
    num_external_computed_blocks = 99
    invalid_block_idx = 50

    num_prompt_tokens = num_prompt_blocks * fail_scheduler.block_size
    num_external_computed_tokens = (
        num_external_computed_blocks * fail_scheduler.block_size
    )

    request = create_request(num_tokens=num_prompt_tokens)
    fail_scheduler.add_request(request=request)

    req_num_new_matched_tokens = {
        request.request_id: num_external_computed_tokens,
    }

    # mock connector indicating sync load
    fail_scheduler.connector = Mock()
    fail_scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(req_num_new_matched_tokens, False)
    )
    fail_scheduler.connector.request_finished.return_value = (False, None)
    fail_scheduler.connector.take_events.return_value = ()

    scheduler_output = fail_scheduler.schedule()

    # request should be running with sync KV load
    assert len(fail_scheduler.running) == 1
    assert request.status == RequestStatus.RUNNING

    # get allocated block IDs
    req_block_ids = scheduler_output.scheduled_new_reqs[0].block_ids[0]
    invalid_block_id = req_block_ids[invalid_block_idx]
    invalid_block_ids = {invalid_block_id}

    # verify the block is in the block pool before we report it as invalid
    block = fail_scheduler.kv_cache_manager.block_pool.blocks[invalid_block_id]
    assert block is not None

    # report invalid blocks - request should fail
    model_runner_output = create_model_runner_output(
        [request],
        invalid_block_ids=invalid_block_ids,
        use_eos=True,
    )

    outputs = fail_scheduler.update_from_output(scheduler_output, model_runner_output)

    # verify request is finished with error
    assert request.status == RequestStatus.FINISHED_ERROR
    assert request.get_finished_reason() == FinishReason.ERROR

    # verify output is generated
    assert len(outputs) == 1
    engine_outputs = next(iter(outputs.values()))
    assert len(engine_outputs.outputs) == 1
    output = engine_outputs.outputs[0]
    assert output.request_id == request.request_id
    assert output.finish_reason == FinishReason.ERROR

    # verify the request was removed from scheduler
    assert request.request_id not in fail_scheduler.requests
    assert len(fail_scheduler.running) == 0

    # critical: verify invalid block was actually freed from cache
    # this is the key assertion - the invalid block should no longer be
    # tracked by the KV cache manager for this request
    # if it's still there, a future request could reuse the invalid data
    try:
        block_ids = fail_scheduler.kv_cache_manager.get_block_ids(request.request_id)
        # if we get here, check if blocks were actually freed
        if block_ids is not None and len(block_ids[0]) > 0:
            pytest.fail(
                f"Invalid blocks still tracked for finished request! "
                f"Request {request.request_id} should have been freed but "
                f"still has {len(block_ids[0])} blocks allocated."
            )
        # blocks list exists but is empty - this is fine, they were freed
    except KeyError:
        # expected - request completely removed from tracking
        pass

    # critical: verify invalid block was evicted from prefix cache
    # the block should no longer have a hash (hash is reset on eviction)
    assert block.block_hash is None, (
        f"Invalid block {invalid_block_id} should have been evicted from cache "
        f"(hash should be None), but hash is still {block.block_hash}"
    )

    # Verify connector prefix cache stats - request failed, no stats recorded
    assert engine_outputs.scheduler_stats is not None
    stats = engine_outputs.scheduler_stats
    assert stats.connector_prefix_cache_stats is not None
    assert stats.connector_prefix_cache_stats.requests == 0, (
        f"Failed requests should not contribute to connector stats, got"
        f"{stats.connector_prefix_cache_stats} requests recorded"
    )


def test_async_recompute_blocks_not_cached_when_invalid(
    recompute_scheduler: Scheduler,
):
    """
    Test async recompute case - invalid blocks not cached after transfer.

    When async KV loading has invalid blocks and retry_policy is 'recompute':
    1. Blocks are allocated but not cached yet
    2. When async transfer completes, only valid blocks should be cached
    3. Invalid blocks should never enter the prefix cache

    This test verifies correctness, the failed_recving_kv_req_ids protection
    ensures only valid blocks are cached when the transfer completes, and we
    only evict blocks from cache that are already hashed in the block table.
    """
    from unittest.mock import patch

    num_prompt_blocks = 100
    num_external_computed_blocks = 99
    invalid_block_idx = 50

    num_prompt_tokens = num_prompt_blocks * recompute_scheduler.block_size
    num_external_computed_tokens = (
        num_external_computed_blocks * recompute_scheduler.block_size
    )

    request = create_request(num_tokens=num_prompt_tokens)
    recompute_scheduler.add_request(request=request)

    req_num_new_matched_tokens = {
        request.request_id: num_external_computed_tokens,
    }

    # mock connector indicating async load
    recompute_scheduler.connector = Mock()
    recompute_scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(req_num_new_matched_tokens, True)
    )
    recompute_scheduler.connector.request_finished.return_value = (False, None)
    recompute_scheduler.connector.take_events.return_value = ()

    scheduler_output = recompute_scheduler.schedule()

    # request should be waiting for remote KVs
    assert len(recompute_scheduler.waiting) == 1
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert request.num_computed_tokens == 0

    # get the allocated block IDs
    (req_block_ids,) = recompute_scheduler.kv_cache_manager.get_block_ids(
        request.request_id
    )
    invalid_block_id = req_block_ids[invalid_block_idx]
    invalid_block_ids = {invalid_block_id}

    # get the block object to verify it's not cached yet and stays uncached
    block = recompute_scheduler.kv_cache_manager.block_pool.blocks[invalid_block_id]

    # verify block has no hash before invalid blocks are reported
    assert block.block_hash is None, (
        "Async loading blocks should not be cached yet (no hash)"
    )

    # report invalid blocks (transfer not finished yet)
    model_runner_output = create_model_runner_output(
        reqs=[],
        finished_recving=None,  # transfer NOT finished
        invalid_block_ids=invalid_block_ids,
        use_eos=False,
    )

    # critical: spy on evict_blocks to verify it's NOT called for async blocks
    original_evict_blocks = recompute_scheduler.kv_cache_manager.evict_blocks
    evict_blocks_calls = []

    def evict_blocks_spy(block_ids):
        evict_blocks_calls.append(set(block_ids))
        return original_evict_blocks(block_ids)

    with patch.object(
        recompute_scheduler.kv_cache_manager, "evict_blocks", evict_blocks_spy
    ):
        recompute_scheduler.update_from_output(scheduler_output, model_runner_output)

    # verify evict_blocks was NOT called (async blocks excluded from eviction)
    assert len(evict_blocks_calls) == 0, (
        f"evict_blocks should not be called for async-only invalid blocks, "
        f"but was called {len(evict_blocks_calls)} time(s) with {evict_blocks_calls}"
    )

    # request should still be waiting (not finished with error due to recompute policy)
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert request.request_id in recompute_scheduler.failed_recving_kv_req_ids

    # verify num_computed_tokens was truncated to before invalid block
    expected_valid_tokens = invalid_block_idx * recompute_scheduler.block_size
    assert request.num_computed_tokens == expected_valid_tokens

    # verify invalid block still has no hash (was not evicted)
    assert block.block_hash is None, (
        f"Async loading blocks shouldn't be cached or evicted. "
        f"Block {invalid_block_id} hash should be None but is {block.block_hash}"
    )

    # now simulate async transfer completing
    model_runner_output_2 = create_model_runner_output(
        reqs=[],
        finished_recving={request.request_id},
        invalid_block_ids=None,
        use_eos=False,
    )

    recompute_scheduler.update_from_output(scheduler_output, model_runner_output_2)

    # verify request is now marked as finished receiving and ready to be processed
    assert request.request_id in recompute_scheduler.finished_recving_kv_req_ids
    assert request.request_id in recompute_scheduler.failed_recving_kv_req_ids

    # critical: verify invalid block still has no hash before recompute
    # the async transfer invalid data was never cached
    assert block.block_hash is None, (
        f"Invalid block {invalid_block_id} should not be cached before recompute "
        f"(hash should be None), but hash is {block.block_hash}"
    )

    # critical end-to-end test: spy on cache_blocks to verify it's called with
    # the truncated num_computed_tokens value
    original_cache_blocks = recompute_scheduler.kv_cache_manager.cache_blocks
    cache_blocks_calls = []

    def cache_blocks_spy(req, num_tokens):
        cache_blocks_calls.append((req.request_id, num_tokens))
        return original_cache_blocks(req, num_tokens)

    with patch.object(
        recompute_scheduler.kv_cache_manager, "cache_blocks", cache_blocks_spy
    ):
        # call schedule() again - this triggers _update_waiting_for_remote_kv()
        # which should call cache_blocks with the truncated value
        scheduler_output_3 = recompute_scheduler.schedule()

    # verify cache_blocks was called with the truncated value
    assert len(cache_blocks_calls) == 1, (
        f"cache_blocks should be called exactly once, "
        f"got {len(cache_blocks_calls)} calls"
    )
    cached_req_id, cached_num_tokens = cache_blocks_calls[0]
    assert cached_req_id == request.request_id
    assert cached_num_tokens == expected_valid_tokens, (
        f"cache_blocks should be called with truncated value {expected_valid_tokens}, "
        f"but was called with {cached_num_tokens}"
    )

    # request should now be RUNNING (scheduled immediately after transfer completes)
    # the flow is: WAITING_FOR_REMOTE_KVS -> WAITING -> RUNNING in same schedule() call
    assert request.status == RequestStatus.RUNNING

    # num_computed_tokens should be >= expected_valid_tokens because the scheduler
    # will schedule additional new tokens (up to max_num_batched_tokens) for the request
    assert request.num_computed_tokens >= expected_valid_tokens, (
        f"num_computed_tokens should be at least {expected_valid_tokens}, "
        f"got {request.num_computed_tokens}"
    )

    # request should no longer be in the failed/finished receiving sets
    assert request.request_id not in recompute_scheduler.failed_recving_kv_req_ids
    assert request.request_id not in recompute_scheduler.finished_recving_kv_req_ids

    # request should be in the running queue
    assert request in recompute_scheduler.running

    # Execute the request to trigger stats recording
    model_runner_output_3 = create_model_runner_output(
        [request],
        invalid_block_ids=None,  # no more invalid blocks
        use_eos=False,
    )

    outputs = recompute_scheduler.update_from_output(
        scheduler_output_3, model_runner_output_3
    )

    # Verify connector prefix cache stats:
    # - queries = num_prompt_tokens (total tokens not in local cache)
    # - hits = only valid tokens (truncated at invalid block boundary)
    assert len(outputs) == 1
    engine_outputs = next(iter(outputs.values()))
    assert engine_outputs.scheduler_stats is not None
    stats = engine_outputs.scheduler_stats
    assert stats.connector_prefix_cache_stats is not None
    conn_stats = stats.connector_prefix_cache_stats
    assert conn_stats.requests == 1
    assert conn_stats.queries == num_prompt_tokens
    assert conn_stats.hits == expected_valid_tokens
