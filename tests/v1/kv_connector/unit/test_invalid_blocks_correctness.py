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

    # Verify connector prefix cache stats:
    # - queries = num_prompt_tokens (total tokens not in local cache)
    # - hits = num_external_computed_tokens (tokens loaded externally)
    assert engine_outputs.scheduler_stats is not None
    stats = engine_outputs.scheduler_stats
    assert stats.connector_prefix_cache_stats is not None
    conn_stats = stats.connector_prefix_cache_stats
    assert conn_stats.requests == 1
    assert conn_stats.queries == num_prompt_tokens
    assert conn_stats.hits == num_external_computed_tokens
