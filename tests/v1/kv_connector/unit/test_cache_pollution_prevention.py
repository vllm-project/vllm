# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
test that invalid blocks are evicted from prefix cache to prevent pollution.

verifies that when sync-loading fails, invalid blocks are removed from the
prefix cache hash table so future requests cannot match and reuse corrupted data.
"""

from collections.abc import Callable
from unittest.mock import Mock

import pytest

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

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


def test_invalid_blocks_evicted_prevents_cache_pollution(
    fail_scheduler: Scheduler,
):
    """
    verify invalid blocks are evicted to prevent future cache hits.

    scenario:
    1. request 1 loads externally-computed blocks (sync mode)
    2. some blocks fail to load and are marked invalid
    3. with fail policy, invalid blocks should be evicted from prefix cache
    4. request is marked as FINISHED_ERROR
    """
    num_prompt_blocks = 100
    num_external_computed_blocks = 99
    invalid_block_idx = 50

    num_prompt_tokens = num_prompt_blocks * fail_scheduler.block_size
    num_external_computed_tokens = (
        num_external_computed_blocks * fail_scheduler.block_size
    )

    # request 1: will have invalid blocks
    request1 = create_request(num_tokens=num_prompt_tokens, request_id=1)
    fail_scheduler.add_request(request=request1)

    req_num_new_matched_tokens = {
        request1.request_id: num_external_computed_tokens,
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
    assert request1.status == RequestStatus.RUNNING

    # get allocated block IDs
    req_block_ids = scheduler_output.scheduled_new_reqs[0].block_ids[0]
    invalid_block_id = req_block_ids[invalid_block_idx]
    invalid_block_ids = {invalid_block_id}

    # get the block object to verify eviction later
    block = fail_scheduler.kv_cache_manager.block_pool.blocks[invalid_block_id]

    # cache the blocks to simulate they've been computed and cached
    # (in real scenario blocks would be cached after compute)
    fail_scheduler.kv_cache_manager.cache_blocks(request1, num_external_computed_tokens)

    # verify block has a hash (is cached) before reporting invalid blocks
    assert block.block_hash is not None, (
        f"block {invalid_block_id} should be cached (have a hash) before "
        f"eviction test, but hash is None"
    )

    # report invalid blocks
    model_runner_output = create_model_runner_output(
        [request1],
        invalid_block_ids=invalid_block_ids,
        use_eos=False,
    )

    fail_scheduler.update_from_output(scheduler_output, model_runner_output)

    # verify request finished with error (fail policy)
    assert request1.status == RequestStatus.FINISHED_ERROR

    # critical assertion: invalid block and all subsequent blocks should be evicted
    # all blocks from invalid_block_idx onwards become invalid since they were
    # computed based on the failed block
    for idx in range(invalid_block_idx, len(req_block_ids)):
        block_id = req_block_ids[idx]
        block_obj = fail_scheduler.kv_cache_manager.block_pool.blocks[block_id]
        assert block_obj.block_hash is None, (
            f"block {block_id} at index {idx} should have been evicted "
            f"(hash reset to None), but hash is {block_obj.block_hash}. "
            f"All blocks from index {invalid_block_idx} onwards should be evicted "
            f"since they depend on the invalid block at index {invalid_block_idx}."
        )

    # verify cache contains exactly the valid blocks (before first affected block)
    # and none of the invalid blocks (from first affected block onwards)

    # valid blocks: all blocks before invalid_block_idx should be cached
    for idx in range(invalid_block_idx):
        block_id = req_block_ids[idx]
        block_obj = fail_scheduler.kv_cache_manager.block_pool.blocks[block_id]
        assert block_obj.block_hash is not None, (
            f"valid block {block_id} at index {idx} should still be cached "
            f"(have a hash), but hash is None. Only blocks from index "
            f"{invalid_block_idx} onwards should be evicted."
        )

    # invalid blocks: verify they're not in the cached_block_hash_to_block map
    cached_blocks = (
        fail_scheduler.kv_cache_manager.block_pool.cached_block_hash_to_block
    )
    cached_block_ids = {
        b.block_id
        for blocks_val in cached_blocks._cache.values()
        for b in (
            [blocks_val] if not isinstance(blocks_val, dict) else blocks_val.values()
        )
    }

    for idx in range(invalid_block_idx, len(req_block_ids)):
        block_id = req_block_ids[idx]
        assert block_id not in cached_block_ids, (
            f"invalid block {block_id} at index {idx} should not be in cache hash table"
        )
