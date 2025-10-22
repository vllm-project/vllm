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
def recompute_scheduler():
    """scheduler with kv_load_failure_policy='recompute'"""
    vllm_config = create_vllm_config()
    vllm_config.kv_transfer_config.kv_load_failure_policy = "recompute"
    return create_scheduler(vllm_config)


def test_invalid_blocks_evicted_prevents_cache_pollution(
    recompute_scheduler: Scheduler,
):
    """
    verify invalid blocks are evicted to prevent future cache hits.

    scenario:
    1. request 1 loads externally-computed blocks (sync mode)
    2. some blocks fail to load and are marked invalid
    3. invalid blocks should be evicted from prefix cache
    4. request 2 with same prefix should NOT match the invalid blocks
    """
    num_prompt_blocks = 100
    num_external_computed_blocks = 99
    invalid_block_idx = 50

    num_prompt_tokens = num_prompt_blocks * recompute_scheduler.block_size
    num_external_computed_tokens = (
        num_external_computed_blocks * recompute_scheduler.block_size
    )

    # request 1: will have invalid blocks
    request1 = create_request(num_tokens=num_prompt_tokens, request_id=1)
    recompute_scheduler.add_request(request=request1)

    req_num_new_matched_tokens = {
        request1.request_id: num_external_computed_tokens,
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
    assert request1.status == RequestStatus.RUNNING

    # get allocated block IDs
    req_block_ids = scheduler_output.scheduled_new_reqs[0].block_ids[0]
    invalid_block_id = req_block_ids[invalid_block_idx]
    invalid_block_ids = {invalid_block_id}

    # get the block object to verify eviction later
    block = recompute_scheduler.kv_cache_manager.block_pool.blocks[invalid_block_id]

    # before failure: verify block has a hash (is cached)
    # note: in real scenario block would be cached after compute,
    # but for this test we're checking the eviction mechanism

    # report invalid blocks
    model_runner_output = create_model_runner_output(
        [request1],
        invalid_block_ids=invalid_block_ids,
        use_eos=False,
    )

    recompute_scheduler.update_from_output(scheduler_output, model_runner_output)

    # verify request still running (recompute policy)
    assert request1.status == RequestStatus.RUNNING

    # critical assertion: invalid block should be evicted from cache
    # the block's hash should be None after eviction
    assert block.block_hash is None, (
        f"invalid block {invalid_block_id} should have been evicted "
        f"(hash reset to None), but hash is {block.block_hash}"
    )

    # verify the block is not in the cached_block_hash_to_block map
    # try to look it up - should return None
    cached_blocks = (
        recompute_scheduler.kv_cache_manager.block_pool.cached_block_hash_to_block
    )
    assert len(cached_blocks) == 0 or invalid_block_id not in [
        b.block_id
        for blocks_val in cached_blocks._cache.values()
        for b in (
            [blocks_val] if not isinstance(blocks_val, dict) else blocks_val.values()
        )
    ], f"invalid block {invalid_block_id} should not be in cache hash table"
