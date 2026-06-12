# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for request-level KV load failure handling."""

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
    vllm_config = create_vllm_config(kv_load_failure_policy="fail")
    return create_scheduler(vllm_config)


@pytest.fixture
def recompute_scheduler():
    vllm_config = create_vllm_config(kv_load_failure_policy="recompute")
    return create_scheduler(vllm_config)


def test_sync_load_fail_policy(fail_scheduler: Scheduler):
    num_prompt_blocks = 100
    num_external_computed_blocks = 99

    num_prompt_tokens = num_prompt_blocks * fail_scheduler.block_size
    num_external_computed_tokens = (
        num_external_computed_blocks * fail_scheduler.block_size
    )

    request = create_request(num_tokens=num_prompt_tokens)
    fail_scheduler.add_request(request=request)

    req_num_new_matched_tokens = {
        request.request_id: num_external_computed_tokens,
    }
    fail_scheduler.connector = Mock()
    fail_scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(req_num_new_matched_tokens, False)
    )
    fail_scheduler.connector.request_finished.return_value = (False, None)
    fail_scheduler.connector.take_events.return_value = ()

    sched_out = fail_scheduler.schedule()
    assert len(fail_scheduler.running) == 1
    assert request.status == RequestStatus.RUNNING

    # Cache blocks so they get hashed into prefix cache.
    fail_scheduler.kv_cache_manager.cache_blocks(request, num_external_computed_tokens)
    block_ids = sched_out.scheduled_new_reqs[0].block_ids[0]
    assert (
        fail_scheduler.kv_cache_manager.block_pool.blocks[block_ids[0]].block_hash
        is not None
    )

    mro = create_model_runner_output(
        [request],
        failed_recv_request_ids={request.request_id},
        use_eos=True,
    )
    outputs = fail_scheduler.update_from_output(sched_out, mro)

    assert request.status == RequestStatus.FINISHED_ERROR
    assert request.get_finished_reason() == FinishReason.ERROR
    assert len(fail_scheduler.running) == 0
    assert request.request_id not in fail_scheduler.requests

    engine_out = next(iter(outputs.values()))
    assert any(
        o.request_id == request.request_id and o.finish_reason == FinishReason.ERROR
        for o in engine_out.outputs
    )

    # All blocks evicted from prefix cache.
    for bid in block_ids:
        assert fail_scheduler.kv_cache_manager.block_pool.blocks[bid].block_hash is None


def test_sync_load_recompute_policy(recompute_scheduler: Scheduler):
    num_prompt_blocks = 100
    num_external_computed_blocks = 99

    num_prompt_tokens = num_prompt_blocks * recompute_scheduler.block_size
    num_external_computed_tokens = (
        num_external_computed_blocks * recompute_scheduler.block_size
    )

    request = create_request(num_tokens=num_prompt_tokens)
    recompute_scheduler.add_request(request=request)

    req_num_new_matched_tokens = {
        request.request_id: num_external_computed_tokens,
    }
    recompute_scheduler.connector = Mock()
    recompute_scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(req_num_new_matched_tokens, False)
    )
    recompute_scheduler.connector.request_finished.return_value = (False, None)
    recompute_scheduler.connector.take_events.return_value = ()

    sched_out = recompute_scheduler.schedule()
    assert request.status == RequestStatus.RUNNING
    assert request.num_computed_tokens > 0

    # Cache blocks so they get hashed into prefix cache.
    recompute_scheduler.kv_cache_manager.cache_blocks(
        request, num_external_computed_tokens
    )
    block_ids = sched_out.scheduled_new_reqs[0].block_ids[0]
    assert (
        recompute_scheduler.kv_cache_manager.block_pool.blocks[block_ids[0]].block_hash
        is not None
    )

    mro = create_model_runner_output(
        [request],
        failed_recv_request_ids={request.request_id},
        use_eos=False,
    )
    outputs = recompute_scheduler.update_from_output(sched_out, mro)

    assert request.num_computed_tokens == 0
    assert request.status == RequestStatus.RUNNING
    assert request in recompute_scheduler.running
    assert request.request_id in recompute_scheduler.requests

    for outs in outputs.values():
        assert request.request_id not in [o.request_id for o in outs.outputs]

    # Blocks still allocated for recomputation.
    alloc_blocks = recompute_scheduler.kv_cache_manager.get_block_ids(
        request.request_id
    )
    assert len(alloc_blocks[0]) > 0

    # Blocks are NOT evicted under recompute policy.
    for bid in block_ids:
        assert (
            recompute_scheduler.kv_cache_manager.block_pool.blocks[bid].block_hash
            is not None
        )


def test_async_load_fail_policy(fail_scheduler: Scheduler):
    num_prompt_blocks = 100
    num_external_computed_blocks = 99

    num_prompt_tokens = num_prompt_blocks * fail_scheduler.block_size
    num_external_computed_tokens = (
        num_external_computed_blocks * fail_scheduler.block_size
    )

    request = create_request(num_tokens=num_prompt_tokens)
    fail_scheduler.add_request(request=request)

    req_num_new_matched_tokens = {
        request.request_id: num_external_computed_tokens,
    }
    fail_scheduler.connector = Mock()
    fail_scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(req_num_new_matched_tokens, True)
    )
    fail_scheduler.connector.request_finished.return_value = (False, None)
    fail_scheduler.connector.take_events.return_value = ()

    sched_out = fail_scheduler.schedule()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS

    # Async uses delay_cache_blocks=True, so blocks are never hashed.
    block_ids = fail_scheduler.kv_cache_manager.get_block_ids(request.request_id)
    assert len(block_ids[0]) > 0
    for bid in block_ids[0]:
        assert fail_scheduler.kv_cache_manager.block_pool.blocks[bid].block_hash is None

    mro = create_model_runner_output(
        reqs=[],
        failed_recv_request_ids={request.request_id},
    )
    outputs = fail_scheduler.update_from_output(sched_out, mro)

    assert request.status == RequestStatus.FINISHED_ERROR
    assert request.get_finished_reason() == FinishReason.ERROR
    assert len(fail_scheduler.waiting) == 0

    engine_out = next(iter(outputs.values()))
    assert any(
        o.request_id == request.request_id and o.finish_reason == FinishReason.ERROR
        for o in engine_out.outputs
    )


def test_async_load_recompute_policy(recompute_scheduler: Scheduler):
    num_prompt_blocks = 100
    num_external_computed_blocks = 99

    num_prompt_tokens = num_prompt_blocks * recompute_scheduler.block_size
    num_external_computed_tokens = (
        num_external_computed_blocks * recompute_scheduler.block_size
    )

    request = create_request(num_tokens=num_prompt_tokens)
    recompute_scheduler.add_request(request=request)

    req_num_new_matched_tokens = {
        request.request_id: num_external_computed_tokens,
    }
    recompute_scheduler.connector = Mock()
    recompute_scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(req_num_new_matched_tokens, True)
    )
    recompute_scheduler.connector.take_events.return_value = ()

    sched_out = recompute_scheduler.schedule()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert request.num_computed_tokens == num_external_computed_tokens

    # Blocks allocated but not cached (async uses delay_cache_blocks=True).
    (orig_block_ids,) = recompute_scheduler.kv_cache_manager.get_block_ids(
        request.request_id
    )
    assert len(orig_block_ids) > 0

    # Report failure (transfer not yet finished).
    mro = create_model_runner_output(
        reqs=[],
        failed_recv_request_ids={request.request_id},
    )
    recompute_scheduler.update_from_output(sched_out, mro)

    assert request.num_computed_tokens == 0
    assert request.request_id in recompute_scheduler.failed_recving_kv_req_ids

    # Transfer finishes.
    mro2 = create_model_runner_output(
        reqs=[],
        finished_recving={request.request_id},
    )
    recompute_scheduler.update_from_output(sched_out, mro2)

    assert request.request_id in recompute_scheduler.finished_recving_kv_req_ids
    assert request.request_id in recompute_scheduler.failed_recving_kv_req_ids

    # _update_waiting_for_remote_kv frees blocks (num_computed_tokens==0),
    # re-adds request to waiting. MockKVConnector re-matches as async,
    # so it goes back to WAITING_FOR_REMOTE_KVS (valid retry cycle).
    recompute_scheduler.schedule()

    assert request.request_id not in recompute_scheduler.failed_recving_kv_req_ids
    assert request.request_id not in recompute_scheduler.finished_recving_kv_req_ids
    assert request.status in (
        RequestStatus.RUNNING,
        RequestStatus.WAITING_FOR_REMOTE_KVS,
    )

    # Original blocks were freed by _update_waiting_for_remote_kv.
    for bid in orig_block_ids:
        assert recompute_scheduler.kv_cache_manager.block_pool.blocks[bid].ref_cnt == 0


def test_multiple_requests_partial_failure(recompute_scheduler: Scheduler):
    num_prompt_blocks = 100
    num_external_computed_blocks = 99

    num_prompt_tokens = num_prompt_blocks * recompute_scheduler.block_size
    num_external_computed_tokens = (
        num_external_computed_blocks * recompute_scheduler.block_size
    )

    requests = [create_request(num_tokens=num_prompt_tokens) for _ in range(3)]
    for r in requests:
        recompute_scheduler.add_request(request=r)

    req_num_new_matched_tokens = {
        r.request_id: num_external_computed_tokens for r in requests
    }
    recompute_scheduler.connector = Mock()
    recompute_scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(req_num_new_matched_tokens, False)
    )
    recompute_scheduler.connector.request_finished.return_value = (False, None)
    recompute_scheduler.connector.take_events.return_value = ()

    sched_out = recompute_scheduler.schedule()
    assert len(recompute_scheduler.running) == 3

    # Only the second request fails.
    failed_req = requests[1]
    mro = create_model_runner_output(
        requests,
        failed_recv_request_ids={failed_req.request_id},
        use_eos=True,
    )
    outputs = recompute_scheduler.update_from_output(sched_out, mro)

    assert failed_req.num_computed_tokens == 0
    assert failed_req.status == RequestStatus.RUNNING

    output_req_ids = {o.request_id for outs in outputs.values() for o in outs.outputs}
    assert requests[0].request_id in output_req_ids
    assert requests[2].request_id in output_req_ids
    assert failed_req.request_id not in output_req_ids
