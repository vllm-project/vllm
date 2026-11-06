# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.base import SupportsHMA
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

from .utils import (
    create_mock_connector,
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
    make_kv_cache_config,
)


def _make_get_num_new_matched_tokens(
    req_num_new_matched_tokens: dict[str, int],
    async_load,
) -> Callable[[Request, int], tuple[int, bool]]:
    def get_num_new_matched_tokens(request: Request, _: int) -> tuple[int, bool]:
        value = req_num_new_matched_tokens.get(request.request_id, 0)
        return value, async_load

    return get_num_new_matched_tokens


@pytest.fixture
def scheduler():
    vllm_config = create_vllm_config(kv_load_failure_policy="recompute")
    return create_scheduler(vllm_config)


@pytest.mark.parametrize("policy", ["fail", "recompute"])
def test_failed_receive_completion_honors_failure_policy(policy):
    """Failed receives still need completion to free blocks or resume computation."""
    scheduler = create_scheduler(create_vllm_config(kv_load_failure_policy=policy))
    request = create_request(num_tokens=3 * scheduler.block_size)
    scheduler.add_request(request)
    scheduler.connector = create_mock_connector()
    scheduler.connector.get_loaded_kv_cache_group_ids.return_value = (0,)
    scheduler.connector.get_num_new_matched_tokens.side_effect = [
        (2 * scheduler.block_size, True),
        (0, False),
    ]
    scheduler.connector.request_finished.return_value = (False, None)
    scheduler.connector.take_events.return_value = ()
    block_pool = scheduler.kv_cache_manager.block_pool
    free_blocks_before = block_pool.get_num_free_blocks()
    scheduler_output = scheduler.schedule()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS

    output = create_model_runner_output([], finished_recving={request.request_id})
    assert output.kv_connector_output is not None
    output.kv_connector_output.failed_recving = {request.request_id}
    scheduler.update_from_output(scheduler_output, output)

    if policy == "fail":
        assert request.status == RequestStatus.FINISHED_ERROR
        assert request.request_id not in scheduler.requests
        assert block_pool.get_num_free_blocks() == free_blocks_before
    else:
        assert request.num_computed_tokens == 0
        next_output = scheduler.schedule()
        assert request.request_id in next_output.num_scheduled_tokens


@pytest.mark.parametrize(
    "num_prompt_blocks,num_external_computed_blocks,invalid_block_idxs",
    [
        (100, 99, {0, 98}),
        (100, 99, {50, 98}),
        (100, 99, {98}),
    ],
)
def test_async_load_failure(
    scheduler: Scheduler,
    num_prompt_blocks: int,
    num_external_computed_blocks: int,
    invalid_block_idxs: set[int],
):
    assert num_prompt_blocks >= num_external_computed_blocks

    num_prompt_tokens = num_prompt_blocks * scheduler.block_size
    num_external_computed_tokens = num_external_computed_blocks * scheduler.block_size

    request1 = create_request(num_tokens=num_prompt_tokens)
    scheduler.add_request(request=request1)
    request2 = create_request(num_tokens=num_prompt_tokens)
    scheduler.add_request(request=request2)
    request3 = create_request(num_tokens=num_prompt_tokens)
    scheduler.add_request(request=request3)

    # Mock KV connector method.
    # req_id -> num_external_computed_tokens
    req_num_new_matched_tokens = {
        request1.request_id: num_external_computed_tokens,
        request2.request_id: num_external_computed_tokens,
        request3.request_id: num_external_computed_tokens,
    }

    scheduler.connector = create_mock_connector()
    scheduler.connector.get_loaded_kv_cache_group_ids.return_value = (0,)
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(req_num_new_matched_tokens, async_load=True)
    )
    scheduler.connector.take_events.return_value = ()

    scheduler_output = scheduler.schedule()

    assert len(scheduler.waiting) == 0
    assert len(scheduler.kv_holding_waiting) == 3
    for request in scheduler.kv_holding_waiting:
        assert request.num_computed_tokens == num_external_computed_tokens
        assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert scheduler.connector.get_num_new_matched_tokens.call_count == 3

    # Simulate a failure in loading some of request2 blocks.
    (req2_block_ids,) = scheduler.kv_cache_manager.get_block_ids(request2.request_id)
    invalid_block_ids = {req2_block_ids[i] for i in invalid_block_idxs}
    model_runner_output = create_model_runner_output(
        reqs=[],
        finished_recving={request1.request_id, request3.request_id},
        invalid_block_ids=invalid_block_ids,
        use_eos=True,
    )

    scheduler.update_from_output(scheduler_output, model_runner_output)

    min_invalid_block_idx = min(invalid_block_idxs)

    assert len(scheduler.waiting) == 0
    assert len(scheduler.kv_holding_waiting) == 3
    for request in scheduler.kv_holding_waiting:
        if request.request_id == request2.request_id:
            assert request.num_computed_tokens == (
                min_invalid_block_idx * scheduler.block_size
            )
        else:
            assert request.num_computed_tokens == num_external_computed_tokens
        assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert scheduler.failed_recving_kv_req_ids == {request2.request_id}
    assert scheduler.connector.get_num_new_matched_tokens.call_count == 3


@pytest.mark.parametrize(
    "num_prompt_blocks,num_external_computed_blocks,invalid_block_idxs",
    [
        (100, 99, {0, 98}),
        (100, 99, {50, 98}),
        (100, 99, {98}),
    ],
)
def test_sync_load_failure(
    scheduler: Scheduler,
    num_prompt_blocks: int,
    num_external_computed_blocks: int,
    invalid_block_idxs: set[int],
):
    assert num_prompt_blocks >= num_external_computed_blocks

    num_prompt_tokens = num_prompt_blocks * scheduler.block_size
    num_external_computed_tokens = num_external_computed_blocks * scheduler.block_size

    request1 = create_request(num_tokens=num_prompt_tokens)
    scheduler.add_request(request=request1)
    request2 = create_request(num_tokens=num_prompt_tokens)
    scheduler.add_request(request=request2)
    request3 = create_request(num_tokens=num_prompt_tokens)
    scheduler.add_request(request=request3)

    # Mock KV connector method.
    # req_id -> num_external_computed_tokens
    req_num_new_matched_tokens = {
        request1.request_id: num_external_computed_tokens,
        request2.request_id: num_external_computed_tokens,
        request3.request_id: num_external_computed_tokens,
    }

    scheduler.connector = create_mock_connector()
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(req_num_new_matched_tokens, async_load=False)
    )
    scheduler.connector.request_finished.return_value = (False, None)
    scheduler.connector.take_events.return_value = ()

    scheduler_output = scheduler.schedule()

    # req_id -> num_computed_tokens
    expected_computed_tokens = {
        request1.request_id: num_external_computed_tokens,
        request2.request_id: num_external_computed_tokens,
        request3.request_id: num_external_computed_tokens,
    }

    assert len(scheduler.running) == 3
    assert len(scheduler_output.scheduled_new_reqs) == 3
    for request in scheduler_output.scheduled_new_reqs:
        assert request.num_computed_tokens == expected_computed_tokens[request.req_id]
    assert scheduler.connector.get_num_new_matched_tokens.call_count == 3

    # Simulate a failure in loading some of request2 blocks.
    req2_block_ids = scheduler_output.scheduled_new_reqs[1].block_ids[0]
    invalid_block_ids = {req2_block_ids[i] for i in invalid_block_idxs}
    model_runner_output = create_model_runner_output(
        [request1, request2, request3],
        invalid_block_ids=invalid_block_ids,
        use_eos=True,
    )

    scheduler.update_from_output(scheduler_output, model_runner_output)

    assert len(scheduler.running) == 1
    assert scheduler.running[0].request_id == request2.request_id
    assert scheduler.running[0].num_computed_tokens == (
        min(invalid_block_idxs) * scheduler.block_size
    )
    assert scheduler.connector.get_num_new_matched_tokens.call_count == 3
    assert scheduler.connector.request_finished.call_count == 2


@pytest.mark.parametrize(
    "num_prompt_blocks,"
    "num_external_computed_blocks,"
    "num_common_prefix_blocks,"
    "invalid_block_idxs",
    [
        (100, 99, 50, {0, 49}),
        (100, 99, 50, {25, 49}),
        (100, 99, 50, {49}),
    ],
)
def test_sync_load_failure_with_shared_blocks(
    scheduler: Scheduler,
    num_prompt_blocks: int,
    num_external_computed_blocks: int,
    num_common_prefix_blocks: int,
    invalid_block_idxs: set[int],
):
    assert num_prompt_blocks >= num_external_computed_blocks >= num_common_prefix_blocks

    num_prompt_tokens = num_prompt_blocks * scheduler.block_size
    num_external_computed_tokens = num_external_computed_blocks * scheduler.block_size
    common_prefix_len = num_common_prefix_blocks * scheduler.block_size

    request1 = create_request(
        num_tokens=num_prompt_tokens, common_prefix_len=common_prefix_len
    )
    scheduler.add_request(request=request1)
    request2 = create_request(
        num_tokens=num_prompt_tokens, common_prefix_len=common_prefix_len
    )
    scheduler.add_request(request=request2)

    # Mock KV connector method.
    # req_id -> num_external_computed_tokens
    req_num_new_matched_tokens = {
        request1.request_id: num_external_computed_tokens,
    }

    scheduler.connector = create_mock_connector()
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(req_num_new_matched_tokens, async_load=False)
    )
    scheduler.connector.take_events.return_value = ()

    scheduler_output = scheduler.schedule()

    # req_id -> num_computed_tokens
    expected_computed_tokens = {
        request1.request_id: num_external_computed_tokens,
        request2.request_id: common_prefix_len,
    }

    assert len(scheduler.running) == 2
    assert len(scheduler_output.scheduled_new_reqs) == 2
    for new_req in scheduler_output.scheduled_new_reqs:
        assert new_req.num_computed_tokens == expected_computed_tokens[new_req.req_id]
    assert scheduler.connector.get_num_new_matched_tokens.call_count == 2

    # Simulate a failure in loading some of the shared blocks.
    req1_block_ids = scheduler_output.scheduled_new_reqs[0].block_ids[0]
    invalid_block_ids = {req1_block_ids[i] for i in invalid_block_idxs}
    model_runner_output = create_model_runner_output(
        [request1, request2], invalid_block_ids=invalid_block_ids, use_eos=True
    )

    scheduler.update_from_output(scheduler_output, model_runner_output)

    # req_id -> num_computed_tokens
    # all the common prefix blocks will be computed by request1
    expected_computed_tokens = {
        request1.request_id: min(invalid_block_idxs) * scheduler.block_size,
        request2.request_id: common_prefix_len,
    }

    assert len(scheduler.running) == 2
    for request in scheduler.running:
        assert (
            request.num_computed_tokens == expected_computed_tokens[request.request_id]
        )
    assert scheduler.connector.get_num_new_matched_tokens.call_count == 2


@pytest.mark.parametrize(
    "num_prompt_blocks,num_external_computed_blocks,invalid_block_idxs",
    [
        (100, 99, {0, 50, 98}),
        (100, 99, {98, 50, 0}),
    ],
)
def test_async_progressive_load_failure(
    scheduler: Scheduler,
    num_prompt_blocks: int,
    num_external_computed_blocks: int,
    invalid_block_idxs: set[int],
):
    assert num_prompt_blocks >= num_external_computed_blocks

    num_prompt_tokens = num_prompt_blocks * scheduler.block_size
    num_external_computed_tokens = num_external_computed_blocks * scheduler.block_size

    request = create_request(num_tokens=num_prompt_tokens)
    scheduler.add_request(request=request)

    # Mock KV connector method.
    # req_id -> num_external_computed_tokens
    req_num_new_matched_tokens = {
        request.request_id: num_external_computed_tokens,
    }

    scheduler.connector = create_mock_connector()
    scheduler.connector.get_loaded_kv_cache_group_ids.return_value = (0,)
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(req_num_new_matched_tokens, async_load=True)
    )
    scheduler.connector.take_events.return_value = ()

    scheduler_output = scheduler.schedule()

    assert len(scheduler.waiting) == 0
    assert len(scheduler.kv_holding_waiting) == 1
    peeked = scheduler.kv_holding_waiting.peek_request()
    assert peeked.request_id == request.request_id
    assert request.num_computed_tokens == num_external_computed_tokens
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert scheduler.connector.get_num_new_matched_tokens.call_count == 1

    min_invalid_block_idx = max(invalid_block_idxs) + 1
    # Simulate failures when progressively loading request blocks.
    for invalid_block_idx in invalid_block_idxs:
        (req_block_ids,) = scheduler.kv_cache_manager.get_block_ids(request.request_id)
        invalid_block_ids = {req_block_ids[invalid_block_idx]}
        model_runner_output = create_model_runner_output(
            reqs=[],
            finished_recving=set(),
            invalid_block_ids=invalid_block_ids,
            use_eos=True,
        )

        scheduler.update_from_output(scheduler_output, model_runner_output)

        min_invalid_block_idx = min(min_invalid_block_idx, invalid_block_idx)

        assert len(scheduler.waiting) == 0
        assert len(scheduler.kv_holding_waiting) == 1
        peeked = scheduler.kv_holding_waiting.peek_request()
        assert peeked.request_id == request.request_id
        assert request.num_computed_tokens == (
            min_invalid_block_idx * scheduler.block_size
        )
        assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        assert scheduler.failed_recving_kv_req_ids == {request.request_id}
        assert scheduler.connector.get_num_new_matched_tokens.call_count == 1


@pytest.mark.cpu_test
def test_block_level_load_failure_fails_closed_with_multiple_groups():
    """Block-level invalid-block reporting is rejected on multi-group layouts.

    Block IDs are only unique within a group, so the scheduler cannot map
    them back to requests; connectors must report failed_recving instead.
    """
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(
        vllm_config,
        kv_cache_config=make_kv_cache_config(block_size=16, swa_enabled=True),
    )
    assert len(scheduler.kv_cache_config.kv_cache_groups) == 2

    with pytest.raises(RuntimeError, match="failed_recving"):
        scheduler._handle_invalid_blocks({1}, {})


class _HMAMockConnector(Mock, SupportsHMA):
    def request_finished_all_groups(self, request, block_ids):
        return False, None


def _multi_group_sync_load_scheduler(policy: str, *requests: Request) -> Scheduler:
    """Two KV cache groups (full + sliding window); the first request loads
    all but its last block synchronously."""
    scheduler = create_scheduler(
        create_vllm_config(kv_load_failure_policy=policy),
        kv_cache_config=make_kv_cache_config(block_size=16, swa_enabled=True),
    )
    for request in requests:
        scheduler.add_request(request=request)
    scheduler.connector = _HMAMockConnector()
    scheduler.connector.get_external_cache_hit_sources.side_effect = (
        create_mock_connector().get_external_cache_hit_sources.side_effect
    )
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(
            {requests[0].request_id: requests[0].num_tokens - 16}, async_load=False
        )
    )
    scheduler.connector.request_finished.return_value = (False, None)
    scheduler.connector.take_events.return_value = ()
    return scheduler


def _cached(scheduler: Scheduler, block_ids: tuple[list[int], ...]) -> bool:
    managers = scheduler.kv_cache_manager.coordinator.single_type_managers
    return any(
        managers[group].block_pool.blocks[block_id].block_hash is not None
        for group, ids in enumerate(block_ids)
        for block_id in ids
    )


@pytest.mark.parametrize("policy", ["recompute", "fail"])
def test_multi_group_sync_load_failure_is_request_level(policy: str):
    """A failed synchronous load under several KV cache groups is recovered
    per request: the loaded blocks leave the prefix cache, and the request is
    recomputed from scratch or failed; other requests are unaffected."""
    failed, other = create_request(num_tokens=64), create_request(num_tokens=64)
    scheduler = _multi_group_sync_load_scheduler(policy, failed, other)

    scheduler_output = scheduler.schedule()
    loaded = scheduler.kv_cache_manager.get_block_ids(failed.request_id)
    assert _cached(scheduler, loaded)

    scheduler.update_from_output(
        scheduler_output,
        create_model_runner_output(
            [failed, other],
            failed_sync_recving={failed.request_id},
            use_eos=True,
        ),
    )

    assert other.status == RequestStatus.FINISHED_STOPPED
    assert not _cached(scheduler, loaded)
    assert scheduler.running == []
    if policy == "recompute":
        assert failed.status == RequestStatus.PREEMPTED
        assert failed.num_computed_tokens == 0
        assert failed.num_output_tokens == 0
        assert scheduler.waiting.peek_request() is failed
    else:
        assert failed.status == RequestStatus.FINISHED_ERROR


def test_multi_group_sync_load_failure_recovers_sharing_requests():
    """A request that already reuses blocks of the failed load is recomputed
    too, since those blocks hold incomplete KV."""
    failed = create_request(num_tokens=64, common_prefix_len=48)
    sharing = create_request(num_tokens=64, common_prefix_len=48)
    scheduler = _multi_group_sync_load_scheduler("recompute", failed, sharing)

    scheduler_output = scheduler.schedule()
    failed_blocks = scheduler.kv_cache_manager.get_block_ids(failed.request_id)
    sharing_blocks = scheduler.kv_cache_manager.get_block_ids(sharing.request_id)
    assert set(failed_blocks[0]) & set(sharing_blocks[0])

    scheduler.update_from_output(
        scheduler_output,
        create_model_runner_output(
            [failed, sharing], failed_sync_recving={failed.request_id}
        ),
    )

    assert failed.status == RequestStatus.PREEMPTED
    assert sharing.status == RequestStatus.PREEMPTED
    assert scheduler.running == []


def test_failed_sync_recving_ignores_requests_without_a_sync_load():
    failed, other = create_request(num_tokens=64), create_request(num_tokens=64)
    scheduler = _multi_group_sync_load_scheduler("recompute", failed, other)

    scheduler_output = scheduler.schedule()
    scheduler.update_from_output(
        scheduler_output,
        create_model_runner_output(
            [failed, other], failed_sync_recving={other.request_id}
        ),
    )

    assert failed.status == RequestStatus.RUNNING
    assert other.status == RequestStatus.RUNNING


@pytest.mark.parametrize("policy", ["recompute", "fail"])
def test_sync_load_failure_of_a_request_preempted_before_its_output(policy: str):
    """Under async scheduling the next schedule() can preempt a request before
    the output reporting its failed load arrives. Its blocks must still leave
    the prefix cache, and its in-flight output must not be delivered."""
    failed, other = create_request(num_tokens=64), create_request(num_tokens=64)
    scheduler = _multi_group_sync_load_scheduler(policy, failed, other)

    scheduler_output = scheduler.schedule()
    loaded = scheduler.kv_cache_manager.get_block_ids(failed.request_id)
    scheduler.running.remove(failed)
    scheduler._preempt_request(failed, time.monotonic())
    assert _cached(scheduler, loaded)

    scheduler.update_from_output(
        scheduler_output,
        create_model_runner_output(
            [failed, other], failed_sync_recving={failed.request_id}
        ),
    )

    assert not _cached(scheduler, loaded)
    assert other.status == RequestStatus.RUNNING
    if policy == "recompute":
        assert failed.status == RequestStatus.PREEMPTED
        assert failed.num_output_tokens == 0
    else:
        assert failed.status == RequestStatus.FINISHED_ERROR


def test_sync_load_bad_blocks_skip_earlier_recurrent_state_checkpoints():
    """A recurrent-state block before the load start is a prefix-cache
    checkpoint the load did not write, so it is not reported as bad."""
    kv_cache_config = make_kv_cache_config(
        block_size=16, mamba_enabled=True, mamba_cache_mode="align"
    )

    def block(block_id: int) -> KVCacheBlock:
        return KVCacheBlock(block_id)

    manager = SimpleNamespace(
        kv_cache_config=kv_cache_config,
        get_blocks=lambda _: SimpleNamespace(
            blocks=([block(1), block(2), block(3)], [block(7), block(8), block(9)])
        ),
    )

    assert KVCacheManager.get_block_ids_from_token(manager, "req", 32) == ([3], [9])
