# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm import SamplingParams, TokensPrompt
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import SupportsHMA
from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (
    ExampleConnector,
    ExampleConnectorMetadata,
)
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

    # The failed request restarts through preemption.
    assert not scheduler.running
    assert request2.status == RequestStatus.PREEMPTED
    assert scheduler.connector.request_finished.call_count == 2

    # It resumes from its valid local prefix without loading again.
    scheduler_output = scheduler.schedule()
    assert scheduler_output.preempted_req_ids == {request2.request_id}
    (resumed,) = scheduler_output.scheduled_new_reqs
    assert resumed.req_id == request2.request_id
    assert resumed.num_computed_tokens == (
        min(invalid_block_idxs) * scheduler.block_size
    )


@pytest.mark.parametrize("async_scheduling", [True, False])
@pytest.mark.parametrize("request_level", [False, True])
def test_sync_load_failure_restarts_running_request(
    async_scheduling: bool, request_level: bool
):
    """A running request cannot be rewound in place after its synchronous load
    failed: the worker keeps its own progress for it, and with async scheduling
    its next step is already in flight. It must be handed to the worker again
    from its valid prefix, without the in-flight token and without another load,
    whether the connector reports the failed blocks or the failed request.
    """
    vllm_config = create_vllm_config(kv_load_failure_policy="recompute")
    vllm_config.scheduler_config.async_scheduling = async_scheduling
    scheduler = create_scheduler(vllm_config)
    block_size = scheduler.block_size
    num_external_tokens = 3 * block_size
    request = create_request(num_tokens=num_external_tokens + 1, max_tokens=3)
    scheduler.add_request(request)

    # The connector offers the same hit on every lookup.
    scheduler.connector = create_mock_connector()
    scheduler.connector.get_num_new_matched_tokens.side_effect = lambda *_: (
        num_external_tokens,
        False,
    )
    scheduler.connector.request_finished.return_value = (False, None)
    scheduler.connector.take_events.return_value = ()

    load_step = scheduler.schedule()
    (block_ids,) = load_step.scheduled_new_reqs[0].block_ids
    in_flight_step = scheduler.schedule()
    assert bool(in_flight_step.num_scheduled_tokens) == async_scheduling

    if request_level:
        # The whole load is bad.
        num_valid_tokens = 0
        load_output = create_model_runner_output(
            [request], failed_sync_recving={request.request_id}, token_id=7
        )
    else:
        num_valid_tokens = block_size
        load_output = create_model_runner_output(
            [request], invalid_block_ids={block_ids[1]}, token_id=7
        )
    scheduler.update_from_output(load_step, load_output)
    if async_scheduling:
        scheduler.update_from_output(
            in_flight_step, create_model_runner_output([request], token_id=8)
        )
    assert not request.output_token_ids

    step = scheduler.schedule()
    assert step.preempted_req_ids == {request.request_id}
    (resumed,) = step.scheduled_new_reqs
    assert resumed.num_computed_tokens == num_valid_tokens
    assert step.num_scheduled_tokens == {
        request.request_id: request.num_tokens - num_valid_tokens
    }

    for _ in range(3):
        assert not request.is_finished()
        scheduler.update_from_output(
            step, create_model_runner_output([request], token_id=9)
        )
        step = scheduler.schedule()
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert list(request.output_token_ids) == [9, 9, 9]


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

    # Both requests read the invalid blocks, so both restart.
    assert not scheduler.running
    assert request1.status == request2.status == RequestStatus.PREEMPTED
    # Nothing from the first invalid block on stays a prefix cache hit: request2
    # computed its own blocks over the invalid ones.
    num_valid_tokens = min(invalid_block_idxs) * scheduler.block_size
    for request in (request1, request2):
        _, num_cached_tokens, _ = scheduler.kv_cache_manager.get_computed_blocks(
            request
        )
        assert num_cached_tokens == num_valid_tokens

    # request1 resumes first and recomputes the shared blocks.
    scheduler_output = scheduler.schedule()
    resumed = scheduler_output.scheduled_new_reqs
    assert resumed[0].req_id == request1.request_id
    assert resumed[0].num_computed_tokens == num_valid_tokens
    for new_req in resumed[1:]:
        assert new_req.req_id == request2.request_id
        assert num_valid_tokens <= new_req.num_computed_tokens <= common_prefix_len


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


@pytest.mark.parametrize("policy", ["recompute", "fail"])
def test_multi_group_sync_load_failure_recovers_sharing_requests(policy: str):
    """A request that already reuses blocks of the failed load is recomputed
    too, since those blocks hold incomplete KV. The blocks it computed over
    them leave the prefix cache as well, or a later request hits them once the
    shared prefix has been recomputed."""
    failed = create_request(num_tokens=64, common_prefix_len=48)
    sharing = create_request(num_tokens=96, common_prefix_len=48)
    scheduler = _multi_group_sync_load_scheduler(policy, failed, sharing)

    scheduler_output = scheduler.schedule()
    failed_blocks = scheduler.kv_cache_manager.get_block_ids(failed.request_id)
    sharing_blocks = scheduler.kv_cache_manager.get_block_ids(sharing.request_id)
    assert sharing_blocks[0][:3] == failed_blocks[0][:3]
    own_blocks = tuple(ids[3:] for ids in sharing_blocks)
    assert _cached(scheduler, own_blocks)

    scheduler.update_from_output(
        scheduler_output,
        create_model_runner_output(
            [failed, sharing], failed_sync_recving={failed.request_id}
        ),
    )

    assert not _cached(scheduler, sharing_blocks)
    assert scheduler.running == []
    expected = (
        RequestStatus.PREEMPTED
        if policy == "recompute"
        else RequestStatus.FINISHED_ERROR
    )
    assert failed.status == sharing.status == expected


@pytest.mark.parametrize("policy", ["recompute", "fail"])
def test_sync_load_failure_evicts_blocks_of_the_step_in_flight(policy: str):
    """Under async scheduling the request's next prefill chunk is scheduled,
    and its blocks cached, before the failed load is reported. Those blocks
    were computed over the failed load and must leave the prefix cache too."""
    vllm_config = create_vllm_config(
        kv_load_failure_policy=policy, max_num_batched_tokens=32
    )
    vllm_config.scheduler_config.async_scheduling = True
    scheduler = create_scheduler(vllm_config)
    block_size = scheduler.block_size
    request = create_request(num_tokens=8 * block_size)
    scheduler.add_request(request)
    scheduler.connector = create_mock_connector()
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(
            {request.request_id: 2 * block_size}, async_load=False
        )
    )
    scheduler.connector.request_finished.return_value = (False, None)
    scheduler.connector.take_events.return_value = ()

    load_step = scheduler.schedule()
    loaded = scheduler.kv_cache_manager.get_block_ids(request.request_id)
    scheduler.schedule()
    (block_ids,) = scheduler.kv_cache_manager.get_block_ids(request.request_id)
    later = ([block_id for block_id in block_ids if block_id not in loaded[0]],)
    assert _cached(scheduler, later)

    scheduler.update_from_output(
        load_step,
        create_model_runner_output([request], failed_sync_recving={request.request_id}),
    )

    assert not _cached(scheduler, (block_ids,))


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


@pytest.mark.parametrize("policy", ["recompute", "fail"])
def test_sync_load_failure_reported_through_both_channels(policy: str):
    """A connector may report a failed load as a request and as blocks in the
    same step; the request is restarted or failed once."""
    scheduler = create_scheduler(create_vllm_config(kv_load_failure_policy=policy))
    block_size = scheduler.block_size
    failed = create_request(num_tokens=4 * block_size)
    other = create_request(num_tokens=4 * block_size)
    scheduler.add_request(failed)
    scheduler.add_request(other)
    scheduler.connector = create_mock_connector()
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(
            {failed.request_id: 2 * block_size}, async_load=False
        )
    )
    scheduler.connector.request_finished.return_value = (False, None)
    scheduler.connector.take_events.return_value = ()

    scheduler_output = scheduler.schedule()
    (block_ids,) = scheduler.kv_cache_manager.get_block_ids(failed.request_id)
    scheduler.update_from_output(
        scheduler_output,
        create_model_runner_output(
            [failed, other],
            failed_sync_recving={failed.request_id},
            invalid_block_ids={block_ids[1]},
        ),
    )

    assert other.status == RequestStatus.RUNNING
    if policy == "recompute":
        assert failed.status == RequestStatus.PREEMPTED
        assert failed.num_preemptions == 1
        assert scheduler.schedule().preempted_req_ids == {failed.request_id}
    else:
        assert failed.status == RequestStatus.FINISHED_ERROR


@pytest.mark.parametrize(("dcp", "expected"), [(1, ([3], [9])), (2, ([2, 3], [9]))])
def test_sync_load_bad_blocks_skip_earlier_recurrent_state_checkpoints(dcp, expected):
    """A recurrent-state block before the load start is a prefix-cache
    checkpoint the load did not write, so it is not reported as bad. Blocks are
    indexed by each manager's block size, which DCP scales for attention."""

    def block(block_id: int) -> KVCacheBlock:
        return KVCacheBlock(block_id)

    manager = SimpleNamespace(
        coordinator=SimpleNamespace(
            single_type_managers=[
                SimpleNamespace(block_size=16 * dcp),
                SimpleNamespace(block_size=16),
            ]
        ),
        get_blocks=lambda _: SimpleNamespace(
            blocks=([block(1), block(2), block(3)], [block(7), block(8), block(9)])
        ),
    )

    assert KVCacheManager.get_block_ids_from_token(manager, "req", 32) == expected


@dataclass
class LostKVConnectorMetadata(ExampleConnectorMetadata):
    # Request of each load in `requests`, in order.
    load_req_ids: list[str] = field(default_factory=list)


class LostKVConnector(ExampleConnector):
    """Reports a load whose stored KV is gone: its blocks, or the request
    itself with `report_requests` in the extra config."""

    def build_connector_meta(self, scheduler_output):
        load_req_ids = [
            new_req.req_id
            for new_req in scheduler_output.scheduled_new_reqs
            if new_req.req_id in self._requests_need_load
        ]
        metadata = super().build_connector_meta(scheduler_output)
        assert isinstance(metadata, ExampleConnectorMetadata)
        return LostKVConnectorMetadata(metadata.requests, load_req_ids)

    def start_load_kv(self, forward_context, **kwargs):
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, LostKVConnectorMetadata)
        self._lost_block_ids: set[int] = set()
        self._lost_req_ids: set[str] = set()
        loads = [r for r in metadata.requests if not r.is_store]
        for req_id, request in zip(metadata.load_req_ids, loads, strict=True):
            if os.listdir(
                self._generate_foldername_debug(request.token_ids, request.mm_hashes)
            ):
                continue
            metadata.requests.remove(request)
            if self._kv_transfer_config.get_from_extra_config("report_requests", False):
                self._lost_req_ids.add(req_id)
            else:
                block_ids = (
                    request.slot_mapping[:: self._block_size] // self._block_size
                )
                self._lost_block_ids.update(block_ids.tolist())
            Path(self._storage_path, "load_failed").touch()
        super().start_load_kv(forward_context, **kwargs)

    def get_block_ids_with_load_errors(self) -> set[int]:
        return self._lost_block_ids

    def get_transfer_results(self, finished_req_ids):
        results = super().get_transfer_results(finished_req_ids)
        results.failed_sync_recving = self._lost_req_ids
        return results


@pytest.mark.timeout(300)
@pytest.mark.parametrize("async_scheduling", [True, False])
@pytest.mark.parametrize("report_requests", [False, True])
def test_recompute_after_failed_sync_load_matches_cold_run(
    vllm_runner, tmp_path, async_scheduling, report_requests
):
    """A synchronous load fills blocks of a request that is already running.
    When it fails, `recompute` must generate what the cold run generated,
    whether the connector reports the failed blocks or the failed request."""
    prompt = TokensPrompt(prompt_token_ids=[100 + i * 37 % 5000 for i in range(135)])
    params = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
    with vllm_runner(
        "facebook/opt-125m",
        enforce_eager=True,
        async_scheduling=async_scheduling,
        kv_transfer_config=KVTransferConfig(
            kv_connector="LostKVConnector",
            kv_connector_module_path=__name__,
            kv_role="kv_both",
            kv_connector_extra_config={
                "shared_storage_path": str(tmp_path),
                "report_requests": report_requests,
            },
            kv_load_failure_policy="recompute",
        ),
        gpu_memory_utilization=0.5,
        max_model_len=256,
        block_size=16,
    ) as runner:
        llm = runner.llm
        computed = llm.generate(prompt, params, use_tqdm=False)
        stored = list(tmp_path.glob("*/*.safetensors"))
        assert stored, "the prompt was not stored"
        for path in stored:
            path.unlink()
        assert llm.reset_prefix_cache()

        recovered = llm.generate(prompt, params, use_tqdm=False)

        assert (tmp_path / "load_failed").exists(), "no load was attempted"
        assert recovered[0].outputs[0].token_ids == computed[0].outputs[0].token_ids
