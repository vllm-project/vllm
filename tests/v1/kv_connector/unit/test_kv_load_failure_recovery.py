# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import pytest

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


@pytest.mark.parametrize("hole_start", [0, 32, 112])
@pytest.mark.parametrize("outcome", ["commit", "abort", "stall"])
@pytest.mark.parametrize("recurrent_groups", [1, 3])
def test_hybrid_attention_replay_shares_normal_scheduler_and_preserves_state(
    hole_start, outcome, recurrent_groups
):
    """A private replay donates only the hole and emits no client output."""
    from vllm.v1.outputs import KVLoadAttestation, KVLoadFailureReport

    config = create_vllm_config(
        kv_load_failure_policy="repair", max_num_batched_tokens=256
    )
    config.scheduler_config.async_scheduling = False
    config.cache_config.mamba_cache_mode = "align"
    config.cache_config.mamba_block_size = 16
    cache_config = make_kv_cache_config(
        16, mamba_enabled=True, mamba_cache_mode="align", num_blocks=100
    )
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec

    cache_config.kv_cache_groups[0].layer_names = [
        "model.layers.0.attn",
        "model.layers.2.attn",
    ]
    state_group = cache_config.kv_cache_groups[1]
    for index in range(1, recurrent_groups):
        cache_config.kv_cache_groups.append(
            KVCacheGroupSpec(
                layer_names=[f"extra_state_{index}"],
                kv_cache_spec=state_group.kv_cache_spec,
            )
        )
    scheduler = create_scheduler(config, kv_cache_config=cache_config)
    parent = create_request(num_tokens=129)
    from vllm.distributed.kv_transfer.kv_connector.v1.base import SupportsHMA

    connector = create_mock_connector()
    connector.__class__ = SupportsHMA
    connector.request_finished_all_groups.return_value = (False, None)
    connector.get_loaded_kv_cache_group_ids.return_value = tuple(
        range(recurrent_groups + 1)
    )
    connector.get_num_new_matched_tokens.side_effect = lambda req, _: (
        (128, True) if req is parent else (0, False)
    )
    connector.request_finished.return_value = (False, None)
    connector.take_events.return_value = ()
    scheduler.connector = connector
    scheduler.add_request(parent)
    load_step = scheduler.schedule()
    assert parent.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    generation = connector.note_transfer_attempt.call_args.args[1]
    before = scheduler.kv_cache_manager.get_block_ids(parent.request_id)
    report = KVLoadFailureReport(
        parent.request_id,
        [before[0][hole_start // 16]],
        (hole_start, hole_start + 16),
        0,
        [0, 2],
        True,
        128,
        source_rank=0,
        transfer_generation=generation,
        failed_blocks_by_group={0: [before[0][hole_start // 16]]},
        healthy_recurrent_group_ids=tuple(range(1, recurrent_groups + 1)),
    )
    attestation = KVLoadAttestation(
        parent.request_id,
        16 + recurrent_groups,
        14 + recurrent_groups,
        2,
        "content",
        14 + recurrent_groups,
        0,
        True,
        transfer_generation=generation,
        source_rank=0,
    )
    output = create_model_runner_output([], finished_recving={parent.request_id})
    output.kv_connector_output.load_failure_reports = [report]
    output.kv_connector_output.load_attestations = [attestation]
    scheduler.update_from_output(load_step, output)
    if outcome == "stall":
        budget = scheduler.max_num_scheduled_tokens
        scheduler.max_num_scheduled_tokens = 0
        for _ in range(Scheduler.REPAIR_MAX_ADMISSION_DEFERRALS + 2):
            scheduler.schedule()
        assert parent.request_id not in scheduler.recovery_plans
        assert not any(req.is_internal_kv_repair for req in scheduler.requests.values())
        scheduler.max_num_scheduled_tokens = budget
        assert parent.request_id in scheduler.schedule().num_scheduled_tokens
        return
    ordinary = create_request(num_tokens=17)
    scheduler.add_request(ordinary)
    step = scheduler.schedule()
    assert parent.request_id not in step.num_scheduled_tokens
    assert ordinary.request_id in step.num_scheduled_tokens
    replays = [
        scheduler.requests[rid]
        for rid in step.num_scheduled_tokens
        if rid != ordinary.request_id
    ]
    assert len(replays) == 1 and replays[0].is_internal_kv_repair
    replay = replays[0]
    source = scheduler.kv_cache_manager.get_block_ids(replay.request_id)
    runner_output = create_model_runner_output(
        [scheduler.requests[rid] for rid in step.num_scheduled_tokens]
    )
    if outcome == "abort":
        scheduler.finish_requests(parent.request_id, RequestStatus.FINISHED_ABORTED)
    response = scheduler.update_from_output(
        step,
        runner_output,
    )
    if outcome == "abort":
        assert parent.request_id not in scheduler.requests
        assert replay.request_id not in scheduler.requests
        return
    after = scheduler.kv_cache_manager.get_block_ids(parent.request_id)
    assert after[1:] == before[1:]
    slot = hole_start // 16
    assert after[0][:slot] == before[0][:slot]
    assert after[0][slot] == source[0][slot]
    assert after[0][slot + 1 :] == before[0][slot + 1 :]
    assert replay.request_id not in scheduler.requests
    assert all(
        row.request_id != replay.request_id
        for client in response.values()
        for row in client.outputs
    )
    assert parent.request_id in scheduler.schedule().num_scheduled_tokens


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
