# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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

    scheduler.connector = Mock()
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(req_num_new_matched_tokens, async_load=True)
    )
    scheduler.connector.take_events.return_value = ()

    scheduler_output = scheduler.schedule()

    assert len(scheduler.waiting) == 0
    assert len(scheduler.skipped_waiting) == 3
    for request in scheduler.skipped_waiting:
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
    assert len(scheduler.skipped_waiting) == 3
    for request in scheduler.skipped_waiting:
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

    scheduler.connector = Mock()
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


def test_sync_load_failure_discards_multitoken_async_frames():
    vllm_config = create_vllm_config(kv_load_failure_policy="recompute")
    vllm_config.scheduler_config.async_scheduling = True
    scheduler = create_scheduler(vllm_config)
    # CPU-only unit tests cannot enable MRV2 without Triton. Worker-side
    # consumption of the rewind marker is covered separately.
    scheduler.use_v2_model_runner = True

    # The second frame carries one sampled token plus three speculative tokens.
    scheduler.num_spec_tokens = 3
    request = create_request(
        num_tokens=3 * scheduler.block_size,
        max_tokens=16,
    )
    scheduler.add_request(request)

    scheduler.connector = Mock()
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(
            {request.request_id: 2 * scheduler.block_size},
            async_load=False,
        )
    )
    scheduler.connector.request_finished.return_value = (False, None)
    scheduler.connector.take_events.return_value = ()

    failed_output = scheduler.schedule()
    stale_output = scheduler.schedule()
    assert request.num_output_placeholders == 5
    assert len(stale_output.scheduled_spec_decode_tokens[request.request_id]) == 3

    block_ids = failed_output.scheduled_new_reqs[0].block_ids[0]
    model_runner_output = create_model_runner_output(
        [request], invalid_block_ids={block_ids[0]}, token_id=101
    )
    scheduler.update_from_output(failed_output, model_runner_output)

    stale_tokens = stale_output.num_scheduled_tokens[request.request_id]
    assert request.num_computed_tokens == 0
    assert request.num_output_placeholders == 0
    assert request.num_stale_output_tokens == stale_tokens
    assert request.drop_stale_output
    assert request.num_output_tokens == 0
    assert request.is_prefill_chunk
    assert request in scheduler._inflight_prefills
    assert scheduler.rewound_req_ids == {request.request_id}

    recovery_output = scheduler.schedule()
    assert recovery_output.scheduled_cached_reqs.rewound_req_ids == {request.request_id}
    assert recovery_output.scheduled_cached_reqs.num_computed_tokens == [0]
    assert not scheduler.rewound_req_ids

    scheduler.update_from_output(
        stale_output, create_model_runner_output([request], token_id=102)
    )
    assert request.num_stale_output_tokens == 0
    assert request.num_output_tokens == 0

    scheduler.update_from_output(
        recovery_output, create_model_runner_output([request], token_id=103)
    )
    assert list(request.output_token_ids) == [103]
    assert request.num_output_placeholders == 0
    assert request.num_in_flight_tokens == 0


@pytest.mark.parametrize("preexisting_stale_tokens", [0, 1])
def test_sync_load_failure_shared_blocks_rewinds_all_async_frames(
    preexisting_stale_tokens: int,
):
    vllm_config = create_vllm_config(kv_load_failure_policy="recompute")
    vllm_config.scheduler_config.async_scheduling = True
    vllm_config.scheduler_config.long_prefill_token_threshold = 32
    scheduler = create_scheduler(vllm_config)
    scheduler.use_v2_model_runner = True

    num_prompt_blocks = 100
    num_external_computed_blocks = 99
    num_common_prefix_blocks = 50
    num_prompt_tokens = num_prompt_blocks * scheduler.block_size
    num_external_computed_tokens = num_external_computed_blocks * scheduler.block_size
    common_prefix_len = num_common_prefix_blocks * scheduler.block_size

    request1 = create_request(
        num_tokens=num_prompt_tokens,
        common_prefix_len=common_prefix_len,
    )
    scheduler.add_request(request1)
    request2 = create_request(
        num_tokens=num_prompt_tokens,
        common_prefix_len=common_prefix_len,
    )
    scheduler.add_request(request2)

    scheduler.connector = Mock()
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(
            {request1.request_id: num_external_computed_tokens},
            async_load=False,
        )
    )
    scheduler.connector.request_finished.return_value = (False, None)
    scheduler.connector.take_events.return_value = ()

    failed_output = scheduler.schedule()
    stale_output = scheduler.schedule()
    assert stale_output.num_scheduled_tokens[request1.request_id] == 1
    assert stale_output.num_scheduled_tokens[request2.request_id] > 0
    assert (
        request2.num_in_flight_tokens
        > failed_output.num_scheduled_tokens[request2.request_id]
    )
    # A shared peer can also have an older frame that was already detached
    # from num_computed_tokens by a preemption/reset but has not returned yet.
    # It must not be subtracted a second time while finding the safe prefix.
    request2.num_computed_tokens -= preexisting_stale_tokens
    request2.num_stale_output_tokens = preexisting_stale_tokens

    req1_block_ids = failed_output.scheduled_new_reqs[0].block_ids[0]
    req2_block_ids = failed_output.scheduled_new_reqs[1].block_ids[0]
    assert req1_block_ids[0] == req2_block_ids[0]

    scheduler.update_from_output(
        failed_output,
        create_model_runner_output(
            [request1, request2],
            invalid_block_ids={req1_block_ids[0]},
        ),
    )

    assert request1.num_computed_tokens == 0
    # request2 relies on request1 to restore the shared invalid block. Its own
    # safe offset is the confirmed local prefix, excluding every async frame
    # still in flight when the load failure is reported.
    assert request2.num_computed_tokens == common_prefix_len
    assert (
        request2.num_stale_output_tokens
        == stale_output.num_scheduled_tokens[request2.request_id]
    )
    assert scheduler.rewound_req_ids == {
        request1.request_id,
        request2.request_id,
    }

    recovery_output = scheduler.schedule()
    assert recovery_output.num_scheduled_tokens == {request1.request_id: 32}


def _setup_shared_sync_load_failure(
    *,
    async_scheduling: bool = False,
    num_sharers: int = 1,
    invalid_block_idx: int = 0,
    num_external_computed_blocks: int = 9,
    num_common_prefix_blocks: int = 4,
    num_sharer_external_blocks: int = 0,
    max_num_batched_tokens: int | None = None,
    long_prefill_token_threshold: int = 32,
) -> tuple[Scheduler, Request, list[Request]]:
    """Create a small owner/sharer recovery with a 32-token prefill cap."""
    vllm_config = create_vllm_config(
        max_num_batched_tokens=(
            max_num_batched_tokens or max(64, 32 * (num_sharers + 1))
        ),
        kv_load_failure_policy="recompute",
    )
    vllm_config.scheduler_config.async_scheduling = async_scheduling
    vllm_config.scheduler_config.long_prefill_token_threshold = (
        long_prefill_token_threshold
    )
    scheduler = create_scheduler(vllm_config)
    scheduler.use_v2_model_runner = True

    num_prompt_blocks = 10
    num_prompt_tokens = num_prompt_blocks * scheduler.block_size
    num_external_computed_tokens = num_external_computed_blocks * scheduler.block_size
    common_prefix_len = num_common_prefix_blocks * scheduler.block_size

    owner = create_request(
        num_tokens=num_prompt_tokens,
        common_prefix_len=common_prefix_len,
    )
    sharers = [
        create_request(
            num_tokens=num_prompt_tokens,
            common_prefix_len=common_prefix_len,
        )
        for _ in range(num_sharers)
    ]
    scheduler.add_request(owner)
    for sharer in sharers:
        scheduler.add_request(sharer)

    scheduler.connector = Mock()
    external_matches = {owner.request_id: num_external_computed_tokens}
    external_matches.update(
        {
            sharer.request_id: num_sharer_external_blocks * scheduler.block_size
            for sharer in sharers
        }
    )
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(
            external_matches,
            async_load=False,
        )
    )
    scheduler.connector.request_finished.return_value = (False, None)
    scheduler.connector.take_events.return_value = ()

    failed_output = scheduler.schedule()
    owner_block_ids = failed_output.scheduled_new_reqs[0].block_ids[0]
    for scheduled_sharer in failed_output.scheduled_new_reqs[1:]:
        assert (
            scheduled_sharer.block_ids[0][:num_common_prefix_blocks]
            == owner_block_ids[:num_common_prefix_blocks]
        )

    scheduler.update_from_output(
        failed_output,
        create_model_runner_output(
            [owner, *sharers],
            invalid_block_ids={owner_block_ids[invalid_block_idx]},
        ),
    )

    invalid_prefix_len = invalid_block_idx * scheduler.block_size
    assert owner.num_computed_tokens == invalid_prefix_len
    for sharer in sharers:
        assert sharer.num_computed_tokens == common_prefix_len
    return scheduler, owner, sharers


def _update_prefill_frame(
    scheduler: Scheduler,
    scheduler_output,
    requests: list[Request],
) -> None:
    model_runner_output = create_model_runner_output(requests)
    model_runner_output.sampled_token_ids = [[] for _ in requests]
    scheduler.update_from_output(scheduler_output, model_runner_output)


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_shared_block_recovery_waits_for_confirmed_owner_progress(
    async_scheduling: bool,
):
    scheduler, owner, (sharer,) = _setup_shared_sync_load_failure(
        async_scheduling=async_scheduling
    )

    first_recovery = scheduler.schedule()
    assert first_recovery.num_scheduled_tokens == {owner.request_id: 32}

    # Async scheduling may enqueue another frame before the first one returns.
    # Optimistically scheduled owner tokens must not release the sharer.
    second_recovery = scheduler.schedule()
    assert second_recovery.num_scheduled_tokens == {owner.request_id: 32}

    _update_prefill_frame(scheduler, first_recovery, [owner])
    assert owner.num_computed_tokens - owner.num_in_flight_tokens == 32

    _update_prefill_frame(scheduler, second_recovery, [owner])
    assert owner.num_computed_tokens - owner.num_in_flight_tokens == 64

    released = scheduler.schedule()
    assert released.num_scheduled_tokens.get(sharer.request_id, 0) > 0


def test_shared_block_recovery_waits_all_sharers():
    scheduler, owner, sharers = _setup_shared_sync_load_failure(num_sharers=3)

    recovery = scheduler.schedule()
    assert recovery.num_scheduled_tokens == {owner.request_id: 32}
    assert all(
        sharer.request_id not in recovery.num_scheduled_tokens for sharer in sharers
    )


def test_shared_block_recovery_allows_ordered_same_frame_release():
    scheduler, owner, (sharer,) = _setup_shared_sync_load_failure()

    first_owner_frame = scheduler.schedule()
    assert first_owner_frame.num_scheduled_tokens == {owner.request_id: 32}
    _update_prefill_frame(scheduler, first_owner_frame, [owner])

    # The owner has 32 confirmed tokens and writes the remaining 32 shared
    # tokens in this frame. MRV2 writes KV for the whole frame before any
    # request's attention reads it, so the sharer's private suffix is safe to
    # schedule in the same frame.
    boundary_frame = scheduler.schedule()
    assert boundary_frame.num_scheduled_tokens == {
        owner.request_id: 32,
        sharer.request_id: 32,
    }


def test_shared_block_recovery_v1_requires_confirmed_release():
    scheduler, owner, (sharer,) = _setup_shared_sync_load_failure()
    scheduler.use_v2_model_runner = False

    first_owner_frame = scheduler.schedule()
    _update_prefill_frame(scheduler, first_owner_frame, [owner])

    boundary_frame = scheduler.schedule()
    assert boundary_frame.num_scheduled_tokens == {owner.request_id: 32}
    _update_prefill_frame(scheduler, boundary_frame, [owner])

    confirmed_release = scheduler.schedule()
    assert confirmed_release.num_scheduled_tokens.get(sharer.request_id, 0) > 0


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize(
    "threshold,batch_budget,invalid_block_idx",
    [
        (1, 32, 0),
        (15, 32, 0),
        (16, 32, 0),
        (17, 32, 0),
        (31, 32, 0),
        (32, 32, 0),
        (33, 64, 0),
        (64, 64, 0),
        (1, 64, 3),
        (15, 64, 3),
        (16, 64, 3),
        (17, 64, 3),
        (31, 64, 3),
        (32, 64, 3),
        (33, 96, 3),
        (64, 96, 3),
    ],
)
def test_shared_block_recovery_scheduling_boundary_matrix(
    async_scheduling: bool,
    threshold: int,
    batch_budget: int,
    invalid_block_idx: int,
):
    scheduler, owner, (sharer,) = _setup_shared_sync_load_failure(
        async_scheduling=async_scheduling,
        invalid_block_idx=invalid_block_idx,
        max_num_batched_tokens=batch_budget,
        long_prefill_token_threshold=threshold,
    )
    required_end = 4 * scheduler.block_size

    for _ in range(required_end + 1):
        confirmed_before = scheduler._get_confirmed_num_computed_tokens(owner)
        active_before = owner.num_in_flight_tokens - owner.num_stale_output_tokens
        owner_offset_before = owner.num_computed_tokens
        output = scheduler.schedule()
        owner_scheduled = output.num_scheduled_tokens.get(owner.request_id, 0)

        if sharer.request_id in output.num_scheduled_tokens:
            ordered_by_confirmed_progress = confirmed_before >= required_end
            ordered_inside_frame = (
                active_before == 0
                and owner_offset_before == confirmed_before
                and confirmed_before + owner_scheduled >= required_end
            )
            assert ordered_by_confirmed_progress or ordered_inside_frame
            break

        assert owner_scheduled > 0
        scheduled_requests = [
            request
            for request in (owner, sharer)
            if request.request_id in output.num_scheduled_tokens
        ]
        _update_prefill_frame(scheduler, output, scheduled_requests)
    else:
        pytest.fail("shared-block recovery dependency did not make progress")


def test_shared_block_recovery_reassigns_cancelled_owner():
    scheduler, owner, (sharer,) = _setup_shared_sync_load_failure()
    owner_block_ids = scheduler.kv_cache_manager.get_block_ids(owner.request_id)[0]
    sharer_block_ids = scheduler.kv_cache_manager.get_block_ids(sharer.request_id)[0]
    owner_only_block_id = next(
        block_id for block_id in owner_block_ids if block_id not in sharer_block_ids
    )
    assert (
        scheduler.kv_cache_manager.block_pool.blocks[owner_only_block_id].block_hash
        is not None
    )

    scheduler.finish_requests(owner.request_id, RequestStatus.FINISHED_ABORTED)

    assert sharer.num_computed_tokens == 0
    assert (
        scheduler.kv_cache_manager.block_pool.blocks[owner_only_block_id].block_hash
        is None
    )
    recovery = scheduler.schedule()
    assert recovery.num_scheduled_tokens == {sharer.request_id: 32}


@pytest.mark.parametrize("invalid_block_idx", [0, 2])
def test_shared_block_recovery_quarantines_prefix_cache_hits(
    invalid_block_idx: int,
):
    scheduler, owner, (sharer,) = _setup_shared_sync_load_failure(
        invalid_block_idx=invalid_block_idx,
        max_num_batched_tokens=96,
    )
    expected_valid_tokens = invalid_block_idx * scheduler.block_size

    newcomer = create_request(
        num_tokens=10 * scheduler.block_size,
        common_prefix_len=4 * scheduler.block_size,
    )
    scheduler.add_request(newcomer)

    recovery = scheduler.schedule()
    assert recovery.num_scheduled_tokens.get(owner.request_id) == 32
    assert recovery.num_scheduled_tokens.get(newcomer.request_id) == 32
    newcomer_data = next(
        req for req in recovery.scheduled_new_reqs if req.req_id == newcomer.request_id
    )
    assert newcomer_data.num_computed_tokens == expected_valid_tokens

    owner_block_ids = scheduler.kv_cache_manager.get_block_ids(owner.request_id)[0]
    newcomer_block_ids = scheduler.kv_cache_manager.get_block_ids(newcomer.request_id)[
        0
    ]
    assert owner_block_ids[invalid_block_idx] != newcomer_block_ids[invalid_block_idx]
    assert (sharer.request_id in recovery.num_scheduled_tokens) is (
        invalid_block_idx == 2
    )


def test_shared_block_recovery_quarantines_divergent_fallback_hit():
    scheduler, owner, (sharer,) = _setup_shared_sync_load_failure(
        invalid_block_idx=2,
        max_num_batched_tokens=96,
    )
    expected_valid_tokens = 2 * scheduler.block_size

    newcomer = create_request(
        num_tokens=10 * scheduler.block_size,
        common_prefix_len=4 * scheduler.block_size,
    )
    scheduler.add_request(newcomer)

    get_local_hit = scheduler._get_local_prefix_cache_hit

    def divergent_local_hit(request):
        blocks, tokens, boundary, _ = get_local_hit(request)
        return blocks, tokens, boundary, True

    scheduler._get_local_prefix_cache_hit = divergent_local_hit
    recovery = scheduler.schedule()
    newcomer_data = next(
        req for req in recovery.scheduled_new_reqs if req.req_id == newcomer.request_id
    )
    assert newcomer_data.num_computed_tokens == expected_valid_tokens

    owner_block_ids = scheduler.kv_cache_manager.get_block_ids(owner.request_id)[0]
    newcomer_block_ids = scheduler.kv_cache_manager.get_block_ids(newcomer.request_id)[
        0
    ]
    assert owner_block_ids[2] != newcomer_block_ids[2]


def test_shared_block_recovery_quarantines_optimistic_downstream_blocks():
    scheduler, owner, (sharer,) = _setup_shared_sync_load_failure(
        invalid_block_idx=5,
        num_external_computed_blocks=6,
        num_common_prefix_blocks=8,
    )
    scheduler.finish_requests(sharer.request_id, RequestStatus.FINISHED_ABORTED)

    first_recovery = scheduler.schedule()
    assert first_recovery.num_scheduled_tokens == {owner.request_id: 32}
    _update_prefill_frame(scheduler, first_recovery, [owner])
    assert owner.num_computed_tokens - owner.num_in_flight_tokens == 112

    newcomer = create_request(
        num_tokens=10 * scheduler.block_size,
        common_prefix_len=8 * scheduler.block_size,
    )
    scheduler.add_request(newcomer)

    next_recovery = scheduler.schedule()
    newcomer_data = next(
        req
        for req in next_recovery.scheduled_new_reqs
        if req.req_id == newcomer.request_id
    )
    # The failed frame optimistically cached through token 128. Only 112
    # tokens have been repaired and confirmed, so the final old block remains
    # quarantined even though the original invalid block is already repaired.
    assert newcomer_data.num_computed_tokens == 112
    owner_block_ids = scheduler.kv_cache_manager.get_block_ids(owner.request_id)[0]
    newcomer_block_ids = scheduler.kv_cache_manager.get_block_ids(newcomer.request_id)[
        0
    ]
    assert owner_block_ids[7] != newcomer_block_ids[7]


def test_shared_block_recovery_orders_partially_shared_repair_chain():
    scheduler, owner, (sharer,) = _setup_shared_sync_load_failure(
        invalid_block_idx=5,
        num_external_computed_blocks=9,
        num_common_prefix_blocks=8,
        num_sharer_external_blocks=1,
    )

    # The sharer keeps the shared prefix repaired by the owner, but rewinds
    # its private downstream block and becomes the next repair owner.
    assert owner.num_computed_tokens == 80
    assert sharer.num_computed_tokens == 128
    owner_block_ids = scheduler.kv_cache_manager.get_block_ids(owner.request_id)[0]
    sharer_block_ids = scheduler.kv_cache_manager.get_block_ids(sharer.request_id)[0]
    assert owner_block_ids[:8] == sharer_block_ids[:8]
    assert owner_block_ids[8] != sharer_block_ids[8]

    first_owner_frame = scheduler.schedule()
    assert first_owner_frame.num_scheduled_tokens == {owner.request_id: 32}
    _update_prefill_frame(scheduler, first_owner_frame, [owner])

    boundary_frame = scheduler.schedule()
    assert boundary_frame.num_scheduled_tokens.get(owner.request_id) == 32
    assert boundary_frame.num_scheduled_tokens.get(sharer.request_id, 0) > 0


def test_shared_block_recovery_transfers_partially_repaired_owner():
    scheduler, owner, (sharer,) = _setup_shared_sync_load_failure()

    owner_frame = scheduler.schedule()
    _update_prefill_frame(scheduler, owner_frame, [owner])
    assert owner.num_computed_tokens - owner.num_in_flight_tokens == 32

    scheduler.running.remove(owner)
    scheduler._preempt_request(owner, 0.0)

    # The first two blocks are confirmed. The sharer takes ownership at the
    # first still-invalid block instead of restarting from zero or hanging.
    assert owner.status == RequestStatus.PREEMPTED
    assert sharer.num_computed_tokens == 32
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens({}, async_load=False)
    )
    recovery = scheduler.schedule()
    assert recovery.num_scheduled_tokens.get(sharer.request_id, 0) > 0


def test_shared_block_recovery_does_not_rewind_after_confirmed_owner_exit():
    scheduler, owner, (sharer,) = _setup_shared_sync_load_failure()

    # Model the owner exiting immediately after the shared repair boundary was
    # confirmed. Lifecycle cleanup must resolve the dependency before it
    # considers ownership transfer.
    owner.num_computed_tokens = 64
    scheduler.finish_requests(owner.request_id, RequestStatus.FINISHED_ABORTED)

    assert sharer.num_computed_tokens == 64
    recovery = scheduler.schedule()
    assert recovery.num_scheduled_tokens.get(sharer.request_id, 0) > 0


def test_shared_block_recovery_keeps_independent_owners_parallel():
    vllm_config = create_vllm_config(
        max_num_batched_tokens=128,
        kv_load_failure_policy="recompute",
    )
    vllm_config.scheduler_config.long_prefill_token_threshold = 32
    scheduler = create_scheduler(vllm_config)
    scheduler.use_v2_model_runner = True

    prompt_tokens = 10 * scheduler.block_size
    external_tokens = 9 * scheduler.block_size
    shared_tokens = 4 * scheduler.block_size
    owner_a = create_request(
        num_tokens=prompt_tokens,
        common_prefix_len=shared_tokens,
    )
    sharer_a = create_request(
        num_tokens=prompt_tokens,
        common_prefix_len=shared_tokens,
    )
    owner_b = create_request(request_id=700, num_tokens=prompt_tokens)
    sharer_b = create_request(request_id=700, num_tokens=prompt_tokens)
    sharer_b.request_id = "id-701"
    requests = [owner_a, sharer_a, owner_b, sharer_b]
    for request in requests:
        scheduler.add_request(request)

    scheduler.connector = Mock()
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(
            {
                owner_a.request_id: external_tokens,
                owner_b.request_id: external_tokens,
            },
            async_load=False,
        )
    )
    scheduler.connector.request_finished.return_value = (False, None)
    scheduler.connector.take_events.return_value = ()

    failed_output = scheduler.schedule()
    block_ids_by_req = {
        req.req_id: req.block_ids[0] for req in failed_output.scheduled_new_reqs
    }
    assert (
        block_ids_by_req[owner_a.request_id][0]
        != block_ids_by_req[owner_b.request_id][0]
    )

    scheduler.update_from_output(
        failed_output,
        create_model_runner_output(
            requests,
            invalid_block_ids={
                block_ids_by_req[owner_a.request_id][0],
                block_ids_by_req[owner_b.request_id][0],
            },
        ),
    )

    recovery = scheduler.schedule()
    assert recovery.num_scheduled_tokens == {
        owner_a.request_id: 32,
        owner_b.request_id: 32,
    }


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

    scheduler.connector = Mock()
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
    for request in scheduler_output.scheduled_new_reqs:
        assert request.num_computed_tokens == expected_computed_tokens[request.req_id]
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

    scheduler.connector = Mock()
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(req_num_new_matched_tokens, async_load=True)
    )
    scheduler.connector.take_events.return_value = ()

    scheduler_output = scheduler.schedule()

    assert len(scheduler.waiting) == 0
    assert len(scheduler.skipped_waiting) == 1
    assert scheduler.skipped_waiting.peek_request().request_id == request.request_id
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
        assert len(scheduler.skipped_waiting) == 1
        assert scheduler.skipped_waiting.peek_request().request_id == request.request_id
        assert request.num_computed_tokens == (
            min_invalid_block_idx * scheduler.block_size
        )
        assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        assert scheduler.failed_recving_kv_req_ids == {request.request_id}
        assert scheduler.connector.get_num_new_matched_tokens.call_count == 1
