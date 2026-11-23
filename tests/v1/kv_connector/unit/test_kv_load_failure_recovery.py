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
    scheduler.connector.on_load_failure.assert_called_once_with({request2.request_id})
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
    scheduler.connector.on_load_failure.assert_called_once_with({request2.request_id})
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


@pytest.fixture
def persistent_store(request, monkeypatch, tmp_path):
    """Keep the storage lookup positive after a reported data-plane failure."""
    from transformers import OPTConfig

    from tests.v1.kv_connector.umbp_test_utils import MemoryRuntime
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store import (
        scheduler as mooncake_scheduler,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.data import (
        MooncakeLookupResult,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.umbp.runtime.factory import (
        UMBPRuntimeFactory,
    )

    class Lookup:
        def __init__(self):
            self.calls = []
            self.enabled = True

        def lookup(self, *args, **kwargs):
            self.calls.append(args)
            if len(args) == 1:
                return [self.enabled] * len(args[0])
            return MooncakeLookupResult(32 if self.enabled else 0)

        def discard(self, request_id):
            pass

        def prefetch(self, *args):
            pass

        def close(self):
            pass

    lookup = Lookup()

    class Runtime(MemoryRuntime):
        def create_scheduler_handle(self, namespace, topology, layout):
            return lookup

    monkeypatch.setitem(UMBPRuntimeFactory._builders, "embedded", lambda _: Runtime())
    monkeypatch.setattr(mooncake_scheduler, "LookupKeyClient", lambda _: lookup)
    OPTConfig().save_pretrained(tmp_path)
    return request.param, lookup, str(tmp_path)


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("policy", ["fail", "recompute"])
@pytest.mark.parametrize(
    "persistent_store", ["UMBPStoreConnector", "MooncakeStoreConnector"], indirect=True
)
def test_store_failure_recovery_survives_preemption(persistent_store, hybrid, policy):
    """Real connectors must not reload a failed request after GPU-only preemption."""
    connector_name, lookup, model = persistent_store
    config = create_vllm_config(
        model=model,
        kv_connector=connector_name,
        kv_role="kv_consumer",
        max_num_batched_tokens=16,
        kv_load_failure_policy=policy,
        kv_connector_extra_config={"key_namespace": "recovery-test"},
    )
    cache = make_kv_cache_config(16, mamba_enabled=hybrid, enable_prefix_caching=True)
    scheduler = create_scheduler(config, kv_cache_config=cache)
    victim = create_request(request_id=9123, num_tokens=49, max_tokens=2)
    scheduler.add_request(victim)
    scheduled = scheduler.schedule()
    assert victim.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert len(lookup.calls) == 1
    scheduler.update_from_output(
        scheduled,
        create_model_runner_output(
            [],
            finished_recving={victim.request_id},
            failed_recving={victim.request_id} if hybrid else None,
            invalid_block_ids=(
                None
                if hybrid
                else {scheduler.kv_cache_manager.get_block_ids(victim.request_id)[0][0]}
            ),
        ),
    )
    if policy == "fail":
        assert victim.status == RequestStatus.FINISHED_ERROR
    else:
        first = scheduler.schedule()
        assert first.num_scheduled_tokens[victim.request_id] == 16
        output = create_model_runner_output([victim])
        output.sampled_token_ids = [[]]
        scheduler.update_from_output(first, output)
        # This public operation invokes the real preemption path. The external
        # store is not reset; its stale-positive metadata remains unchanged.
        assert scheduler.reset_prefix_cache(reset_running_requests=True)
        assert victim.status == RequestStatus.PREEMPTED
        assert victim.num_preemptions == 1
        computed = 0
        for _ in range(8):
            resumed = scheduler.schedule()
            count = resumed.num_scheduled_tokens.get(victim.request_id, 0)
            assert count > 0
            computed += count
            output = create_model_runner_output([victim])
            if victim.num_computed_tokens < victim.num_tokens:
                output.sampled_token_ids = [[]]
            scheduler.update_from_output(resumed, output)
            if victim.is_finished():
                break
        assert victim.is_finished()
        assert computed == 50
        assert len(lookup.calls) == 1

    scheduler.schedule()  # Deliver terminal IDs to the connector.
    assert scheduler.reset_prefix_cache()
    # Reusing a terminated ID must not inherit a previous request's bypass.
    replacement = create_request(request_id=9123, num_tokens=65)
    scheduler.add_request(replacement)
    scheduler.schedule()
    assert replacement.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert len(lookup.calls) == 2
    scheduler.shutdown()


@pytest.mark.parametrize(
    "persistent_store", ["UMBPStoreConnector", "MooncakeStoreConnector"], indirect=True
)
def test_failed_store_load_stays_local_after_capacity_preemption(persistent_store):
    """Growing the first request must not send the evicted second one back to IO."""
    connector_name, lookup, model = persistent_store
    config = create_vllm_config(
        model=model,
        kv_connector=connector_name,
        kv_role="kv_consumer",
        max_num_batched_tokens=100,
        kv_load_failure_policy="recompute",
        kv_connector_extra_config={"key_namespace": "recovery-test"},
    )
    # Ten usable blocks fit two five-block prompts, but not the next decode page.
    scheduler = create_scheduler(config, num_blocks=11)
    first = create_request(num_tokens=80, max_tokens=2)
    victim = create_request(num_tokens=80, max_tokens=2)
    lookup.enabled = False
    scheduler.add_request(first)
    first_step = scheduler.schedule()
    assert first_step.num_scheduled_tokens[first.request_id] == 80
    lookup.enabled = True
    scheduler.add_request(victim)
    load_step = scheduler.schedule()
    assert victim.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.update_from_output(
        load_step,
        create_model_runner_output(
            [],
            finished_recving={victim.request_id},
            invalid_block_ids={
                scheduler.kv_cache_manager.get_block_ids(victim.request_id)[0][0]
            },
        ),
    )
    local_step = scheduler.schedule()
    assert local_step.num_scheduled_tokens[victim.request_id] == 80
    scheduler.update_from_output(first_step, create_model_runner_output([first]))
    scheduler.update_from_output(local_step, create_model_runner_output([victim]))
    pressure_step = scheduler.schedule()
    assert victim.status == RequestStatus.PREEMPTED
    assert victim.num_preemptions == 1
    assert pressure_step.num_scheduled_tokens == {first.request_id: 1}
    scheduler.update_from_output(pressure_step, create_model_runner_output([first]))
    resumed = scheduler.schedule()
    assert len(lookup.calls) == 2
    # Four valid local prefix blocks survive preemption and remain reusable.
    assert resumed.num_scheduled_tokens[victim.request_id] == 17
    scheduler.update_from_output(resumed, create_model_runner_output([victim]))
    assert first.is_finished() and victim.is_finished()
    scheduler.shutdown()


@pytest.mark.parametrize("persistent_store", ["UMBPStoreConnector"], indirect=True)
def test_failed_umbp_load_recomputed_pages_are_stored(persistent_store):
    """A failed load is not evidence that the recomputed prefix is resident."""
    connector_name, lookup, model = persistent_store
    config = create_vllm_config(
        model=model,
        kv_connector=connector_name,
        kv_role="kv_both",
        max_num_batched_tokens=16,
        kv_load_failure_policy="recompute",
        kv_connector_extra_config={"key_namespace": "recovery-test"},
    )
    scheduler = create_scheduler(config)
    victim = create_request(num_tokens=49)
    scheduler.add_request(victim)
    loading = scheduler.schedule()
    assert victim.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.update_from_output(
        loading,
        create_model_runner_output(
            [], finished_recving={victim.request_id}, failed_recving={victim.request_id}
        ),
    )
    local_step = scheduler.schedule()
    assert local_step.num_scheduled_tokens[victim.request_id] == 16
    stores = local_step.kv_connector_metadata.store_requests
    assert len(stores[victim.request_id].keys) == 1
    assert len(lookup.calls) == 1
    scheduler.shutdown()
