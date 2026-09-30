# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from unittest.mock import Mock, patch

import pytest

from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    KVConnectorOutput,
    ModelRunnerOutput,
)
from vllm.v1.request import FinishReason, RequestStatus

from .utils import (
    assert_scheduler_empty,
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
    make_kv_cache_config,
)

pytestmark = pytest.mark.cpu_test


def _num_waiting_requests(scheduler) -> int:
    return len(scheduler.waiting) + len(scheduler.skipped_waiting)


def _shared_load_scheduler(monkeypatch, policy="recompute", matched_tokens=32):
    config = create_vllm_config(
        kv_connector="MockKVConnector",
        kv_connector_extra_config={
            "matched_tokens": matched_tokens,
            "is_async": True,
        },
        kv_load_failure_policy=policy,
    )
    scheduler = create_scheduler(config, num_blocks=32)
    connector = scheduler.connector
    monkeypatch.setattr(connector, "supports_shared_prefix_loads", lambda: True)
    connector.update_state_after_alloc = Mock()
    return scheduler


def _abort_scheduled_batch(scheduler, output, requests):
    blocks = [
        block
        for request in requests
        for block in scheduler.kv_cache_manager.get_blocks(request.request_id).blocks[0]
    ]
    scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)
    if scheduler.defer_block_free and blocks:
        # Aborting cannot fence an already scheduled batch's GPU writes.
        assert scheduler.deferred_frees
        assert all(block.ref_cnt > 0 for block in blocks)
    scheduler.update_from_output(output, create_model_runner_output(reqs=requests))
    scheduler.schedule()
    assert_scheduler_empty(scheduler)


@pytest.mark.parametrize("cancel", [None, "owner", "follower", "both"])
def test_shared_external_prefix_lifetime(monkeypatch, cancel):
    """One transfer owns the write; aborting either reader preserves the other."""
    scheduler = _shared_load_scheduler(monkeypatch)
    owner, follower = [
        create_request(num_tokens=48, common_prefix_len=32) for _ in range(2)
    ]
    for request in (owner, follower):
        scheduler.add_request(request)
    output = scheduler.schedule()
    manager = scheduler.kv_cache_manager
    blocks = list(manager.get_blocks(owner.request_id).blocks[0])
    assert manager.get_block_ids(owner.request_id) == manager.get_block_ids(
        follower.request_id
    )
    assert len(blocks) == 2
    assert all(block.ref_cnt == 2 and block.block_hash is None for block in blocks)
    assert scheduler.connector.update_state_after_alloc.call_count == 1
    assert not output.num_scheduled_tokens
    assert not scheduler.reset_connector_cache()

    if cancel in ("owner", "both"):
        scheduler.finish_requests(owner.request_id, RequestStatus.FINISHED_ABORTED)
    if cancel in ("follower", "both"):
        scheduler.finish_requests(follower.request_id, RequestStatus.FINISHED_ABORTED)
    # The owner pins the destination until the transfer reports completion.
    assert all(block.ref_cnt >= 1 for block in blocks)
    scheduler.update_from_output(
        output,
        create_model_runner_output(reqs=[], finished_recving={owner.request_id}),
    )
    active = [r for r in (owner, follower) if not r.is_finished()]
    output = scheduler.schedule()
    assert set(output.num_scheduled_tokens) == {r.request_id for r in active}
    for request in active:
        assert manager.get_blocks(request.request_id).blocks[0][:2] == blocks
        assert output.num_scheduled_tokens[request.request_id] == 16
    assert all(block.ref_cnt == len(active) for block in blocks)
    assert not scheduler._shared_prefix_loads
    assert not scheduler._shared_load_followers
    _abort_scheduled_batch(scheduler, output, active)


@pytest.mark.parametrize("policy", ["recompute", "fail"])
@pytest.mark.parametrize("abort_owner", [False, True])
@pytest.mark.parametrize(
    "error_kind,early_error",
    [("blocks", False), ("blocks", True), ("request", False)],
)
def test_shared_external_prefix_load_failure(
    monkeypatch, policy, abort_owner, error_kind, early_error
):
    """Failed shared destinations cannot become cache hits or shared writes."""
    scheduler = _shared_load_scheduler(monkeypatch, policy)
    owner, follower = [
        create_request(num_tokens=48, common_prefix_len=32) for _ in range(2)
    ]
    for request in (owner, follower):
        scheduler.add_request(request)
    output = scheduler.schedule()
    blocks = list(scheduler.kv_cache_manager.get_blocks(owner.request_id).blocks[0])
    if abort_owner:
        scheduler.finish_requests(owner.request_id, RequestStatus.FINISHED_ABORTED)
    error = {blocks[1].block_id}
    if early_error:
        scheduler.update_from_output(
            output, create_model_runner_output(reqs=[], invalid_block_ids=error)
        )
        output = scheduler.schedule()
        assert not output.num_scheduled_tokens
    result = create_model_runner_output(
        reqs=[],
        finished_recving={owner.request_id},
        invalid_block_ids=error if error_kind == "blocks" and not early_error else None,
    )
    if error_kind == "request":
        result.kv_connector_output.failed_recving = {owner.request_id}
    scheduler.update_from_output(output, result)
    assert not scheduler._shared_prefix_loads
    assert not scheduler._shared_load_owners
    assert not scheduler._shared_load_followers
    assert all(block.ref_cnt == 0 for block in blocks)
    assert all(block.block_hash is None for block in blocks)
    assert not scheduler.kv_cache_manager.block_pool.cached_block_hash_to_block
    if policy == "fail":
        assert follower.status == RequestStatus.FINISHED_ERROR
        scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)
        scheduler.schedule()
        assert_scheduler_empty(scheduler)
    else:
        assert follower.num_computed_tokens == 0
        assert not scheduler.kv_cache_manager.get_block_ids(follower.request_id)[0]
        assert all(block.ref_cnt == 0 for block in blocks)
        assert all(block.block_hash is None for block in blocks)
        output = scheduler.schedule()
        assert follower.status == RequestStatus.RUNNING
        assert scheduler.connector.update_state_after_alloc.call_count >= 2
        assert all(
            call.args[2] == 0
            for call in scheduler.connector.update_state_after_alloc.call_args_list[1:]
        )
        active = [
            r for r in (owner, follower) if r.request_id in output.num_scheduled_tokens
        ]
        _abort_scheduled_batch(scheduler, output, active)


@pytest.mark.parametrize("num_tokens,matched_tokens", [(48, 31), (48, 48), (49, 49)])
def test_shared_external_prefix_excludes_mutable_tail(
    monkeypatch, num_tokens, matched_tokens
):
    """Full and unaligned loads share full blocks but never the writable tail."""
    scheduler = _shared_load_scheduler(monkeypatch, matched_tokens=matched_tokens)
    monkeypatch.setattr(
        scheduler.connector, "supports_shared_prefix_load_slicing", lambda: True
    )
    owner, follower = [
        create_request(num_tokens=num_tokens, common_prefix_len=num_tokens)
        for _ in range(2)
    ]
    for request in (owner, follower):
        scheduler.add_request(request)
    output = scheduler.schedule()
    manager = scheduler.kv_cache_manager
    shared_tokens = min(matched_tokens, num_tokens - 1) // 16 * 16
    shared_blocks = manager.get_block_ids(follower.request_id)[0]
    assert len(shared_blocks) == shared_tokens // 16
    assert manager.get_block_ids(owner.request_id)[0][: len(shared_blocks)] == (
        shared_blocks
    )
    assert scheduler.connector.update_state_after_alloc.call_count == 1
    scheduler.update_from_output(
        output,
        create_model_runner_output(reqs=[], finished_recving={owner.request_id}),
    )
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {
        owner.request_id: max(1, num_tokens - matched_tokens),
        follower.request_id: num_tokens - shared_tokens,
    }
    # Neither writer may reuse the other's tail, even after load completion.
    assert set(
        manager.get_block_ids(owner.request_id)[0][len(shared_blocks) :]
    ).isdisjoint(manager.get_block_ids(follower.request_id)[0][len(shared_blocks) :])


@pytest.mark.parametrize("matched_tokens", [31, 48])
def test_shared_external_prefix_slicing_requires_opt_in(monkeypatch, matched_tokens):
    scheduler = _shared_load_scheduler(monkeypatch, matched_tokens=matched_tokens)
    for _ in range(2):
        scheduler.add_request(create_request(num_tokens=48, common_prefix_len=48))
    scheduler.schedule()
    assert scheduler.connector.update_state_after_alloc.call_count == 2
    assert not scheduler._shared_prefix_loads


def test_shared_external_prefix_checks_connector_compatibility(monkeypatch):
    scheduler = _shared_load_scheduler(monkeypatch)
    connector = scheduler.connector
    connector.is_shared_prefix_load_compatible = Mock(return_value=False)
    connector.on_shared_prefix_load = Mock()
    for _ in range(2):
        scheduler.add_request(create_request(num_tokens=48, common_prefix_len=32))
    scheduler.schedule()
    connector.is_shared_prefix_load_compatible.assert_called_once()
    connector.on_shared_prefix_load.assert_not_called()
    assert connector.update_state_after_alloc.call_count == 2


def _nixl_shared_requests(num_tokens):
    requests = [
        create_request(
            num_tokens=num_tokens,
            common_prefix_len=num_tokens,
            do_remote_prefill=True,
        )
        for _ in range(2)
    ]
    for i, request in enumerate(requests):
        request.kv_transfer_params.update(
            remote_block_ids=[list(range(1, (num_tokens - 1) // 16 + 1)) + [10 + i]],
            remote_num_tokens=num_tokens,
            pcp_size=1,
            remote_block_size=16,
            transfer_mode="pull",
        )
    return requests


@pytest.mark.parametrize(
    "unsupported",
    [
        "disabled",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
        "host_staging",
        "bidirectional",
        "remote_pcp",
        "remote_pcp_unknown",
    ],
)
def test_nixl_unsupported_config_uses_independent_loads(monkeypatch, unsupported):
    """Unsupported sharing keeps independent destinations and producer leases."""
    config = create_vllm_config(
        kv_connector_extra_config={
            "enable_shared_prefix_loads": unsupported != "disabled"
        }
    )
    scheduler = create_scheduler(config, num_blocks=32)
    nixl_scheduler = scheduler.connector.connector_scheduler
    # Exercise admission without initializing distributed groups or host buffers.
    if unsupported.endswith("parallel_size"):
        monkeypatch.setattr(config.parallel_config, unsupported, 2)
    elif unsupported == "host_staging":
        monkeypatch.setattr(nixl_scheduler, "use_host_buffer", True)
    elif unsupported == "bidirectional":
        monkeypatch.setattr(nixl_scheduler, "is_bidirectional_kv_xfer_enabled", True)
    remote_config = unsupported.startswith("remote_")
    if not remote_config:
        assert scheduler.connector.supports_shared_prefix_loads() is False
    owner, follower = _nixl_shared_requests(49)
    if remote_config:
        for request in (owner, follower):
            if unsupported == "remote_pcp":
                request.kv_transfer_params["pcp_size"] = 2
            else:
                request.kv_transfer_params.pop("pcp_size")
    for request in (owner, follower):
        scheduler.add_request(request)
    output = scheduler.schedule()
    assert len(output.kv_connector_metadata.reqs_to_recv) == 2
    assert all(
        entry.awaiting_kvs and entry.local_block_ids
        for entry in output.kv_connector_metadata.reqs_to_recv.values()
    )
    manager = scheduler.kv_cache_manager
    assert set(manager.get_block_ids(owner.request_id)[0]).isdisjoint(
        manager.get_block_ids(follower.request_id)[0]
    )
    if not remote_config:
        assert not scheduler._shared_prefix_loads
    assert not scheduler._shared_load_followers
    assert set(nixl_scheduler._heartbeat_req_engine) == {
        owner.request_id,
        follower.request_id,
    }
    assert nixl_scheduler._reqs_recving == {owner.request_id, follower.request_id}
    assert {
        entry.remote.request_id
        for entry in output.kv_connector_metadata.reqs_to_recv.values()
    } == {
        request.kv_transfer_params["remote_request_id"] for request in (owner, follower)
    }


@pytest.mark.parametrize("num_tokens", [48, 49])
@pytest.mark.parametrize("cancel", [None, "owner", "follower", "both"])
@pytest.mark.parametrize("local_tokens", [0, 16])
def test_nixl_shared_full_prompt_lease_lifecycle(num_tokens, cancel, local_tokens):
    """One real load, one unused-lease notification, and abort-safe heartbeats."""
    config = create_vllm_config(
        kv_connector_extra_config={"enable_shared_prefix_loads": True}
    )
    scheduler = create_scheduler(config, num_blocks=32)
    manager = scheduler.kv_cache_manager
    if local_tokens:
        warm = create_request(num_tokens=num_tokens, common_prefix_len=num_tokens)
        manager.allocate_slots(warm, local_tokens)
        manager.free(warm)
    owner, follower = _nixl_shared_requests(num_tokens)
    for request in (owner, follower):
        scheduler.add_request(request)
    output = scheduler.schedule()
    meta = output.kv_connector_metadata
    assert set(meta.reqs_to_recv) == {owner.request_id, follower.request_id}
    assert meta.reqs_to_recv[owner.request_id].awaiting_kvs
    assert meta.reqs_to_recv[owner.request_id].local_block_ids
    assert meta.reqs_to_recv[owner.request_id].local_num_computed_blocks == (
        local_tokens // 16,
    )
    assert len(meta.reqs_to_recv[owner.request_id].local_block_ids[0]) == (
        (num_tokens + 15) // 16 - local_tokens // 16
    )
    assert not meta.reqs_to_recv[follower.request_id].awaiting_kvs
    assert not meta.reqs_to_recv[follower.request_id].local_block_ids
    nixl_scheduler = scheduler.connector.connector_scheduler
    assert set(nixl_scheduler._heartbeat_req_engine) == {owner.request_id}
    assert nixl_scheduler._reqs_recving == {owner.request_id}
    prefix = list(manager.get_blocks(follower.request_id).blocks[0])
    if cancel in ("owner", "both"):
        scheduler.finish_requests(owner.request_id, RequestStatus.FINISHED_ABORTED)
    if cancel in ("follower", "both"):
        scheduler.finish_requests(follower.request_id, RequestStatus.FINISHED_ABORTED)
    assert set(nixl_scheduler._heartbeat_req_engine) == {owner.request_id}
    assert all(block.ref_cnt > 0 for block in prefix)
    # Cancellation must not enqueue a second release for either producer.
    assert not nixl_scheduler._reqs_need_recv
    scheduler.update_from_output(
        output,
        create_model_runner_output(reqs=[], finished_recving={owner.request_id}),
    )
    assert not nixl_scheduler._heartbeat_req_engine
    assert not nixl_scheduler._reqs_recving
    output = scheduler.schedule()
    active = [r for r in (owner, follower) if not r.is_finished()]
    assert set(output.num_scheduled_tokens) == {r.request_id for r in active}
    assert all(block.ref_cnt == len(active) for block in prefix)
    _abort_scheduled_batch(scheduler, output, active)


@pytest.mark.parametrize("sharing", [False, True])
@pytest.mark.parametrize("abort_owner", [False, True])
def test_nixl_rejection_cleanup_preserves_an_active_receive_lease(sharing, abort_owner):
    """HTTP cancellation cleanup cannot release a still-pending READ's lease."""
    config = create_vllm_config(
        kv_connector_extra_config={"enable_shared_prefix_loads": sharing}
    )
    scheduler = create_scheduler(config, num_blocks=32)
    owner, follower = _nixl_shared_requests(49)
    rejected = create_request(num_tokens=1)
    rejected.kv_transfer_params = copy.deepcopy(owner.kv_transfer_params)
    unused = create_request(num_tokens=1, do_remote_prefill=True)
    for request in (owner, follower):
        scheduler.add_request(request)
    output = scheduler.schedule()
    awaiting = {
        req_id
        for req_id, entry in output.kv_connector_metadata.reqs_to_recv.items()
        if entry.awaiting_kvs
    }
    blocks = list(scheduler.kv_cache_manager.get_blocks(owner.request_id).blocks[0])
    if abort_owner:
        scheduler.finish_requests(owner.request_id, RequestStatus.FINISHED_ABORTED)
    for request in (rejected, unused):
        scheduler.add_request(request)
        scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    scheduler.update_from_output(output, create_model_runner_output(reqs=[]))
    output = scheduler.schedule()
    receives = output.kv_connector_metadata.reqs_to_recv
    assert set(receives) == {unused.request_id}
    assert not receives[unused.request_id].awaiting_kvs
    assert not receives[unused.request_id].local_block_ids
    nixl_scheduler = scheduler.connector.connector_scheduler
    assert owner.request_id in nixl_scheduler._heartbeat_req_engine
    assert all(block.ref_cnt > 0 for block in blocks)
    scheduler.update_from_output(
        output, create_model_runner_output(reqs=[], finished_recving=awaiting)
    )
    assert not nixl_scheduler._heartbeat_req_engine
    assert not nixl_scheduler._reqs_recving
    output = scheduler.schedule()
    active = [r for r in (owner, follower) if not r.is_finished()]
    _abort_scheduled_batch(scheduler, output, active)


@pytest.mark.parametrize("abort_owner", [False, True])
def test_nixl_failed_shared_load_recomputes_without_reusing_released_lease(abort_owner):
    config = create_vllm_config(
        kv_connector_extra_config={"enable_shared_prefix_loads": True},
        kv_load_failure_policy="recompute",
        max_num_batched_tokens=128,
    )
    scheduler = create_scheduler(config, num_blocks=32)
    owner, follower = _nixl_shared_requests(49)
    for request in (owner, follower):
        scheduler.add_request(request)
    output = scheduler.schedule()
    manager = scheduler.kv_cache_manager
    failed_blocks = list(manager.get_blocks(owner.request_id).blocks[0])
    if abort_owner:
        scheduler.finish_requests(owner.request_id, RequestStatus.FINISHED_ABORTED)
    result = create_model_runner_output(reqs=[], finished_recving={owner.request_id})
    result.kv_connector_output.failed_recving = {owner.request_id}
    scheduler.update_from_output(output, result)
    active = [r for r in (owner, follower) if not r.is_finished()]
    assert all(not manager.get_block_ids(r.request_id)[0] for r in active)
    assert all(r.num_computed_tokens == 0 for r in active)
    assert all(
        block.ref_cnt == 0 and block.block_hash is None for block in failed_blocks
    )
    assert not manager.block_pool.cached_block_hash_to_block
    assert not scheduler._shared_prefix_loads
    assert not scheduler._shared_load_owners
    assert not scheduler._shared_load_followers
    with patch.object(
        scheduler.connector,
        "update_state_after_alloc",
        wraps=scheduler.connector.update_state_after_alloc,
    ) as allocations:
        output = scheduler.schedule()
    assert allocations.call_count == len(active)
    assert all(call.args[2] == 0 for call in allocations.call_args_list)
    computed = {r.req_id: r.num_computed_tokens for r in output.scheduled_new_reqs}
    if abort_owner:
        assert output.num_scheduled_tokens == {follower.request_id: 49}
        assert computed == {follower.request_id: 0}
    else:
        # The owner's new local prefill publishes full blocks within this batch.
        # The follower may reuse that prefix, never the failed remote contents.
        assert output.num_scheduled_tokens == {
            owner.request_id: 49,
            follower.request_id: 1,
        }
        assert computed == {owner.request_id: 0, follower.request_id: 48}
        owner_blocks = manager.get_block_ids(owner.request_id)[0]
        follower_blocks = manager.get_block_ids(follower.request_id)[0]
        assert owner_blocks[:3] == follower_blocks[:3]
        assert owner_blocks[3] != follower_blocks[3]
    assert not output.kv_connector_metadata.reqs_to_recv
    assert not scheduler.connector.connector_scheduler._heartbeat_req_engine
    _abort_scheduled_batch(scheduler, output, active)


def test_nixl_late_follower_can_use_an_aborted_owners_live_read():
    config = create_vllm_config(
        kv_connector_extra_config={"enable_shared_prefix_loads": True}
    )
    scheduler = create_scheduler(config, num_blocks=32)
    owner, follower = _nixl_shared_requests(49)
    scheduler.add_request(owner)
    output = scheduler.schedule()
    scheduler.update_from_output(output, create_model_runner_output(reqs=[]))
    scheduler.finish_requests(owner.request_id, RequestStatus.FINISHED_ABORTED)
    scheduler.add_request(follower)
    output = scheduler.schedule()
    recv = output.kv_connector_metadata.reqs_to_recv[follower.request_id]
    assert not recv.awaiting_kvs and not recv.local_block_ids
    assert set(scheduler.connector.connector_scheduler._heartbeat_req_engine) == {
        owner.request_id
    }
    scheduler.update_from_output(
        output,
        create_model_runner_output(reqs=[], finished_recving={owner.request_id}),
    )
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {follower.request_id: 1}


@pytest.mark.parametrize("isolation", ["salt", "tokens", "skip", "opt_out"])
def test_shared_external_prefix_preserves_isolation(monkeypatch, isolation):
    scheduler = _shared_load_scheduler(monkeypatch)
    owner = create_request(num_tokens=48, common_prefix_len=32)
    follower = create_request(
        num_tokens=48, common_prefix_len=0 if isolation == "tokens" else 32
    )
    if isolation == "salt":
        follower.cache_salt = "another-tenant"
        follower.block_hashes.clear()
        follower.update_block_hashes()
    elif isolation == "skip":
        follower.skip_reading_prefix_cache = True
    elif isolation == "opt_out":
        monkeypatch.setattr(
            scheduler.connector, "supports_shared_prefix_loads", lambda: False
        )
    for request in (owner, follower):
        scheduler.add_request(request)
    scheduler.schedule()
    manager = scheduler.kv_cache_manager
    assert set(manager.get_block_ids(owner.request_id)[0]).isdisjoint(
        manager.get_block_ids(follower.request_id)[0]
    )
    assert scheduler.connector.update_state_after_alloc.call_count == 2


def test_shared_external_prefix_late_follower_after_owner_abort(monkeypatch):
    scheduler = _shared_load_scheduler(monkeypatch)
    owner = create_request(num_tokens=48, common_prefix_len=32)
    scheduler.add_request(owner)
    output = scheduler.schedule()
    scheduler.update_from_output(output, create_model_runner_output(reqs=[]))
    blocks = list(scheduler.kv_cache_manager.get_blocks(owner.request_id).blocks[0])
    scheduler.finish_requests(owner.request_id, RequestStatus.FINISHED_ABORTED)
    follower = create_request(num_tokens=48, common_prefix_len=32)
    scheduler.add_request(follower)
    output = scheduler.schedule()
    assert scheduler.connector.update_state_after_alloc.call_count == 1
    assert all(block.ref_cnt == 2 for block in blocks)
    scheduler.update_from_output(
        output,
        create_model_runner_output(reqs=[], finished_recving={owner.request_id}),
    )
    output = scheduler.schedule()
    assert output.num_scheduled_tokens[follower.request_id] == 16
    assert all(block.ref_cnt == 1 for block in blocks)


def test_shared_external_prefix_extends_local_hit(monkeypatch):
    """A locally cached leading block remains shared with the loaded suffix."""
    scheduler = _shared_load_scheduler(monkeypatch)
    monkeypatch.setattr(
        scheduler.connector,
        "get_num_new_matched_tokens",
        lambda request, local_tokens: (32 - local_tokens, True),
    )
    manager = scheduler.kv_cache_manager
    warm = create_request(num_tokens=48, common_prefix_len=32)
    manager.allocate_slots(warm, 16)
    manager.free(warm)
    owner, follower = [
        create_request(num_tokens=48, common_prefix_len=32) for _ in range(2)
    ]
    for request in (owner, follower):
        scheduler.add_request(request)
    output = scheduler.schedule()
    blocks = list(manager.get_blocks(owner.request_id).blocks[0])
    assert manager.get_blocks(follower.request_id).blocks[0] == blocks
    assert blocks[0].block_hash is not None
    assert blocks[1].block_hash is None
    assert all(block.ref_cnt == 2 for block in blocks)
    scheduler.connector.update_state_after_alloc.assert_called_once()
    assert scheduler.connector.update_state_after_alloc.call_args.args[2] == 16
    scheduler.update_from_output(
        output,
        create_model_runner_output(reqs=[], finished_recving={owner.request_id}),
    )
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {
        owner.request_id: 16,
        follower.request_id: 16,
    }
    assert all(block.block_hash is not None for block in blocks)


def test_basic_lifecycle():
    """Test lifecycle of a remote prefill."""
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)

    # 2 Full Blocks and 1 Half Block.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_EXTERNAL_FULL_BLOCKS = 2
    NUM_TOKENS = int(BLOCK_SIZE * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))
    START_FREE_BLOCK_QUEUE_SIZE = (
        scheduler.kv_cache_manager.block_pool.free_block_queue.num_free_blocks
    )

    request = create_request(
        request_id=1,
        block_size=BLOCK_SIZE,
        num_tokens=NUM_TOKENS,
        do_remote_prefill=True,
    )

    scheduler.add_request(request)
    request_id = request.request_id

    # STEP (1):
    # (1a): schedule()
    scheduler_output = scheduler.schedule()

    # Nothing running and empty scheduler output.
    assert len(scheduler.running) == 0
    assert len(scheduler_output.scheduled_new_reqs) == 0
    assert scheduler_output.scheduled_cached_reqs.num_reqs == 0
    assert len(scheduler_output.num_scheduled_tokens) == 0
    assert scheduler_output.total_num_scheduled_tokens == 0

    # Req waiting for KVs with no computed/scheduled toks ...
    assert _num_waiting_requests(scheduler) == 1
    assert request in scheduler.skipped_waiting
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert request.num_computed_tokens == NUM_TOKENS

    # ... but should have (uncached) blocks allocated to it.
    block_pool = scheduler.kv_cache_manager.block_pool
    assert block_pool.free_block_queue.num_free_blocks < START_FREE_BLOCK_QUEUE_SIZE
    assert len(block_pool.cached_block_hash_to_block) == 0
    blocks = scheduler.kv_cache_manager.coordinator.single_type_managers[
        0
    ].req_to_blocks[request_id]
    for block in blocks:
        assert block._block_hash is None

    # (1b): forward()
    model_runner_output = EMPTY_MODEL_RUNNER_OUTPUT

    # (1c): update_from_output()
    engine_core_outputs = scheduler.update_from_output(
        scheduler_output, model_runner_output
    )
    assert not engine_core_outputs or not engine_core_outputs[0].outputs

    # STEP (2):
    # (2a): schedule(): nothing happens!
    scheduler_output = scheduler.schedule()
    assert _num_waiting_requests(scheduler) == 1
    assert len(scheduler.running) == 0

    # (2b): forward(): request finishes recv.
    model_runner_output = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    model_runner_output.kv_connector_output = KVConnectorOutput(
        finished_recving={request_id}
    )

    # (2c): update_from_output():
    engine_core_outputs = scheduler.update_from_output(
        scheduler_output, model_runner_output
    )
    assert _num_waiting_requests(scheduler) == 1
    assert request_id in scheduler.finished_recving_kv_req_ids

    # STEP (3):
    # (3a): schedule(): this should actually schedule.
    scheduler_output = scheduler.schedule()
    assert len(scheduler.running) == 1

    # Confirm the block are actually allocated.
    num_hashed_blocks = 0
    blocks = scheduler.kv_cache_manager.coordinator.single_type_managers[
        0
    ].req_to_blocks[request_id]
    for block in blocks:
        assert block.ref_cnt == 1
        num_hashed_blocks += 1 if block._block_hash is not None else 0
    assert num_hashed_blocks == NUM_EXTERNAL_FULL_BLOCKS

    # Confirm the rest of the prompt is scheduled in this step.
    scheduled_req = scheduler_output.scheduled_new_reqs[0]
    num_scheduled_tokens = scheduler_output.num_scheduled_tokens[request_id]
    num_computed_tokens = scheduled_req.num_computed_tokens
    total_prompt_tokens = len(scheduled_req.prompt_token_ids)
    assert num_scheduled_tokens == total_prompt_tokens - num_computed_tokens

    # (3b): execute_model()
    model_runner_output = create_model_runner_output([request])
    # (3c): update_from_output()
    scheduler.update_from_output(scheduler_output, model_runner_output)

    # Step (4): Hit EOS.
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output([request], use_eos=True)
    engine_core_outputs = scheduler.update_from_output(
        scheduler_output, model_runner_output
    )
    scheduler.schedule()

    outputs = engine_core_outputs[0].outputs
    assert len(outputs) == 1
    output = outputs[0]
    assert output.finish_reason == FinishReason.STOP
    assert_scheduler_empty(scheduler)


def test_interleaved_lifecycle():
    """Test Remote Prefills Work Well With Other Requests."""
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)

    # 2 Full Blocks and 1 Half Block.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_EXTERNAL_FULL_BLOCKS = 2
    NUM_TOKENS = int(BLOCK_SIZE * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))

    request_remote = create_request(
        request_id=1,
        block_size=BLOCK_SIZE,
        num_tokens=NUM_TOKENS,
        do_remote_prefill=True,
    )
    request_local_a = create_request(
        request_id=2,
        block_size=BLOCK_SIZE,
        num_tokens=NUM_TOKENS,
    )
    request_local_b = create_request(
        request_id=3,
        block_size=BLOCK_SIZE,
        num_tokens=NUM_TOKENS,
    )

    # STEP 1: Regular request is running.
    scheduler.add_request(request_local_a)
    scheduler_output = scheduler.schedule()
    assert len(scheduler.running) == 1

    model_runner_output = create_model_runner_output([request_local_a])
    scheduler.update_from_output(scheduler_output, model_runner_output)

    # STEP 2: Add a local and remote request.
    scheduler.add_request(request_local_b)
    scheduler.add_request(request_remote)
    scheduler_output = scheduler.schedule()
    assert len(scheduler.running) == 2
    assert _num_waiting_requests(scheduler) == 1
    assert len(scheduler_output.scheduled_new_reqs) == 1
    assert scheduler_output.scheduled_cached_reqs.num_reqs == 1

    model_runner_output = create_model_runner_output([request_local_a, request_local_b])
    scheduler.update_from_output(scheduler_output, model_runner_output)

    # STEP 3: continue running, KVs not arrived yet.
    scheduler_output = scheduler.schedule()
    assert len(scheduler.running) == 2
    assert _num_waiting_requests(scheduler) == 1
    assert len(scheduler_output.scheduled_new_reqs) == 0
    assert scheduler_output.scheduled_cached_reqs.num_reqs == 2

    model_runner_output = create_model_runner_output(
        reqs=[request_local_a, request_local_b]
    )
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert len(scheduler.running) == 2
    assert _num_waiting_requests(scheduler) == 1
    assert len(scheduler_output.scheduled_new_reqs) == 0
    assert scheduler_output.scheduled_cached_reqs.num_reqs == 2

    # STEP 4: KVs arrive.
    scheduler_output = scheduler.schedule()
    assert len(scheduler.running) == 2
    assert _num_waiting_requests(scheduler) == 1
    assert len(scheduler_output.scheduled_new_reqs) == 0
    assert scheduler_output.scheduled_cached_reqs.num_reqs == 2

    model_runner_output = create_model_runner_output(
        [request_local_a, request_local_b], finished_recving={request_remote.request_id}
    )
    scheduler.update_from_output(scheduler_output, model_runner_output)

    # STEP 5: RECVed KVs are sent to ModelRunner.
    scheduler_output = scheduler.schedule()
    assert len(scheduler.running) == 3
    assert _num_waiting_requests(scheduler) == 0
    assert len(scheduler_output.scheduled_new_reqs) == 1
    assert scheduler_output.scheduled_cached_reqs.num_reqs == 2

    model_runner_output = create_model_runner_output(
        [request_local_a, request_local_b, request_remote]
    )
    scheduler.update_from_output(scheduler_output, model_runner_output)

    # STEP 6: Hit EOS and free.
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(
        [request_local_a, request_local_b, request_remote],
        use_eos=True,
    )
    scheduler.update_from_output(scheduler_output, model_runner_output)
    scheduler.schedule()
    assert_scheduler_empty(scheduler)


def test_no_spurious_prefix_caching():
    """With P/D, blocks can be allocated but uncomputed for
    multiple engine steps. This test confirms that we do
    not accidentally have cache hits against uncomputed
    blocks.
    """
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)

    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)

    # 2 and a half full external blocks.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_EXTERNAL_FULL_BLOCKS = 2
    NUM_TOKENS = int(BLOCK_SIZE * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))

    # Both of these requests have prompts like [1,1,1,1,1, ...]
    request_remote = create_request(
        request_id=1,
        block_size=BLOCK_SIZE,
        num_tokens=NUM_TOKENS,
        common_prefix_len=NUM_TOKENS,
        do_remote_prefill=True,
    )

    request_local = create_request(
        request_id=2,
        block_size=BLOCK_SIZE,
        num_tokens=NUM_TOKENS,
        common_prefix_len=NUM_TOKENS,
        do_remote_prefill=False,
    )

    # Schedule the remote prefill request. This should not
    # cause any blocks to be cached.
    scheduler.add_request(request_remote)
    scheduler_output = scheduler.schedule()
    scheduler.update_from_output(scheduler_output, EMPTY_MODEL_RUNNER_OUTPUT)
    assert _num_waiting_requests(scheduler) == 1

    # Schedule the local prefill request. This should
    # cause blocks to be cached, but separately from
    scheduler.add_request(request_local)
    scheduler_output = scheduler.schedule()
    assert len(scheduler.running) == 1
    assert _num_waiting_requests(scheduler) == 1

    local_blocks = scheduler.kv_cache_manager.coordinator.single_type_managers[
        0
    ].req_to_blocks[request_local.request_id]
    remote_blocks = scheduler.kv_cache_manager.coordinator.single_type_managers[
        0
    ].req_to_blocks[request_remote.request_id]

    # Local should have cached blocks (but not all due to preallocate).
    num_hashed_blocks = 0
    for block in local_blocks:
        assert block.ref_cnt == 1
        num_hashed_blocks += 1 if block._block_hash is not None else 0
    assert num_hashed_blocks > 0

    # Remote blocks should not be cached.
    for block in remote_blocks:
        assert block.ref_cnt == 1
        assert block._block_hash is None


def test_full_block_prompt():
    """Test that we handle a prompt that is the full block size."""
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)

    # 2 Full Blocks and 1 Half Block.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_EXTERNAL_FULL_BLOCKS = 2
    NUM_TOKENS = int(BLOCK_SIZE * NUM_EXTERNAL_FULL_BLOCKS)

    request = create_request(
        request_id=1,
        block_size=BLOCK_SIZE,
        num_tokens=NUM_TOKENS,
        do_remote_prefill=True,
    )

    scheduler.add_request(request)
    request_id = request.request_id

    # STEP (1): Initialize a recv.
    scheduler_output = scheduler.schedule()
    # All blocks should be allocated.
    num_blocks = len(
        scheduler.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks[
            request_id
        ]
    )
    assert num_blocks == NUM_EXTERNAL_FULL_BLOCKS
    model_runner_output = EMPTY_MODEL_RUNNER_OUTPUT
    scheduler.update_from_output(scheduler_output, model_runner_output)

    # # STEP (2): Recv.
    scheduler_output = scheduler.schedule()
    model_runner_output = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    model_runner_output.kv_connector_output = KVConnectorOutput(
        finished_recving={request_id}
    )
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert _num_waiting_requests(scheduler) == 1
    assert request_id in scheduler.finished_recving_kv_req_ids

    # # STEP (3): Run as usual.
    scheduler_output = scheduler.schedule()

    # We need to recompute the final token of the prompt to generate
    # the first new token, so we should not have a new block.
    num_blocks = len(
        scheduler.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks[
            request_id
        ]
    )
    assert num_blocks == NUM_EXTERNAL_FULL_BLOCKS
    assert scheduler_output.scheduled_new_reqs[0].num_computed_tokens == NUM_TOKENS - 1
    assert scheduler_output.num_scheduled_tokens[request_id] == 1

    model_runner_output = create_model_runner_output([request])
    scheduler.update_from_output(scheduler_output, model_runner_output)

    # # Step (4): Hit EOS.
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output([request], use_eos=True)
    engine_core_outputs = scheduler.update_from_output(
        scheduler_output, model_runner_output
    )
    scheduler.schedule()

    outputs = engine_core_outputs[0].outputs
    assert len(outputs) == 1
    output = outputs[0]
    assert output.finish_reason == FinishReason.STOP
    assert_scheduler_empty(scheduler)


def test_cannot_schedule_after_recv():
    """Test that we can handle no schedule after recv due to not
    enough remaining KV blocks.
    """
    # NOTE: the KVCacheManager will use 1 null block.
    # So there are 5 total working blocks.
    TOTAL_NUM_BLOCKS = 6
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config, num_blocks=TOTAL_NUM_BLOCKS)

    # Prime the KVCache.
    NUM_PROMPT_BLOCKS = 2
    BLOCK_SIZE = vllm_config.cache_config.block_size
    # Prompt will use 2 blocks + 1 block after we schedule.
    NUM_TOKENS_LOCAL = int(BLOCK_SIZE * NUM_PROMPT_BLOCKS)
    NUM_TOKENS_REMOTE = int(BLOCK_SIZE * NUM_PROMPT_BLOCKS)

    request_normal = create_request(
        request_id=1, block_size=BLOCK_SIZE, num_tokens=NUM_TOKENS_LOCAL
    )
    request_remote = create_request(
        request_id=2,
        block_size=BLOCK_SIZE,
        num_tokens=NUM_TOKENS_REMOTE,
        do_remote_prefill=True,
    )

    # STEP 1: 3 blocks are in use (2 for prompt, 1 for decode).
    scheduler.add_request(request_normal)
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(reqs=[request_normal])
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert len(scheduler.running) == 1
    assert _num_waiting_requests(scheduler) == 0

    # Step 2: 5 blocks are in use (2 new for remote blocks).
    scheduler.add_request(request_remote)
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(reqs=[request_normal])
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert len(scheduler.running) == 1
    assert _num_waiting_requests(scheduler) == 1

    # Step 3: finish recving (5 blocks in use)
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(
        reqs=[request_normal], finished_recving={request_remote.request_id}
    )
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert len(scheduler.running) == 1
    assert _num_waiting_requests(scheduler) == 1

    # Step 4: try to schedule, remote request is put to running list
    # because the transfer is completed.
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(
        reqs=[request_normal, request_remote]
    )
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert len(scheduler.running) == 2
    assert _num_waiting_requests(scheduler) == 0

    # Step 5: Remote request will be put back to waiting list
    # because it needs new block to hold generated token.
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(reqs=[request_normal])
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert len(scheduler.running) == 1
    assert _num_waiting_requests(scheduler) == 1

    # Step 6: finish the request, free it.
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(
        reqs=[request_normal], use_eos=True
    )
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert len(scheduler.running) == 0
    assert _num_waiting_requests(scheduler) == 1

    # Step 7: now we can schedule (with 2 blocks computed),
    # request is retrieved from preempted list.
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(reqs=[request_remote])
    # V2 emits a resumed (previously preempted) request as a NewRequestData
    # rather than a cached request.
    if scheduler.use_v2_model_runner:
        num_computed = scheduler_output.scheduled_new_reqs[0].num_computed_tokens
    else:
        cached = scheduler_output.scheduled_cached_reqs
        num_computed = cached.num_computed_tokens[0]
    assert num_computed == NUM_PROMPT_BLOCKS * BLOCK_SIZE
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert len(scheduler.running) == 1
    assert _num_waiting_requests(scheduler) == 0

    # Step 8: free everything.
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(
        reqs=[request_remote], use_eos=True
    )
    scheduler.update_from_output(scheduler_output, model_runner_output)
    _ = scheduler.schedule()
    assert_scheduler_empty(scheduler)


def test_cannot_recv():
    """Test that we can handle no schedule KV block transfer due to not
    enough remaining KV blocks.
    """
    # NOTE: the KVCacheManager will use 1 null block.
    # So there are 5 total working blocks.
    TOTAL_NUM_BLOCKS = 6
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config, num_blocks=TOTAL_NUM_BLOCKS)

    # Prime the KVCache.
    NUM_PROMPT_BLOCKS = 2
    BLOCK_SIZE = vllm_config.cache_config.block_size
    # Prompt will use 2 blocks + 1 block after we schedule.
    NUM_TOKENS_LOCAL = int(BLOCK_SIZE * NUM_PROMPT_BLOCKS)
    NUM_TOKENS_REMOTE = int(BLOCK_SIZE * (NUM_PROMPT_BLOCKS + 0.5))

    request_normal = create_request(
        request_id=1, block_size=BLOCK_SIZE, num_tokens=NUM_TOKENS_LOCAL
    )
    request_remote = create_request(
        request_id=2,
        block_size=BLOCK_SIZE,
        num_tokens=NUM_TOKENS_REMOTE,
        do_remote_prefill=True,
    )

    # STEP 1: 3 blocks are in use (2 for prompt, 1 for decode).
    scheduler.add_request(request_normal)
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(reqs=[request_normal])
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert len(scheduler.running) == 1
    assert _num_waiting_requests(scheduler) == 0

    # Step 2: 3 blocks are in use,
    # need 3 new for remote blocks but only 2 are available.
    scheduler.add_request(request_remote)
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(reqs=[request_normal])
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert len(scheduler.running) == 1
    assert _num_waiting_requests(scheduler) == 1
    # Should not have KV transfer in progress.
    assert request_remote.status != RequestStatus.WAITING_FOR_REMOTE_KVS

    # Step 3: finish the request, free it.
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(
        reqs=[request_normal], use_eos=True
    )
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert len(scheduler.running) == 0
    assert _num_waiting_requests(scheduler) == 1

    # Step 4: now we can initiate KV transfer (with 2 blocks computed).
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(reqs=[])
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert len(scheduler.running) == 0
    assert _num_waiting_requests(scheduler) == 1
    assert request_remote.status == RequestStatus.WAITING_FOR_REMOTE_KVS

    # Step 5: finish recving (5 blocks in use)
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(
        reqs=[], finished_recving={request_remote.request_id}
    )
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert len(scheduler.running) == 0
    assert _num_waiting_requests(scheduler) == 1

    # Step 6: schedule remote request
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(reqs=[request_remote])
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert len(scheduler.running) == 1
    assert _num_waiting_requests(scheduler) == 0

    # Step 7: free everything.
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(
        reqs=[request_remote], use_eos=True
    )
    scheduler.update_from_output(scheduler_output, model_runner_output)
    _ = scheduler.schedule()
    assert_scheduler_empty(scheduler)


@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler.current_platform"
)
def test_p_side_chunked_prefill_mamba(mock_platform):
    """P-side integration: Mamba N-1 truncation + chunked prefill completes.

    A 64-token P-side request is truncated to 63 by the N-1 fix, then
    chunked into two prefill steps (32 + 31) and finishes with
    LENGTH_CAPPED because max_tokens is set to 1.
    """
    mock_platform.device_type = "cpu"

    BATCH_SIZE = 32
    NUM_TOKENS = 64
    BLOCK_SIZE = 16

    vllm_config = create_vllm_config(
        max_num_batched_tokens=BATCH_SIZE,
        block_size=BLOCK_SIZE,
    )
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = False

    kv_cache_config = make_kv_cache_config(
        block_size=BLOCK_SIZE,
        mamba_enabled=True,
        num_blocks=10000,
    )

    scheduler = create_scheduler(vllm_config, kv_cache_config=kv_cache_config)

    request = create_request(
        num_tokens=NUM_TOKENS,
        do_remote_decode=True,
        block_size=BLOCK_SIZE,
    )
    request.max_tokens = 128
    scheduler.add_request(request)
    request_id = request.request_id

    # ── Step 1: first chunk ──
    scheduler_output = scheduler.schedule()

    assert len(request.prompt_token_ids) == NUM_TOKENS - 1
    assert request.max_tokens == 1
    assert scheduler_output.num_scheduled_tokens[request_id] == BATCH_SIZE
    assert request.num_computed_tokens == BATCH_SIZE

    # Model returns no tokens for intermediate prefill chunk
    intermediate_output = ModelRunnerOutput(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[[]],
    )
    scheduler.update_from_output(scheduler_output, intermediate_output)

    # ── Step 2: remaining chunk ──
    scheduler_output = scheduler.schedule()

    remaining = NUM_TOKENS - 1 - BATCH_SIZE  # 31
    assert scheduler_output.num_scheduled_tokens[request_id] == remaining
    assert request.num_computed_tokens == NUM_TOKENS - 1

    # Prefill complete: model generates 1 decode token
    final_output = create_model_runner_output([request])
    engine_core_outputs = scheduler.update_from_output(scheduler_output, final_output)

    # max_tokens=1 → request finishes with LENGTH
    outputs = engine_core_outputs[0].outputs
    assert len(outputs) == 1
    assert outputs[0].finish_reason == FinishReason.LENGTH


def test_async_load_reserves_blocks_for_inflight():
    """A second async KV-connector load is not admitted if its initial
    allocation would consume blocks reserved for an already in-flight sequence.

    req_a gets a 1-block prefix (full sequence = 4 blocks), reserving 3 more.
    req_b would need a 4-block initial allocation, but only
    (free - req_a's 3-block reservation) = 3 blocks are available to it, so it is
    held back in WAITING (holding no blocks) rather than wedging req_a.
    """
    vllm_config = create_vllm_config()
    BLOCK_SIZE = vllm_config.cache_config.block_size
    scheduler = create_scheduler(vllm_config, num_blocks=8)  # usable = 7

    req_a = create_request(
        request_id=1,
        block_size=BLOCK_SIZE,
        num_tokens=BLOCK_SIZE * 4,
        do_remote_prefill=True,
        num_remote_blocks=1,
    )
    req_b = create_request(
        request_id=2,
        block_size=BLOCK_SIZE,
        num_tokens=BLOCK_SIZE * 5,
        do_remote_prefill=True,
        num_remote_blocks=1,
    )
    scheduler.add_request(req_a)
    scheduler.add_request(req_b)

    # Partial external matches: req_a loads 1 block, req_b loads 4 blocks.
    with patch.object(
        scheduler.connector,
        "get_num_new_matched_tokens",
        side_effect=[(BLOCK_SIZE, True), (BLOCK_SIZE * 4, True)],
    ):
        scheduler.schedule()

    assert req_a.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert req_b.status == RequestStatus.WAITING

    req_to_blocks = scheduler.kv_cache_manager.coordinator.single_type_managers[
        0
    ].req_to_blocks
    assert req_a.request_id in req_to_blocks
    assert req_b.request_id not in req_to_blocks


def test_async_loads_both_admitted_when_pool_fits():
    """Sanity: with a pool large enough, the reservation gate admits both async
    loads (it is not over-conservative)."""
    vllm_config = create_vllm_config()
    BLOCK_SIZE = vllm_config.cache_config.block_size
    scheduler = create_scheduler(vllm_config, num_blocks=64)

    reqs = [
        create_request(
            request_id=i,
            block_size=BLOCK_SIZE,
            num_tokens=BLOCK_SIZE * 5,
            do_remote_prefill=True,
            num_remote_blocks=1,
        )
        for i in (1, 2)
    ]
    for req in reqs:
        scheduler.add_request(req)

    with patch.object(
        scheduler.connector,
        "get_num_new_matched_tokens",
        side_effect=[(BLOCK_SIZE, True), (BLOCK_SIZE, True)],
    ):
        scheduler.schedule()

    for req in reqs:
        assert req.status == RequestStatus.WAITING_FOR_REMOTE_KVS


def test_async_load_reserves_blocks_for_promotion_margin():
    """An async load is not admitted unless the blocks its own promotion will
    need are still free.

    A parked load is allocated without lookahead slots, but promotion pads it
    to ``1 + num_spec_tokens`` and asks for the lookahead margin on top. Without
    reserving that margin, two loads can be admitted that together consume the
    whole pool; the head then fails ``allocate_slots`` forever while the load
    behind it is never reached, and with nothing running no block is ever freed.

    req_a (4 blocks) and req_b (3 blocks) exactly fill the 7 usable blocks, so
    req_b must be held back in WAITING holding no blocks, leaving req_a room to
    be promoted and run.
    """
    vllm_config = create_vllm_config(num_speculative_tokens=3)
    BLOCK_SIZE = vllm_config.cache_config.block_size
    scheduler = create_scheduler(vllm_config, num_blocks=8)  # usable = 7

    req_a = create_request(
        request_id=1,
        block_size=BLOCK_SIZE,
        num_tokens=BLOCK_SIZE * 4,
        do_remote_prefill=True,
        max_tokens=1,
    )
    req_b = create_request(
        request_id=2,
        block_size=BLOCK_SIZE,
        num_tokens=BLOCK_SIZE * 3,
        do_remote_prefill=True,
        max_tokens=1,
    )
    scheduler.add_request(req_a)
    scheduler.add_request(req_b)

    # Both get a full external hit, so each would hold its whole prompt.
    with patch.object(
        scheduler.connector,
        "get_num_new_matched_tokens",
        side_effect=[(BLOCK_SIZE * 4, True), (BLOCK_SIZE * 3, True)],
    ):
        scheduler.schedule()

    assert req_a.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert req_b.status == RequestStatus.WAITING
    req_to_blocks = scheduler.kv_cache_manager.coordinator.single_type_managers[
        0
    ].req_to_blocks
    assert req_b.request_id not in req_to_blocks

    # req_a's load lands: it must be promotable and actually get scheduled.
    scheduler.update_from_output(
        scheduler.schedule(),
        create_model_runner_output([], finished_recving={req_a.request_id}),
    )
    scheduler_output = scheduler.schedule()
    assert req_a.status == RequestStatus.RUNNING
    assert scheduler_output.num_scheduled_tokens[req_a.request_id] > 0
