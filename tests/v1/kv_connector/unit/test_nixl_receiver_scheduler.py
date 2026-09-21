# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission and lifetime fences for the optional background pull receiver.

The scheduler converts lookup/allocation/lifecycle events into bounded receiver
jobs and heartbeat snapshots. These CPU tests exercise that public contract with
small request/block stubs; they need neither a model nor a native NIXL agent.
"""

from types import SimpleNamespace

import pytest

from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import (
    NixlPullConnectorScheduler,
)
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import RequestStatus


def _scheduler(**extra):
    options = {"background_receiver": True, "background_receiver_max_pending": 1}
    options.update(extra)
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16, mamba_cache_mode="none"),
        parallel_config=SimpleNamespace(
            data_parallel_index=0,
            tensor_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        num_prefill_lookahead_tokens=0,
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=True),
        kv_transfer_config=SimpleNamespace(
            get_from_extra_config=lambda name, default: options.get(name, default),
            kv_buffer_device="cuda",
        ),
    )
    return NixlPullConnectorScheduler(
        config,
        "decoder",
        SimpleNamespace(
            transfer_groups=[],
            has_mamba_layers=False,
            select_transfer_block_ids=lambda blocks: blocks,
        ),
    )


def _request(req_id="r1", *, remote_id=None, producer=False):
    return SimpleNamespace(
        request_id=req_id,
        prompt_token_ids=list(range(32)),
        num_prompt_tokens=32,
        num_computed_tokens=32,
        status=RequestStatus.FINISHED_ABORTED,
        kv_transfer_params={
            "do_remote_prefill": not producer,
            "do_remote_decode": producer,
            "remote_block_ids": ([5, 6],),
            "remote_engine_id": "producer",
            "remote_request_id": remote_id or "p-" + req_id,
            "remote_host": "localhost",
            "remote_port": 1234,
            "tp_size": 1,
        },
    )


def _allocate(scheduler, request, num_external_tokens=32):
    blocks = SimpleNamespace(
        get_unhashed_block_ids_all_groups=lambda: ([1, 2],),
        blocks=([],),
    )
    scheduler.update_state_after_alloc(request, blocks, num_external_tokens)


def _output(*finished):
    return SimpleNamespace(finished_recving=set(finished))


def test_lookup_does_not_spend_credit_and_full_capacity_defers():
    scheduler = _scheduler()
    first, second = _request(), _request("r2")
    for request in (first, second, first):
        assert scheduler.get_num_new_matched_tokens(request, 0) == (32, True)
    _allocate(scheduler, first)
    assert scheduler.get_num_new_matched_tokens(second, 0) == (None, False)
    scheduler.build_connector_meta(SimpleNamespace())
    assert scheduler.get_num_new_matched_tokens(second, 0) == (None, False)
    scheduler.update_connector_output(_output(first.request_id))
    assert scheduler.get_num_new_matched_tokens(second, 0) == (32, True)


def test_abort_cannot_reuse_credit_before_aggregated_retirement():
    scheduler = _scheduler()
    request = _request()
    _allocate(scheduler, request)
    scheduler.build_connector_meta(SimpleNamespace())
    scheduler.request_finished(request, ())
    assert scheduler.get_num_new_matched_tokens(_request("r2"), 0) == (None, False)
    scheduler.update_connector_output(_output("unrelated"))
    assert scheduler.get_num_new_matched_tokens(_request("r2"), 0) == (None, False)
    scheduler.update_connector_output(_output(request.request_id))
    scheduler.update_connector_output(_output(request.request_id))
    assert scheduler.get_num_new_matched_tokens(_request("r2"), 0) == (32, True)


def test_tp_skew_retains_credit_until_every_rank_retires():
    scheduler = _scheduler()
    request = _request()
    _allocate(scheduler, request)
    aggregator = KVOutputAggregator(expected_finished_count=2)

    def step(*ranks):
        outputs = [
            ModelRunnerOutput(
                req_ids=[],
                req_id_to_index={},
                kv_connector_output=KVConnectorOutput(finished_recving=finished),
            )
            for finished in ranks
        ]
        output = aggregator.aggregate(outputs)
        scheduler.update_connector_output(output.kv_connector_output)

    step({request.request_id}, None)
    assert scheduler.get_num_new_matched_tokens(_request("r2"), 0) == (None, False)
    step(None, {request.request_id})
    assert scheduler.get_num_new_matched_tokens(_request("r2"), 0) == (32, True)


def test_multiconnector_preserves_admission_deferral():
    scheduler = _scheduler()
    _allocate(scheduler, _request())
    multi = object.__new__(MultiConnector)
    multi._connectors = [scheduler]
    multi._requests_to_connector = {}
    assert multi.get_num_new_matched_tokens(_request("r2"), 0) == (None, False)


def test_prefix_hit_has_generation_without_spending_receive_credit():
    scheduler = _scheduler()
    first, prefix_hit = _request(), _request("r2")
    _allocate(scheduler, first)
    assert scheduler.get_num_new_matched_tokens(prefix_hit, 32) == (0, False)
    _allocate(scheduler, prefix_hit, num_external_tokens=0)
    meta = scheduler.build_connector_meta(SimpleNamespace())
    first_meta, hit_meta = meta.reqs_to_recv["r1"], meta.reqs_to_recv["r2"]
    assert first_meta.receiver_is_async
    assert not hit_meta.receiver_is_async
    assert hit_meta.local_block_ids == ()
    assert hit_meta.receiver_generation > first_meta.receiver_generation > 0
    assert scheduler.get_num_new_matched_tokens(_request("r3"), 0) == (None, False)
    scheduler.update_connector_output(_output("r1"))
    assert scheduler.get_num_new_matched_tokens(_request("r3"), 0) == (32, True)


def test_preallocation_abort_is_notify_only_with_owned_remote_metadata():
    scheduler = _scheduler()
    request = _request()
    assert scheduler.request_finished(request, ()) == (False, None)
    meta = scheduler.build_connector_meta(SimpleNamespace()).reqs_to_recv["r1"]
    request.kv_transfer_params["remote_block_ids"][0].append(7)
    assert meta.receiver_generation > 0
    assert not meta.receiver_is_async
    assert meta.local_block_ids == ()
    assert meta.remote.block_ids == ([5, 6],)
    assert scheduler.get_num_new_matched_tokens(_request("r2"), 0) == (32, True)


def test_heartbeat_snapshots_are_immutable_versioned_and_include_removal():
    scheduler = _scheduler()
    empty = scheduler.build_connector_meta(SimpleNamespace())
    assert empty.receiver_heartbeat_version == 0
    assert empty.heartbeat_by_engine == {}
    unchanged = scheduler.build_connector_meta(SimpleNamespace())
    assert unchanged.receiver_heartbeat_version is None
    scheduler.on_new_request(_request())
    added = scheduler.build_connector_meta(SimpleNamespace())
    assert added.receiver_heartbeat_version == 1
    assert added.heartbeat_by_engine["producer"].req_ids == {"p-r1"}
    scheduler.on_new_request(_request("r2"))
    assert added.heartbeat_by_engine["producer"].req_ids == {"p-r1"}
    scheduler.update_connector_output(_output("r1", "r2"))
    removed = scheduler.build_connector_meta(SimpleNamespace())
    assert removed.receiver_heartbeat_version > added.receiver_heartbeat_version
    assert removed.heartbeat_by_engine == {}
    assert (
        scheduler.build_connector_meta(SimpleNamespace()).receiver_heartbeat_version
        is None
    )


def test_identity_tombstone_survives_request_retirement():
    scheduler = _scheduler()
    request = _request(producer=True)
    scheduler.on_new_request(request)
    scheduler.request_finished(request, ())
    with pytest.raises(RuntimeError, match="request ID reuse"):
        scheduler.on_new_request(_request(producer=True))


def test_independent_identity_fence_protects_feature_off_producer():
    scheduler = _scheduler(
        background_receiver=False, background_receiver_enforce_unique_ids=True
    )
    scheduler.on_new_request(_request(producer=True))
    with pytest.raises(RuntimeError, match="request ID reuse"):
        scheduler.on_new_request(_request(producer=True))


def test_resumable_input_cannot_reuse_an_identity_without_a_new_request_hook():
    scheduler = _scheduler()
    request = _request(producer=True)
    request.resumable = True
    with pytest.raises(ValueError, match="does not support resumable requests"):
        scheduler.on_new_request(request)


def test_distinct_local_request_cannot_consume_same_producer_identity_twice():
    scheduler = _scheduler()
    scheduler.on_new_request(_request(remote_id="shared"))
    scheduler.update_connector_output(_output("r1"))
    with pytest.raises(RuntimeError, match="cannot consume a producer request twice"):
        scheduler.on_new_request(_request("r2", remote_id="shared"))


def test_identity_capacity_fails_without_evicting_old_fences():
    scheduler = _scheduler(background_receiver_max_seen_requests=1)
    scheduler.on_new_request(_request(producer=True))
    scheduler.request_finished(_request(producer=True), ())
    with pytest.raises(RuntimeError, match="identity capacity exhausted"):
        scheduler.on_new_request(_request("r2", producer=True))
    with pytest.raises(RuntimeError, match="request ID reuse"):
        scheduler.on_new_request(_request(producer=True))


def test_heartbeat_capacity_failure_is_visible():
    scheduler = _scheduler(background_receiver_max_heartbeat_targets=1)
    scheduler.on_new_request(_request())
    with pytest.raises(RuntimeError, match="heartbeat capacity exhausted"):
        scheduler.on_new_request(_request("r2"))


def test_feature_off_preserves_legacy_unbounded_lookup_and_heartbeat_throttle():
    scheduler = _scheduler(background_receiver=False)
    first, second = _request(), _request("r2")
    scheduler.on_new_request(first)
    _allocate(scheduler, first)
    assert scheduler.get_num_new_matched_tokens(second, 0) == (32, True)
    meta = scheduler.build_connector_meta(SimpleNamespace())
    assert meta.reqs_to_recv["r1"].receiver_generation == 0
    assert meta.receiver_heartbeat_version is None
    assert meta.heartbeat_by_engine["producer"].req_ids == {"p-r1"}
    assert scheduler.build_connector_meta(SimpleNamespace()).heartbeat_by_engine == {}


@pytest.mark.parametrize(
    "option",
    [
        "background_receiver_max_pending",
        "background_receiver_max_heartbeat_targets",
        "background_receiver_max_seen_requests",
    ],
)
def test_invalid_capacity_fails_at_initialization(option):
    with pytest.raises(ValueError, match="must be positive"):
        _scheduler(**{option: 0})


@pytest.mark.parametrize(
    "option", ["background_receiver", "background_receiver_enforce_unique_ids"]
)
def test_string_false_is_not_silently_enabled(option):
    with pytest.raises(ValueError, match="flags must be boolean"):
        _scheduler(**{option: "false"})


def test_notify_capacity_must_cover_the_lifetime_identity_budget():
    with pytest.raises(ValueError, match="notification capacity"):
        _scheduler(background_receiver_max_notify_only=128)


def test_receiver_cannot_disable_request_identity_fencing():
    with pytest.raises(ValueError, match="unique request IDs"):
        _scheduler(background_receiver_enforce_unique_ids=False)


def test_push_connector_rejects_receiver_before_constructing_scheduler(monkeypatch):
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
        NixlBaseConnector,
        NixlPushConnector,
    )

    monkeypatch.setattr(NixlBaseConnector, "__init__", lambda *args: None)
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            get_from_extra_config=lambda name, default: True,
        )
    )
    with pytest.raises(ValueError, match="only by NixlConnector"):
        NixlPushConnector(config, None, None)
