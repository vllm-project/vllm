# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Control mutations must target one engine even under the DP utility broadcast."""

import uuid
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from vllm.config import CacheConfig, KVControlConfig, ParallelConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    get_request_block_hasher,
    init_none_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.core_client import DPLBAsyncMPClient, InprocClient
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.kv_hints.api import KVControlService
from vllm.v1.kv_hints.control import ControlContext, ControlError
from vllm.v1.kv_hints.engine import create_control_service
from vllm.v1.kv_hints.protocol import KvHintAction, KvHintsEnvelope
from vllm.v1.request import Request
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

pytestmark = pytest.mark.cpu_test


def make_service(
    *, engine_id="test", rank=0, weight_version="default", requests=None, **limits
):
    config = SimpleNamespace(
        kv_control_config=KVControlConfig(engine_id=engine_id, **limits),
        cache_config=CacheConfig(block_size=16),
        model_config=SimpleNamespace(compute_hash=lambda: "model"),
        parallel_config=ParallelConfig(data_parallel_size=2, data_parallel_rank=rank),
    )
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=16, dtype=torch.float16
    )
    manager = KVCacheManager(
        KVCacheConfig(32, [], [KVCacheGroupSpec(["layer"], spec)]),
        max_model_len=128,
        scheduler_block_size=16,
        hash_block_size=16,
    )
    return create_control_service(
        config,
        manager,
        weight_version,
        request_lookup=requests.get if requests is not None else None,
    )


def test_engine_epochs_differ_but_compatible_dp_ranks_share_namespace():
    a, b = make_service(rank=0), make_service(rank=1)
    assert a.controller.registry.engine_id == "test:dp0"
    assert b.controller.registry.engine_id == "test:dp1"
    assert a.controller.registry.engine_epoch != b.controller.registry.engine_epoch
    assert a.controller.index.namespace == b.controller.index.namespace
    assert (
        make_service(weight_version="updated").controller.index.namespace
        != a.controller.index.namespace
    )


def make_weight_update_engine():
    engine = object.__new__(EngineCore)
    engine._weight_version = "default"
    engine.kv_control_service = make_service()
    manager = engine.kv_control_service.controller.executor.manager
    engine.vllm_config = SimpleNamespace(
        kv_control_config=KVControlConfig(engine_id="test"),
        cache_config=CacheConfig(block_size=16),
        model_config=SimpleNamespace(compute_hash=lambda: "model"),
        parallel_config=ParallelConfig(data_parallel_size=2, data_parallel_rank=0),
    )
    engine.batch_queue = None
    engine.reset_mm_cache = lambda: None
    engine.reset_encoder_cache = lambda: None
    engine.scheduler = SimpleNamespace(
        kv_cache_manager=manager,
        requests={},
        connector=None,
        reset_prefix_cache=lambda *args, **kwargs: manager.reset_prefix_cache(),
    )
    return engine


def test_weight_version_transition_invalidates_old_namespace_and_epoch():
    engine = make_weight_update_engine()
    before = engine.get_kv_control_capabilities()
    engine.set_weight_version("default")
    assert engine.get_kv_control_capabilities() == before
    engine.set_weight_version("updated")
    after = engine.get_kv_control_capabilities()
    assert after["namespace"] != before["namespace"]
    assert after["engine_epoch"] != before["engine_epoch"]
    assert after["engine_id"] == before["engine_id"]
    assert after["cache_profile"]["request_cache_salt_prefix"] == after["namespace"]
    old_context = {k: before[k] for k in ("engine_id", "engine_epoch", "namespace")}
    assert (
        engine.kv_control("associations", old_context, {})["error"]["code"]
        == "stale_epoch"
    )
    engine.set_weight_version("next")
    assert engine.get_kv_control_capabilities()["namespace"] != after["namespace"]
    engine.kv_control_service = None
    engine.set_weight_version("updated")
    assert engine.get_weight_version() == "updated"


def test_weight_transition_refuses_active_requests_and_uncommitted_tensors():
    engine = make_weight_update_engine()
    engine.scheduler.requests["running"] = object()
    with pytest.raises(RuntimeError, match="Drain inference"):
        engine.set_weight_version("updated")
    assert engine.get_weight_version() == "default"
    engine.scheduler.requests.clear()
    engine._prepare_kv_weight_update()
    with pytest.raises(RuntimeError, match="Finish transferring"):
        engine.set_weight_version("updated")
    assert engine.kv_control("submit", {}, {})["error"]["code"] == "busy"
    engine._kv_weight_update_finished = True
    with pytest.raises(RuntimeError, match="new weight version"):
        engine.set_weight_version("default")
    engine.set_weight_version("updated")
    assert not engine._kv_weight_update_active


@pytest.mark.parametrize(
    "settings",
    [
        {"max_records": 0},
        {"result_ttl": float("inf")},
        {"attached_consumer": "guess"},
        {"engine_id": ""},
    ],
)
def test_invalid_control_configuration_fails_before_engine_startup(settings):
    with pytest.raises(ValidationError):
        KVControlConfig(**settings)


def test_engine_utility_uses_safe_serialization_and_disabled_path_has_no_state():
    engine = object.__new__(EngineCore)
    engine.kv_control_service = None
    assert engine.get_kv_control_capabilities() is None
    assert engine.kv_control("associations", {}, {})["error"]["code"] == "unsupported"
    engine.kv_control_service = make_service()
    caps = engine.get_kv_control_capabilities()
    invocation = {key: caps[key] for key in ("engine_id", "engine_epoch")}
    reply = engine.kv_control("associations", invocation, {})
    assert reply["result"]["items"] == []
    assert MsgpackDecoder().decode(MsgpackEncoder().encode(reply)) == reply


def test_residency_query_observes_eviction_without_deleting_logical_content():
    service = make_service()
    controller = service.controller
    pool = controller.executor.manager.block_pool
    block = pool.get_new_blocks(1)[0]
    key = make_block_hash_with_group_id(BlockHash(b"x" * 32), 0)
    pool.cache_transferred_block(block, key, 16)
    pool.free_blocks([block])
    invocation = {
        "engine_id": controller.registry.engine_id,
        "engine_epoch": controller.registry.engine_epoch,
        "namespace": controller.index.namespace,
    }
    payload = {
        "target": {
            "kind": "hashes",
            "namespace": controller.index.namespace,
            "blocks": [
                {
                    "block_hash": (b"x" * 32).hex(),
                    "group_id": 0,
                    "start_token": 0,
                    "end_token": 16,
                }
            ],
        }
    }
    before = service.call("residency", invocation, payload)
    assert before["ok"]
    assert before["result"]["items"][0]["resident_copies"] == 1
    assert before["result"]["items"][0]["referenced_copies"] == 0
    pool.evict_unreferenced([key])
    after = service.call("residency", invocation, payload)
    assert after["ok"]
    assert after["result"]["items"][0]["resident_copies"] == 0
    assert after["result"]["engine_epoch"] == invocation["engine_epoch"]


def test_engine_liveness_includes_request_free_executor_progress():
    engine = object.__new__(EngineCore)
    engine.scheduler = SimpleNamespace(has_requests=lambda: False)
    engine.batch_queue = None
    engine.engines_running = False
    engine.kv_control_service = SimpleNamespace(
        controller=SimpleNamespace(
            executor=SimpleNamespace(has_pending=True), policies=None
        )
    )
    assert engine.has_pending_kv_control()
    engine.kv_control_service.controller.executor.has_pending = False
    assert not engine.has_pending_kv_control()


@pytest.mark.asyncio
async def test_dp_control_mutation_is_sent_only_to_the_named_engine():
    client = object.__new__(DPLBAsyncMPClient)
    client.core_engines = [b"a", b"b"]
    calls = []

    async def utility(method, *args, engine):
        calls.append((engine, method))
        if method == "get_kv_control_capabilities":
            return {"engine_id": engine.decode(), "engine_epoch": "epoch"}
        return {"ok": True, "result": {"engine_id": engine.decode()}}

    client._call_utility_async = utility
    result = await client.kv_control_async(
        "submit", {"engine_id": "b", "engine_epoch": "epoch"}, {}
    )
    assert result["engine_id"] == "b"
    assert [engine for engine, method in calls if method == "kv_control"] == [b"b"]
    with pytest.raises(ControlError, match="not connected"):
        await client.kv_control_async(
            "submit", {"engine_id": "absent", "engine_epoch": "epoch"}, {}
        )
    assert [engine for engine, method in calls if method == "kv_control"] == [b"b"]


@pytest.mark.asyncio
async def test_dp_discovery_rejects_ambiguous_engine_ids_before_mutation():
    client = object.__new__(DPLBAsyncMPClient)
    client.core_engines = [b"a", b"b"]

    async def utility(method, *args, engine):
        assert method == "get_kv_control_capabilities"
        return {"engine_id": "duplicate", "engine_epoch": engine.decode()}

    client._call_utility_async = utility
    with pytest.raises(ControlError, match="duplicate"):
        await client.kv_control_async(
            "submit", {"engine_id": "duplicate", "engine_epoch": "epoch"}, {}
        )


@pytest.mark.parametrize("consumer", ["native", "backend", None])
def test_request_envelope_has_one_interpretation_owner_before_connector_hook(consumer):
    engine = object.__new__(EngineCore)
    engine.kv_control_service = make_service() if consumer is not None else None
    engine.vllm_config = SimpleNamespace(
        kv_control_config=KVControlConfig(attached_consumer=consumer or "native")
    )
    forwarded = []
    engine.scheduler = SimpleNamespace(
        add_request=lambda request: forwarded.append(request.kv_hints),
    )
    hints = KvHintsEnvelope(
        "0.1",
        "attached",
        [
            KvHintAction(
                "action",
                "signal",
                "1",
                {
                    "target": {"kind": "session", "session_id": "s"},
                    "signal": "tool_wait",
                },
            )
        ],
    )
    if consumer == "backend":
        hints = KvHintsEnvelope(
            "opaque-version",
            "attached",
            [
                KvHintAction(
                    "action",
                    "backend.future",
                    "v9",
                    {"values": [None, True, {"n": 3.5}]},
                )
            ],
        )
    request = Request(
        "request", [1], SamplingParams(max_tokens=1), None, kv_hints=hints
    )
    engine.add_request(request)
    assert forwarded == [None if consumer == "native" else hints]
    if consumer is not None:
        operation = engine.kv_control_service.controller.registry.get(
            "attached", "action"
        )
        if consumer == "native":
            assert operation.status == "unconsumed"
            assert operation.binding_id == "request"
        else:
            assert operation is None


def test_invalid_attached_wrapper_aborts_only_that_request_and_never_forwards():
    engine = object.__new__(EngineCore)
    engine.kv_control_service = make_service()
    engine.vllm_config = SimpleNamespace(kv_control_config=KVControlConfig())
    forwarded, aborted = [], []
    engine.scheduler = SimpleNamespace(
        add_request=lambda request: forwarded.append(request.kv_hints)
    )
    engine.abort_requests = lambda ids: aborted.extend(ids)
    request = Request(
        "bad",
        [1],
        SamplingParams(max_tokens=1),
        None,
        kv_hints=KvHintsEnvelope("0.1", "", []),
    )
    engine.add_request(request)
    assert forwarded == [None]
    assert aborted == ["bad"]
    assert engine.kv_control_service.controller.index.query().items == ()


def token_request(identity, tokens, **kwargs):
    init_none_hash(sha256)
    return Request(
        identity,
        tokens,
        SamplingParams(max_tokens=64),
        None,
        block_hasher=get_request_block_hasher(16, sha256),
        **kwargs,
    )


def test_weight_hashes_match_fresh_peer_and_rollback_without_merging_user_salts():
    first, peer = make_weight_update_engine(), make_weight_update_engine()
    for engine in (first, peer):
        engine._kv_weight_hash_salt = (
            engine.kv_control_service.controller.cache_profile[
                "request_cache_salt_prefix"
            ]
        )
        engine.scheduler.add_request = lambda request: None

    def hashes(engine, salt):
        request = token_request(uuid.uuid4().hex, list(range(32)), cache_salt=salt)
        engine.add_request(request)
        return request.block_hashes

    original = hashes(first, None)
    assert original == hashes(peer, None)
    assert original != hashes(first, "tenant-a")
    first.set_weight_version("next")
    assert original != hashes(first, None)
    first.set_weight_version("default")
    assert original == hashes(first, None) == hashes(peer, None)


def native_submit(service, kind, payload):
    registry = service.controller.registry
    return service.controller.submit(
        KvHintsEnvelope(
            "0.1", uuid.uuid4().hex, [KvHintAction("action", kind, "1", payload)]
        ),
        ControlContext(registry.engine_id, registry.engine_epoch, "native"),
    ).results[0]


def associate_request(
    service, request, *, continuation="main", revision=0, growth=False, **extra
):
    return native_submit(
        service,
        "associate",
        {
            "session_id": "s",
            "continuation_id": continuation,
            "expected_revision": revision,
            "target": {"kind": "request", "request_id": request.request_id},
            "observe_growth": growth,
            **extra,
        },
    )


@pytest.mark.parametrize("growth", [False, True])
def test_request_association_grows_only_with_explicit_observation(growth):
    request = token_request("r", [1] * 32)
    service = make_service(requests={request.request_id: request})
    result = associate_request(service, request, growth=growth)
    assert result.status == "completed"
    before = service.controller.index.get("s", "main")
    request.append_output_token_ids([2] * 16)
    service.controller.request_observer.observe()
    after = service.controller.index.get("s", "main")
    assert after.checkpoint.token_count == (48 if growth else 32)
    assert len(after.blocks) == (3 if growth else 2)
    assert after.incarnation == before.incarnation


@pytest.mark.parametrize("tokens", [[9, 2, 3, 4], [1, 2]])
def test_rewrite_below_the_first_cache_block_is_rejected_by_token_evidence(tokens):
    original = token_request("original", [1, 2, 3])
    requests = {original.request_id: original}
    service = make_service(requests=requests)
    assert associate_request(service, original).status == "partial"
    before = service.controller.index.query()
    replacement = token_request("replacement", tokens)
    requests[replacement.request_id] = replacement
    result = associate_request(service, replacement, revision=before.items[0].revision)
    assert result.status == "failed" and "new continuation ID" in result.reason
    assert service.controller.index.query() == before


def test_metadata_growth_preserves_retention_view_and_original_expiry():
    request = token_request("r", [1] * 32)
    service = make_service(requests={request.request_id: request})
    assert associate_request(service, request, growth=True).status == "completed"
    native_submit(
        service,
        "retain",
        {
            "target": {"kind": "session", "session_id": "s", "continuation_id": "main"},
            "owner_id": "s",
            "declaration_id": "d",
            "priority": 70,
            "ttl_seconds": 300,
            "scope": "follow_growth",
        },
    )
    declarations = service.controller.declarations
    declared = declarations.get("s", "d")
    version = declarations.priority_snapshot()[0]
    request.append_output_token_ids(2)
    service.controller.request_observer.observe()
    assert service.controller.index.get("s", "main").checkpoint.token_count == 33
    assert declarations.priority_snapshot()[0] == version
    request.append_output_token_ids([2] * 15)
    service.controller.request_observer.observe()
    assert declarations.priority(0, request.block_hashes[2].hex()) == 70
    assert declarations.get("s", "d") == declared


def test_live_request_binding_defers_a_concurrent_writer_but_allows_a_branch():
    first, other = token_request("a", [1] * 32), token_request("b", [1] * 32)
    service = make_service(requests={"a": first, "b": other})
    associate_request(service, first, growth=True)
    current = service.controller.index.get("s", "main")
    assert (
        associate_request(service, other, revision=current.revision).status
        == "deferred"
    )
    assert (
        associate_request(service, other, continuation="branch", growth=True).status
        == "completed"
    )
    assert service.controller.index.members(0, first.block_hashes[0].hex()) == (
        ("s", "branch"),
        ("s", "main"),
    )


def test_growth_capacity_failure_is_an_event_and_preserves_the_last_valid_state():
    request = token_request("r", [1] * 32)
    service = make_service(requests={"r": request}, max_associations=2)
    associate_request(service, request, growth=True)
    before = service.controller.index.get("s", "main")
    request.append_output_token_ids([2] * 16)
    service.controller.request_observer.observe()
    assert service.controller.index.get("s", "main") == before
    events = service.controller.index.events(
        0, epoch=service.controller.registry.engine_epoch
    )
    assert events[-1].kind == "observation_failed" and "capacity" in events[-1].reason
    position = events[-1].sequence
    service.controller.request_observer.observe()
    assert (
        service.controller.index.events(position, epoch=events[-1].engine_epoch) == ()
    )


def test_task_cleanup_prevents_late_observation_from_rebinding_a_recreated_id():
    request = token_request("r", [1] * 32)
    service = make_service(requests={"r": request})
    associate_request(service, request, growth=True)
    old = service.controller.index.get("s", "main")
    assert (
        native_submit(
            service,
            "finish_task",
            {"session_id": "s", "expected_revision": old.revision},
        ).status
        == "completed"
    )
    assert (
        native_submit(
            service,
            "associate",
            {
                "session_id": "s",
                "continuation_id": "main",
                "expected_revision": 0,
                "target": {
                    "kind": "hashes",
                    "namespace": service.controller.index.namespace,
                    "blocks": [],
                },
            },
        ).status
        == "completed"
    )
    recreated = service.controller.index.get("s", "main")
    request.append_output_token_ids([2] * 16)
    service.controller.request_observer.observe()
    assert service.controller.index.get("s", "main") == recreated
    assert recreated.incarnation != old.incarnation


@pytest.mark.parametrize("kwargs", [{"cache_salt": "changed"}, {"resumable": True}])
def test_request_context_changes_or_unsupported_streaming_do_not_change_a_branch(
    kwargs,
):
    first = token_request("a", [1] * 32)
    requests = {"a": first}
    service = make_service(requests=requests)
    associate_request(service, first)
    before = service.controller.index.query()
    other = token_request("b", [1] * 32, **kwargs)
    requests["b"] = other
    result = associate_request(service, other, revision=before.items[0].revision)
    assert result.status == ("unsupported" if kwargs.get("resumable") else "failed")
    assert service.controller.index.query() == before


def test_cache_hit_request_adds_logical_membership_without_allocating_another_copy():
    first, hit = token_request("a", [1] * 33), token_request("b", [1] * 33)
    service = make_service(requests={"a": first, "b": hit})
    manager = service.controller.executor.manager
    assert manager.allocate_slots(first, num_new_tokens=32) is not None
    manager.free(first)
    _, count, _ = manager.get_computed_blocks(hit)
    assert count == 32
    before = manager.block_pool.get_num_free_blocks()
    assert associate_request(service, first).status == "partial"
    assert associate_request(service, hit, continuation="hit").status == "partial"
    assert manager.block_pool.get_num_free_blocks() == before
    assert service.controller.index.members(0, first.block_hashes[1].hex()) == (
        ("s", "hit"),
        ("s", "main"),
    )


def test_finished_request_publishes_final_growth_and_then_stops_observing():
    from vllm.v1.request import RequestStatus

    request = token_request("r", [1] * 32)
    service = make_service(requests={"r": request})
    associate_request(service, request, growth=True)
    request.append_output_token_ids([2] * 16)
    request.status = RequestStatus.FINISHED_STOPPED
    engine = object.__new__(EngineCore)
    engine.kv_control_service = service
    engine.check_for_draft_tokens = False
    engine.post_step(model_executed=True)
    final = service.controller.index.get("s", "main")
    assert final.checkpoint.token_count == 48
    request.append_output_token_ids([2] * 16)
    engine.post_step(model_executed=False)
    assert service.controller.index.get("s", "main") == final


def test_child_request_cannot_rewrite_the_captured_parent_prefix():
    parent, child = token_request("parent", [1] * 32), token_request("child", [9] * 32)
    service = make_service(requests={"parent": parent, "child": child})
    associate_request(service, parent)
    result = associate_request(
        service,
        child,
        continuation="child",
        parent_prefix={
            "session_id": "s",
            "continuation_id": "main",
            "end_token": 16,
        },
    )
    assert result.status == "failed" and "previously associated" in result.reason
    assert len(service.controller.index.query().items) == 1


@pytest.mark.parametrize("inference_active", [False, True])
def test_inproc_control_advances_idle_transfers_without_consuming_inference(
    inference_active,
):
    from tests.v1.core.test_prefix_caching import make_movement_controller

    init_none_hash(sha256)
    manager, request, controller, adapter, target = make_movement_controller()
    engine = object.__new__(EngineCore)
    engine.kv_control_service = KVControlService(controller)
    engine.check_for_draft_tokens = False
    engine.scheduler = SimpleNamespace(
        has_unfinished_requests=lambda: inference_active,
        has_requests=lambda: bool(adapter.jobs),
    )

    def step():
        adapter.finish()
        return {}, False

    engine.step_fn = step
    client = object.__new__(InprocClient)
    client.engine_core = engine
    invocation = {"engine_id": "engine", "engine_epoch": "epoch"}
    client.kv_control(
        "submit",
        invocation,
        {
            "protocol_version": "0.1",
            "message_id": "load",
            "actions": [
                {
                    "action_id": "a",
                    "action_type": "prefetch",
                    "action_version": "1",
                    "payload": {"target": target, "source": "mooncake"},
                }
            ],
        },
    )
    result = client.kv_control(
        "operation", invocation, {"message_id": "load", "action_id": "a"}
    )
    assert result["status"] == ("running" if inference_active else "completed")
    assert manager.get_computed_blocks(request)[1] == (0 if inference_active else 4)
    if inference_active:
        client.get_output()
        assert manager.get_computed_blocks(request)[1] == 4


def primitive_envelope(name, kind, payload):
    return KvHintsEnvelope("0.1", name, [KvHintAction("action", kind, "1.0", payload)])


def primitive_scheduler(*, async_scheduling=False):
    from tests.v1.core.test_prefix_caching import QueuedControlTransfers
    from tests.v1.core.utils import create_scheduler
    from vllm.v1.kv_hints.transfers import KVTransferOperations

    scheduler = create_scheduler(
        enable_prefix_caching=True,
        num_blocks=9,
        max_num_seqs=2,
        max_num_batched_tokens=128,
        async_scheduling=async_scheduling,
    )
    scheduler.vllm_config.kv_control_config = KVControlConfig(engine_id="test")
    service = create_control_service(
        scheduler.vllm_config,
        scheduler.kv_cache_manager,
        "default",
        request_lookup=scheduler.requests.get,
    )
    controller = service.controller
    adapter = QueuedControlTransfers()
    controller.executor.transfers = KVTransferOperations(
        scheduler.kv_cache_manager,
        controller.registry,
        controller.index.namespace,
        (adapter,),
    )
    scheduler.kv_hint_dispatcher = controller.dispatcher
    engine = object.__new__(EngineCore)
    engine.scheduler = scheduler
    engine.vllm_config = scheduler.vllm_config
    engine.kv_control_service = service
    engine._kv_weight_hash_salt = service.controller.cache_profile[
        "request_cache_salt_prefix"
    ]
    return engine, controller, adapter


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("abort", [False, True])
def test_attached_move_acquires_source_holds_at_successful_completion(
    async_scheduling, abort
):
    from vllm.v1.outputs import ModelRunnerOutput

    engine, controller, adapter = primitive_scheduler(async_scheduling=async_scheduling)
    request = token_request("producer", [1] * 32)
    request.sampling_params.max_tokens = request.max_tokens = 1
    request.kv_hints = primitive_envelope(
        "move",
        "vllm.move_to",
        {
            "target": "request_cached",
            "src": {"tier": "gpu"},
            "dst": {"tier": "mooncake"},
        },
    )
    engine.add_request(request)
    assert request.kv_hints is None
    operation = controller.registry.get("move", "action")
    assert operation.status == "queued" and not operation.status.terminal
    assert not adapter.jobs
    scheduled = engine.scheduler.schedule()
    manager = engine.scheduler.kv_cache_manager
    blocks = list(manager.get_blocks(request.request_id).blocks[0])
    if abort:
        engine.abort_requests([request.request_id])
    else:
        engine.scheduler.update_from_output(
            scheduled,
            ModelRunnerOutput(
                req_ids=[request.request_id],
                req_id_to_index={request.request_id: 0},
                sampled_token_ids=[[0]],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )
    result = controller.registry.get("move", "action")
    assert result.status == ("cancelled" if abort else "running")
    if abort:
        assert not adapter.jobs
    else:
        assert len(adapter.jobs) == 2
        assert all(block.ref_cnt >= 1 for block in blocks)
        adapter.finish()
        controller.executor.progress()
        result = controller.registry.get("move", "action")
        assert result.status == "completed"
        assert result.completed_steps == ("cold_copy_stored", "gpu_copies_reclaimed")
        assert all(block.ref_cnt == 0 and block.block_hash is None for block in blocks)
    assert request.is_finished()
    assert not controller.dispatcher._pending


def test_attached_priority_ttl_starts_at_completion_and_duplicate_does_not_renew():
    from vllm.v1.outputs import ModelRunnerOutput

    engine, controller, _ = primitive_scheduler()
    now = [0.0]
    copies = controller.executor.retention_priority.copies
    copies.clock = lambda: now[0]
    request = token_request("priority", [1] * 32)
    request.sampling_params.max_tokens = request.max_tokens = 1
    hint = primitive_envelope(
        "retain",
        "vllm.set_priority",
        {
            "target": "request_cached",
            "location": "local_g1",
            "claim_id": "router/task",
            "revision": 1,
            "value": -10,
            "ttl_seconds": 5,
        },
    )
    request.kv_hints = hint
    engine.add_request(request)
    scheduled = engine.scheduler.schedule()
    blocks = list(
        engine.scheduler.kv_cache_manager.get_blocks(request.request_id).blocks[0]
    )
    assert all(copies.priority(b) is None for b in blocks)
    now[0] = 100
    engine.scheduler.update_from_output(
        scheduled,
        ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[[0]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    assert controller.registry.get("retain", "action").status == "completed"
    assert all(copies.priority(b) == -10 for b in blocks)
    now[0] = 104
    registry = controller.registry
    replay = controller.submit(
        hint,
        ControlContext(registry.engine_id, registry.engine_epoch, "native"),
        request_id=request.request_id,
    )
    assert replay.replayed
    now[0] = 105
    copies.expire()
    assert all(copies.priority(b) is None for b in blocks)


def test_primitive_location_cannot_mutate_another_engine():
    service = make_service()
    registry = service.controller.registry
    result = service.controller.submit(
        primitive_envelope(
            "remote",
            "vllm.copy_to",
            {
                "target": {
                    "kind": "hashes",
                    "namespace": service.controller.index.namespace,
                    "blocks": [],
                },
                "src": {"tier": "mooncake"},
                "dst": {"tier": "gpu", "engine_id": "another"},
            },
        ),
        ControlContext(registry.engine_id, registry.engine_epoch, "native"),
    ).results[0]
    assert result.status == "unsupported" and "Remote engines" in result.reason


@pytest.mark.parametrize("logical", [False, True])
def test_completion_range_resolves_cached_checkpoints_no_longer_held_by_request(
    logical,
):
    from vllm.v1.request import RequestStatus

    engine, controller, adapter = primitive_scheduler()
    request = token_request("detached", [1] * 33)
    manager = engine.scheduler.kv_cache_manager
    manager.allocate_slots(request, 32)
    manager.free(request)
    assert not manager.get_blocks(request.request_id).blocks[0]
    target = (
        {"kind": "request", "request_id": "$request", "end_token": 32}
        if logical
        else "request_cached"
    )
    request.kv_hints = primitive_envelope(
        "detached",
        "vllm.copy_to",
        {
            "target": target,
            "src": {"tier": "gpu"},
            "dst": {"tier": "mooncake"},
        },
    )
    engine.add_request(request)
    engine.scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_STOPPED)
    result = controller.registry.get("detached", "action")
    assert result.status == ("running" if logical else "missing")
    assert len(adapter.jobs) == (2 if logical else 0)
    if logical:
        adapter.finish()
        controller.executor.progress()
        result = controller.registry.get("detached", "action")
        assert [o.end_token for o in result.target_outcomes] == [16, 32]
        assert all(o.transfer_id is not None for o in result.target_outcomes)


def test_primitive_dispatch_preserves_valid_siblings_and_explicit_backend_ownership():
    engine, controller, adapter = primitive_scheduler()
    request = token_request("mixed-validity", [1] * 32)
    good = primitive_envelope(
        "siblings",
        "vllm.set_priority",
        {
            "target": "request_cached",
            "location": "local_g1",
            "claim_id": "router",
            "revision": 1,
            "value": 5,
            "ttl_seconds": 10,
        },
    )
    request.kv_hints = KvHintsEnvelope(
        "0.1",
        "siblings",
        [
            KvHintAction("invalid", "vllm.move_to", "future", {}),
            *good.actions,
        ],
    )
    engine.add_request(request)
    assert controller.registry.get("siblings", "invalid").status == "unsupported"
    assert controller.registry.get("siblings", "action").status == "queued"
    engine.abort_requests([request.request_id])
    assert controller.registry.get("siblings", "action").status == "cancelled"
    registry = controller.registry
    assert (
        controller.consume_attached(
            good,
            ControlContext(registry.engine_id, registry.engine_epoch, "backend"),
            request_id="opaque",
        )
        is None
    )
    assert not adapter.jobs
