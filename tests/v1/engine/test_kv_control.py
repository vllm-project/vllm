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
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
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
