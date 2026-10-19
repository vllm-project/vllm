# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Control correlation is independent of native or backend payload semantics."""

from dataclasses import FrozenInstanceError, replace
from typing import Any

import msgspec
import pytest

from vllm.v1.kv_hints import KvHintAction, KvHintsEnvelope
from vllm.v1.kv_hints.api import KVControlService, unwrap_control_reply
from vllm.v1.kv_hints.control import (
    ControlContext,
    ControlError,
    OperationOrigin,
    OperationRegistry,
    OperationStatus,
)
from vllm.v1.kv_hints.controller import NativeKVController
from vllm.v1.kv_hints.native import (
    AssociatePayload,
    DemotePayload,
    EvictPayload,
    FinishTaskPayload,
    HashTarget,
    PrefetchPayload,
    ReleasePayload,
    RetainPayload,
    SharePayload,
    SignalPayload,
    parse_native_action,
)
from vllm.v1.kv_hints.retention import RetentionDeclarations
from vllm.v1.kv_hints.state import SessionPrefixIndex

CONTEXT = ControlContext("engine-a", "epoch-a", "native")


def envelope(message_id="message", *, action_ids=("action",), payload=None):
    return KvHintsEnvelope(
        protocol_version="0.1",
        message_id=message_id,
        actions=[
            KvHintAction(
                action_id=action_id,
                action_type="retain",
                action_version="1",
                payload=payload if payload is not None else {"priority": 10},
            )
            for action_id in action_ids
        ],
    )


@pytest.fixture
def now():
    return [100.0]


@pytest.fixture
def registry(now):
    return OperationRegistry(
        "engine-a", engine_epoch="epoch-a", result_ttl=10, clock=lambda: now[0]
    )


def test_retry_observes_original_progress_without_refreshing_lifetime(registry, now):
    hint = envelope()
    received = registry.submit(hint, CONTEXT)
    assert not received.replayed
    assert received.results[0].status == OperationStatus.RECEIVED
    approved = registry.update(
        registry.get("message", "action"), OperationStatus.ACCEPTED
    )
    assert registry.submit(hint, CONTEXT).results == (approved,)
    registry.update(registry.get("message", "action"), OperationStatus.RUNNING)
    completed = registry.update(
        registry.get("message", "action"),
        OperationStatus.COMPLETED,
        completed_steps=("cold_copy", "gpu_reclamation"),
    )
    now[0] += 9
    retry = registry.submit(hint, CONTEXT)
    assert retry.replayed
    assert retry.results == (completed,)
    now[0] += 1
    assert registry.get("message", "action") is None
    assert not registry.submit(hint, CONTEXT).replayed


@pytest.mark.parametrize(
    "changed", ["payload", "order", "consumer", "version", "binding"]
)
def test_conflicting_message_cannot_reuse_ids(registry, changed):
    hint = envelope(action_ids=("a", "b"))
    registry.submit(hint, CONTEXT)
    context = CONTEXT
    if changed == "payload":
        hint.actions[0].payload["priority"] = 20
    elif changed == "order":
        hint.actions.reverse()
    elif changed == "consumer":
        context = replace(CONTEXT, consumer="backend")
    elif changed == "binding":
        context = replace(CONTEXT, binding_id="another-request")
    else:
        hint = msgspec.structs.replace(hint, protocol_version="0.2")
    with pytest.raises(ControlError) as error:
        registry.submit(hint, context)
    assert error.value.code == "conflict"
    assert len(registry.events(0, epoch="epoch-a")) == 2


def test_opaque_backend_payload_survives_without_native_version_validation(registry):
    payload = {"b": [None, True, False, 1, -2, 0.5, {"x": "雪"}], "a": "value"}
    hint = msgspec.structs.replace(envelope(payload=payload), protocol_version="kvcr")
    hint.actions[0] = msgspec.structs.replace(
        hint.actions[0], action_type="backend_only", action_version="2026-10"
    )
    original = msgspec.json.encode(hint)
    context = replace(CONTEXT, consumer="backend")
    registry.submit(hint, context)
    assert msgspec.json.encode(hint) == original
    reordered = dict(reversed(list(payload.items())))
    retry = msgspec.structs.replace(
        hint,
        actions=[msgspec.structs.replace(hint.actions[0], payload=reordered)],
    )
    result = registry.submit(retry, context)
    assert result.replayed
    assert result.results[0].status == OperationStatus.RECEIVED


@pytest.mark.parametrize(
    "payload",
    [
        {"n": float("nan")},
        {"n": float("inf")},
        {"n": b"bytes"},
        {"n": {1, 2}},
        {"n": (1, 2)},
        {1: "non-string key"},
    ],
)
def test_non_json_payload_has_no_receipt_or_effect(registry, payload):
    with pytest.raises(ControlError, match="JSON"):
        registry.submit(envelope(payload=payload), CONTEXT)
    assert registry.get("message", "action") is None
    assert registry.events(0, epoch="epoch-a") == ()


@pytest.mark.parametrize("invalid", ["duplicate", "empty_id", "wrong_type", "depth"])
def test_invalid_wrapper_is_rejected_before_recording_any_action(registry, invalid):
    hint = envelope(action_ids=("a", "b"))
    if invalid == "duplicate":
        hint.actions.append(hint.actions[0])
    elif invalid == "empty_id":
        hint = msgspec.structs.replace(hint, message_id="")
    elif invalid == "wrong_type":
        hint.actions.append({"action_id": "c"})
    else:
        payload: dict[str, Any] = {}
        for _ in range(33):
            payload = {"nested": payload}
        hint = envelope(payload=payload)
    with pytest.raises(ControlError):
        registry.submit(hint, CONTEXT)
    assert registry.events(0, epoch="epoch-a") == ()


@pytest.mark.parametrize("bound", ["max_actions", "max_envelope_bytes"])
def test_envelope_limits_apply_before_admission(bound):
    registry = OperationRegistry("engine-a", engine_epoch="epoch-a", **{bound: 1})
    with pytest.raises(ControlError):
        registry.submit(envelope(action_ids=("a", "b")), CONTEXT)
    assert registry.events(0, epoch="epoch-a") == ()


def test_capacity_rejection_is_atomic_and_never_prunes_active_or_unexpired(now):
    registry = OperationRegistry(
        "engine-a",
        engine_epoch="epoch-a",
        max_records=2,
        result_ttl=10,
        clock=lambda: now[0],
    )
    registry.submit(envelope("active"), CONTEXT)
    with pytest.raises(ControlError) as error:
        registry.submit(envelope("large", action_ids=("a", "b")), CONTEXT)
    assert error.value.code == "capacity"
    assert registry.get("large", "a") is None
    registry.submit(envelope("done"), CONTEXT)
    registry.update(registry.get("done", "action"), OperationStatus.COMPLETED)
    with pytest.raises(ControlError, match="capacity"):
        registry.submit(envelope("next"), CONTEXT)
    now[0] += 10
    registry.submit(envelope("next"), CONTEXT)
    assert registry.get("active", "action").status == OperationStatus.RECEIVED
    assert registry.get("done", "action") is None


def test_multi_action_message_expires_only_after_every_action_finishes(registry, now):
    hint = envelope(action_ids=("a", "b"))
    registry.submit(hint, CONTEXT)
    done = registry.update(registry.get("message", "a"), OperationStatus.COMPLETED)
    now[0] += 100
    assert registry.get("message", "a") == done
    registry.update(registry.get("message", "b"), OperationStatus.FAILED)
    now[0] += 10
    assert registry.get("message", "a") is None
    assert registry.get("message", "b") is None


def test_late_cancellation_cannot_cancel_a_reused_message_incarnation(registry, now):
    index = SessionPrefixIndex("namespace", registry.engine_epoch)
    service = KVControlService(
        NativeKVController(registry, index, RetentionDeclarations(index))
    )
    original = registry.submit(envelope(), CONTEXT).results[0]
    registry.update(original, OperationStatus.COMPLETED)
    now[0] += 10
    replacement = registry.submit(envelope(), CONTEXT).results[0]
    context = {"engine_id": CONTEXT.engine_id, "engine_epoch": CONTEXT.engine_epoch}
    payload = {
        "message_id": "message",
        "action_id": "action",
        "operation_id": original.operation_id,
    }
    reply = service.call("cancel", context, payload)
    assert reply["error"]["code"] == "stale_operation"
    assert registry.get("message", "action").status == OperationStatus.RECEIVED
    payload["operation_id"] = replacement.operation_id
    assert service.call("cancel", context, payload)["result"]["status"] == "cancelled"


def test_acknowledgement_frees_capacity_without_reexecuting_or_renewing(now):
    registry = OperationRegistry(
        "engine-a",
        engine_epoch="epoch-a",
        max_records=1,
        result_ttl=10,
        clock=lambda: now[0],
    )
    hint = envelope()
    operation = registry.submit(hint, CONTEXT).results[0]
    registry.update(operation, OperationStatus.COMPLETED)
    now[0] += 9
    registry.acknowledge("message")
    registry.acknowledge("message")
    assert registry.capacity()["records"] == 0
    assert registry.capacity()["acknowledged_messages"] == 1
    with pytest.raises(ControlError) as error:
        registry.submit(hint, CONTEXT)
    assert error.value.code == "acknowledged"
    with pytest.raises(ControlError, match="different content"):
        registry.submit(envelope(payload={"priority": 11}), CONTEXT)
    registry.submit(envelope("next"), CONTEXT)
    now[0] += 1
    assert registry.capacity()["acknowledged_messages"] == 0
    assert registry.capacity()["records"] == 1


def test_acknowledgement_preserves_active_siblings_and_bounded_replay_storage(now):
    registry = OperationRegistry(
        "engine-a",
        engine_epoch="epoch-a",
        max_acknowledged_messages=1,
        clock=lambda: now[0],
    )
    receipt = registry.submit(envelope(action_ids=("a", "b")), CONTEXT)
    registry.update(receipt.results[0], OperationStatus.COMPLETED)
    with pytest.raises(ControlError, match="active operations"):
        registry.acknowledge("message")
    assert registry.get("message", "a").status == OperationStatus.COMPLETED
    registry.update(receipt.results[1], OperationStatus.CANCELLED)
    registry.acknowledge("message")
    next_result = registry.submit(envelope("next"), CONTEXT).results[0]
    registry.update(next_result, OperationStatus.COMPLETED)
    with pytest.raises(ControlError, match="capacity exhausted"):
        registry.acknowledge("next")
    assert registry.get("next", "action").status == OperationStatus.COMPLETED


def test_result_expiry_follows_completion_order_and_reused_ids(registry, now):
    first = registry.submit(envelope("first"), CONTEXT).results[0]
    second = registry.submit(envelope("second"), CONTEXT).results[0]
    registry.update(second, OperationStatus.COMPLETED)
    now[0] += 5
    registry.update(first, OperationStatus.COMPLETED)
    registry.submit(envelope("empty", action_ids=()), CONTEXT)
    now[0] += 5
    assert registry.get("second", "action") is None
    replacement = registry.submit(envelope("second"), CONTEXT).results[0]
    assert registry.get("first", "action") is not None
    now[0] += 5
    assert registry.get("first", "action") is None
    assert not registry.submit(envelope("empty", action_ids=()), CONTEXT).replayed
    assert registry.get("second", "action") == replacement


def test_empty_envelopes_are_bounded_and_expire(now):
    registry = OperationRegistry(
        "engine-a",
        engine_epoch="epoch-a",
        max_records=1,
        result_ttl=10,
        clock=lambda: now[0],
    )
    empty = envelope(action_ids=())
    assert registry.submit(empty, CONTEXT).results == ()
    assert registry.submit(empty, CONTEXT).replayed
    with pytest.raises(ControlError, match="capacity"):
        registry.submit(envelope("next", action_ids=()), CONTEXT)
    now[0] += 10
    assert not registry.submit(empty, CONTEXT).replayed


@pytest.mark.parametrize("status", [s for s in OperationStatus if s.terminal])
def test_terminal_outcomes_are_distinct_immutable_and_idempotent(registry, status):
    registry.submit(envelope(), CONTEXT)
    result = registry.update(registry.get("message", "action"), status)
    assert result.status == status
    assert registry.update(registry.get("message", "action"), status) is result
    with pytest.raises(ControlError, match="terminal"):
        registry.update(registry.get("message", "action"), OperationStatus.ACCEPTED)
    with pytest.raises(FrozenInstanceError):
        result.reason = "changed"


def test_progress_requires_approval_and_preserves_completed_steps(registry):
    registry.submit(envelope(), CONTEXT)
    with pytest.raises(ControlError, match="approved"):
        registry.update(registry.get("message", "action"), OperationStatus.RUNNING)
    registry.update(registry.get("message", "action"), OperationStatus.ACCEPTED)
    progress = registry.update(
        registry.get("message", "action"),
        OperationStatus.RUNNING,
        completed_steps=("stored",),
    )
    with pytest.raises(ControlError, match="approval"):
        registry.update(registry.get("message", "action"), OperationStatus.ACCEPTED)
    with pytest.raises(ValueError, match="append-only"):
        registry.update(
            registry.get("message", "action"),
            OperationStatus.PARTIAL,
            completed_steps=(),
        )
    assert registry.get("message", "action") == progress


def test_queries_and_events_share_results_and_report_replay_gaps():
    registry = OperationRegistry("engine-a", engine_epoch="epoch-a", max_events=2)
    received = registry.submit(envelope(), CONTEXT).results[0]
    registry.update(registry.get("message", "action"), OperationStatus.ACCEPTED)
    done = registry.update(registry.get("message", "action"), OperationStatus.COMPLETED)
    with pytest.raises(ControlError) as error:
        registry.events(0, epoch="epoch-a")
    assert error.value.code == "event_gap"
    page = registry.events(received.sequence, epoch="epoch-a", limit=1)
    assert page[0].status == OperationStatus.ACCEPTED
    assert registry.events(page[0].sequence, epoch="epoch-a") == (done,)
    assert registry.get("message", "action") == done
    assert received.status == OperationStatus.RECEIVED


@pytest.mark.parametrize(
    "field,value,code",
    [
        ("engine_id", "engine-b", "wrong_engine"),
        ("engine_epoch", "epoch-old", "stale_epoch"),
        ("consumer", "broadcast", "unsupported_consumer"),
    ],
)
def test_routing_is_checked_before_records_change(registry, field, value, code):
    with pytest.raises(ControlError) as error:
        registry.submit(envelope(), replace(CONTEXT, **{field: value}))
    assert error.value.code == code
    assert registry.events(0, epoch="epoch-a") == ()


def test_restart_invalidates_old_results_and_event_cursors(registry):
    registry.submit(envelope(), CONTEXT)
    restarted = OperationRegistry("engine-a")
    assert restarted.get("message", "action") is None
    with pytest.raises(ControlError, match="epoch"):
        restarted.submit(envelope(), CONTEXT)
    with pytest.raises(ControlError, match="epoch"):
        restarted.events(0, epoch=CONTEXT.engine_epoch)


def test_late_completion_cannot_publish_into_reused_message_ids(registry, now):
    receipt = registry.submit(envelope(), CONTEXT).results[0]
    registry.update(receipt, OperationStatus.CANCELLED)
    now[0] += 10
    replacement = registry.submit(envelope(), CONTEXT).results[0]
    assert replacement.operation_id != receipt.operation_id
    with pytest.raises(ControlError) as error:
        registry.update(receipt, OperationStatus.COMPLETED)
    assert error.value.code == "stale_operation"
    assert registry.get("message", "action") == replacement


def native_action(action_type, payload, version="1"):
    return KvHintAction("action", action_type, version, payload)


SESSION_TARGET = {"kind": "session", "session_id": "s", "continuation_id": "c"}
HASH_TARGET: dict[str, Any] = {
    "kind": "hashes",
    "namespace": "model-layout-weights",
    "blocks": [
        {"block_hash": "0123", "group_id": 0, "start_token": 0, "end_token": 16}
    ],
}


@pytest.mark.parametrize(
    "action_type,fields,expected",
    [
        (
            "retain",
            {"target": SESSION_TARGET, "owner_id": "s", "declaration_id": "d"},
            RetainPayload,
        ),
        ("release", {"owner_id": "s", "declaration_id": "d"}, ReleasePayload),
        ("evict", {"target": SESSION_TARGET}, EvictPayload),
        ("demote", {"target": HASH_TARGET, "destination": "mooncake"}, DemotePayload),
        ("prefetch", {"target": HASH_TARGET, "source": "cpu"}, PrefetchPayload),
        (
            "share",
            {"target": HASH_TARGET, "destination_engine_id": "engine-b"},
            SharePayload,
        ),
        (
            "associate",
            {
                "session_id": "s",
                "continuation_id": "c",
                "expected_revision": 0,
                "target": HASH_TARGET,
            },
            AssociatePayload,
        ),
        ("finish_task", {"session_id": "s", "expected_revision": 3}, FinishTaskPayload),
        ("signal", {"target": SESSION_TARGET, "signal": "tool_wait"}, SignalPayload),
    ],
)
def test_native_payloads_preserve_supported_intent_through_json(
    action_type, fields, expected
):
    action = native_action(action_type, fields)
    wire_action = msgspec.json.decode(msgspec.json.encode(action), type=KvHintAction)
    payload = parse_native_action(wire_action)
    assert isinstance(payload, expected)
    assert payload == parse_native_action(action)


@pytest.mark.parametrize(
    "field,value",
    [
        ("priority", True),
        ("priority", -1),
        ("priority", 101),
        ("priority", "10"),
        ("ttl_seconds", 0),
        ("ttl_seconds", -1),
        ("ttl_seconds", float("inf")),
        ("ttl_seconds", float("nan")),
        ("scope", "implicit"),
        ("declaration_id", ""),
        ("unknown_field", 1),
        ("tier", "disk"),
    ],
)
def test_invalid_retention_cannot_silently_change_meaning(field, value):
    fields = {
        "target": SESSION_TARGET,
        "owner_id": "s",
        "declaration_id": "d",
        field: value,
    }
    with pytest.raises(ControlError) as error:
        parse_native_action(native_action("retain", fields))
    assert error.value.code == "invalid_payload"


@pytest.mark.parametrize(
    "target",
    [
        HASH_TARGET,
        {"kind": "session", "session_id": "s"},
        {"kind": "request", "request_id": "r"},
    ],
)
def test_follow_growth_cannot_implicitly_change_branches_or_request_lifetime(target):
    fields = {
        "target": target,
        "owner_id": "s",
        "declaration_id": "d",
        "scope": "follow_growth",
    }
    with pytest.raises(ControlError, match="explicit continuation"):
        parse_native_action(native_action("retain", fields))
    fields["target"] = SESSION_TARGET
    payload = parse_native_action(native_action("retain", fields))
    assert payload.scope == "follow_growth"


@pytest.mark.parametrize(
    "action_type,version,code",
    [
        ("pin", "1", "unsupported_action"),
        ("custom_rollout_handler", "1", "unsupported_action"),
        ("retain", "future", "unsupported_version"),
    ],
)
def test_native_unknown_actions_and_versions_are_explicit(action_type, version, code):
    with pytest.raises(ControlError) as error:
        parse_native_action(native_action(action_type, {}, version))
    assert error.value.code == code


@pytest.mark.parametrize(
    "field,value",
    [
        ("block_hash", "123"),
        ("block_hash", "zz"),
        ("group_id", -1),
        ("group_id", 1 << 32),
        ("end_token", 0),
        ("start_token", -1),
        ("block_id", 42),
    ],
)
def test_hash_selectors_reject_invalid_content_and_physical_addresses(field, value):
    block = {**HASH_TARGET["blocks"][0], field: value}
    target = {**HASH_TARGET, "blocks": [block]}
    with pytest.raises(ControlError) as error:
        parse_native_action(
            native_action("prefetch", {"target": target, "source": "cpu"})
        )
    assert error.value.code == "invalid_payload"


def test_group_ranges_are_independent_and_same_group_overlap_is_rejected():
    block = HASH_TARGET["blocks"][0]
    target = {**HASH_TARGET, "blocks": [block, {**block, "group_id": 1}]}
    payload = parse_native_action(
        native_action("prefetch", {"target": target, "source": "mooncake"})
    )
    assert isinstance(payload.target, HashTarget)
    assert len(payload.target.blocks) == 2
    for other in (block, {**block, "start_token": 8, "end_token": 24}):
        target = {**HASH_TARGET, "blocks": [block, other]}
        with pytest.raises(ControlError):
            parse_native_action(
                native_action("prefetch", {"target": target, "source": "mooncake"})
            )


@pytest.mark.parametrize("end", [0, 4])
def test_invalid_target_range_is_not_rounded_into_another_range(end):
    target = {**SESSION_TARGET, "start_token": 4, "end_token": end}
    with pytest.raises(ControlError, match="end must be greater"):
        parse_native_action(native_action("evict", {"target": target}))


def test_task_management_is_explicit_and_does_not_accept_policy_signal_schema():
    with pytest.raises(ControlError):
        parse_native_action(
            native_action(
                "finish_task", {"target": SESSION_TARGET, "signal": "tool_wait"}
            )
        )

    with pytest.raises(ControlError, match="parent itself"):
        parse_native_action(
            native_action(
                "associate",
                {
                    "session_id": "s",
                    "continuation_id": "c",
                    "expected_revision": 0,
                    "target": HASH_TARGET,
                    "task_parent_id": "s",
                },
            )
        )


@pytest.fixture
def controller(registry, now):
    index = SessionPrefixIndex("model-layout-weights", registry.engine_epoch)
    declarations = RetentionDeclarations(index, clock=lambda: now[0])
    return NativeKVController(registry, index, declarations)


def command(message_id, action_type, payload):
    return KvHintsEnvelope("0.1", message_id, [native_action(action_type, payload)])


def create_session(controller, session_id="s"):
    submission = controller.submit(
        command(
            "associate-" + session_id,
            "associate",
            {
                "session_id": session_id,
                "continuation_id": "c",
                "expected_revision": 0,
                "target": HASH_TARGET,
            },
        ),
        CONTEXT,
    )
    assert submission.results[0].status == OperationStatus.COMPLETED


def test_native_directives_manage_state_without_a_policy_or_inference(controller):
    create_session(controller)
    result = controller.submit(
        command(
            "retain",
            "retain",
            {
                "target": SESSION_TARGET,
                "owner_id": "s",
                "declaration_id": "d",
                "priority": 80,
            },
        ),
        CONTEXT,
    ).results[0]
    assert result.completed_steps == ("declaration_effective",)
    assert controller.declarations.priority(0, "0123") == 80
    released = controller.submit(
        command(
            "release",
            "release",
            {
                "owner_id": "s",
                "declaration_id": "d",
            },
        ),
        CONTEXT,
    ).results[0]
    assert released.status == OperationStatus.COMPLETED
    assert controller.declarations.priority(0, "0123") == 0
    finished = controller.submit(
        command(
            "finish",
            "finish_task",
            {
                "session_id": "s",
                "expected_revision": controller.index.session_revision("s"),
            },
        ),
        CONTEXT,
    ).results[0]
    assert finished.status == OperationStatus.COMPLETED
    assert controller.index.query().items == ()


def test_native_retry_does_not_renew_a_declaration(controller, now):
    create_session(controller)
    hint = command(
        "retain",
        "retain",
        {
            "target": SESSION_TARGET,
            "owner_id": "s",
            "declaration_id": "d",
            "priority": 80,
            "ttl_seconds": 5,
        },
    )
    first = controller.submit(hint, CONTEXT)
    now[0] += 4
    replay = controller.submit(hint, CONTEXT)
    assert replay.replayed
    assert replay.results == first.results
    now[0] += 1
    assert controller.declarations.get("s", "d") is None


def test_native_rewrite_conflict_is_correlated_and_replayed_without_effects(controller):
    create_session(controller)
    before = controller.index.query()
    hint = command(
        "rewrite",
        "associate",
        {
            "session_id": "s",
            "continuation_id": "c",
            "expected_revision": controller.index.get("s", "c").revision,
            "target": {**HASH_TARGET, "blocks": []},
        },
    )
    first = controller.submit(hint, CONTEXT)
    assert first.results[0].status == OperationStatus.FAILED
    assert "new continuation ID" in first.results[0].reason
    assert controller.index.query() == before
    retry = controller.submit(hint, CONTEXT)
    assert retry.replayed
    assert retry.results == first.results


def test_unconsumed_signal_does_not_even_create_an_unknown_association(controller):
    result = controller.submit(
        command(
            "signal",
            "signal",
            {
                "target": SESSION_TARGET,
                "signal": "tool_wait",
                "resume_after_seconds": 1,
            },
        ),
        CONTEXT,
    ).results[0]
    assert result.status == OperationStatus.UNCONSUMED
    assert controller.index.query().items == ()
    assert controller.index.events(0, epoch=CONTEXT.engine_epoch) == ()


def test_native_action_errors_are_correlated_without_rolling_back_earlier_actions(
    controller,
):
    actions = [
        native_action(
            "associate",
            {
                "session_id": "s",
                "continuation_id": "c",
                "expected_revision": 0,
                "target": HASH_TARGET,
            },
        ),
        native_action(
            "retain",
            {
                "target": SESSION_TARGET,
                "owner_id": "s",
                "declaration_id": "d",
                "priority": -1,
            },
        ),
        native_action("signal", {"target": SESSION_TARGET, "signal": "resume"}),
    ]
    hint = KvHintsEnvelope(
        "0.1",
        "multi",
        [msgspec.structs.replace(a, action_id=str(i)) for i, a in enumerate(actions)],
    )
    results = controller.submit(hint, CONTEXT).results
    assert [r.status for r in results] == [
        OperationStatus.COMPLETED,
        OperationStatus.FAILED,
        OperationStatus.UNCONSUMED,
    ]
    assert [r.action_id for r in results] == ["0", "1", "2"]
    assert controller.index.get("s", "c").blocks
    assert controller.declarations.get("s", "d") is None


def test_native_unknown_protocol_does_not_execute_valid_payload(controller):
    hint = msgspec.structs.replace(
        command(
            "unknown",
            "associate",
            {
                "session_id": "s",
                "continuation_id": "c",
                "expected_revision": 0,
                "target": HASH_TARGET,
            },
        ),
        protocol_version="future",
    )
    assert (
        controller.submit(hint, CONTEXT).results[0].status
        == OperationStatus.UNSUPPORTED
    )
    assert controller.index.query().items == ()


def test_backend_request_payload_is_left_to_the_existing_hook(controller):
    hint = command("opaque", "retain", {"backend_specific": [None, False, {"x": 17}]})
    hint = msgspec.structs.replace(hint, protocol_version="backend-version")
    wire = msgspec.json.encode(hint)
    assert (
        controller.consume_attached(
            hint, replace(CONTEXT, consumer="backend"), request_id="req"
        )
        is None
    )
    assert msgspec.json.encode(hint) == wire
    assert controller.registry.get("opaque", "action") is None
    assert controller.index.query().items == ()
    outcome = controller.submit(hint, replace(CONTEXT, consumer="backend")).results[0]
    assert outcome.status == OperationStatus.UNSUPPORTED
    assert "request-scoped" in outcome.reason


def test_attached_and_independent_request_targets_use_the_same_resolver(controller):
    calls = []

    def resolve(target, request_id):
        resolved = request_id if target.request_id == "$request" else target.request_id
        calls.append((target.request_id, request_id))
        if resolved != "active":
            raise ControlError("missing", "Request is not active")
        return msgspec.convert(HASH_TARGET, type=HashTarget)

    controller.request_resolver = resolve
    for session, target_id, attached in (
        ("attached", "$request", True),
        ("independent", "active", False),
    ):
        hint = command(
            session,
            "associate",
            {
                "session_id": session,
                "continuation_id": "c",
                "expected_revision": 0,
                "target": {"kind": "request", "request_id": target_id},
            },
        )
        result = (
            controller.consume_attached(hint, CONTEXT, request_id="active")
            if attached
            else controller.submit(hint, CONTEXT)
        )
        assert result.results[0].status == OperationStatus.COMPLETED
    assert calls == [("$request", "active"), ("active", None)]
    assert controller.index.members(0, "0123") == (
        ("attached", "c"),
        ("independent", "c"),
    )


def test_native_capabilities_do_not_claim_unconfigured_physical_execution(controller):
    caps = controller.capabilities()
    assert caps.engine_epoch == CONTEXT.engine_epoch
    assert {a.action_type for a in caps.actions} == {
        "retain",
        "release",
        "associate",
        "finish_task",
        "signal",
    }
    assert all("request" not in a.targets for a in caps.actions)
    result = controller.submit(
        command("prefetch", "prefetch", {"target": HASH_TARGET, "source": "cpu"}),
        CONTEXT,
    )
    assert result.results[0].status == OperationStatus.UNSUPPORTED


@pytest.mark.parametrize(
    "failure,status",
    [
        (ControlError("capacity", "No destination capacity"), OperationStatus.DEFERRED),
        (RuntimeError("Transfer backend failed"), OperationStatus.FAILED),
    ],
)
def test_execution_errors_are_not_reported_as_cancellation(controller, failure, status):
    class Executor:
        def submit(self, operation, payload, target):
            raise failure

    controller.executor = Executor()
    result = controller.submit(
        command("prefetch", "prefetch", {"target": HASH_TARGET, "source": "cpu"}),
        CONTEXT,
    )
    assert result.results[0].status == status


def test_approved_async_execution_is_not_reported_as_completed(controller):
    pending = []

    class Executor:
        def submit(self, operation, payload, target):
            pending.append((operation, target))
            controller.registry.update(operation, OperationStatus.ACCEPTED)

    controller.executor = Executor()
    result = controller.submit(
        command("prefetch", "prefetch", {"target": HASH_TARGET, "source": "cpu"}),
        CONTEXT,
    )
    assert result.results[0].status == OperationStatus.ACCEPTED
    assert result.results[0].completed_steps == ()
    assert pending[0][1].prefixes[0].blocks
    controller.registry.update(
        pending[0][0],
        OperationStatus.COMPLETED,
        completed_steps=("destination_registered",),
    )
    assert (
        controller.registry.get("prefetch", "action").status
        == OperationStatus.COMPLETED
    )


def test_failure_after_task_cleanup_reports_partial_effects(controller):
    class Executor:
        def cancel_owners(self, owners):
            assert owners == ("s",)
            raise ControlError("busy", "Worker interests could not be cancelled")

    create_session(controller)
    controller.executor = Executor()
    result = controller.submit(
        command(
            "finish",
            "finish_task",
            {
                "session_id": "s",
                "expected_revision": controller.index.session_revision("s"),
            },
        ),
        CONTEXT,
    ).results[0]
    assert result.status == OperationStatus.PARTIAL
    assert result.completed_steps == ("declarations_released", "associations_removed")
    assert controller.index.query().items == ()


def test_attached_retry_cannot_reinterpret_current_request_as_another_request(
    controller,
):
    controller.request_resolver = lambda target, request_id: msgspec.convert(
        HASH_TARGET, type=HashTarget
    )
    hint = command(
        "bound",
        "associate",
        {
            "session_id": "s",
            "continuation_id": "c",
            "expected_revision": 0,
            "target": {"kind": "request", "request_id": "$request"},
        },
    )
    first = controller.consume_attached(hint, CONTEXT, request_id="first")
    assert first.results[0].binding_id == "first"
    replay = controller.consume_attached(hint, CONTEXT, request_id="first")
    assert replay.replayed
    with pytest.raises(ControlError, match="different content"):
        controller.consume_attached(hint, CONTEXT, request_id="second")


def test_native_result_and_deduplication_are_bound_to_cache_compatibility(controller):
    hint = command(
        "signal", "signal", {"target": SESSION_TARGET, "signal": "tool_wait"}
    )
    first = controller.submit(hint, CONTEXT).results[0]
    assert first.namespace == "model-layout-weights"
    with pytest.raises(ControlError, match="another cache namespace"):
        controller.submit(hint, replace(CONTEXT, namespace="obsolete-weights"))
    new_index = SessionPrefixIndex("new-weights", CONTEXT.engine_epoch)
    updated = NativeKVController(
        controller.registry, new_index, RetentionDeclarations(new_index)
    )
    with pytest.raises(ControlError, match="different content"):
        updated.submit(hint, CONTEXT)


def test_utility_pagination_cursor_survives_json_conversion(controller):
    create_session(controller, "a")
    create_session(controller, "b")
    invocation = {"engine_id": CONTEXT.engine_id, "engine_epoch": CONTEXT.engine_epoch}
    service = KVControlService(controller)
    first = unwrap_control_reply(service.call("associations", invocation, {"limit": 1}))
    second = unwrap_control_reply(
        service.call(
            "associations",
            invocation,
            {"after": first["next_after"], "revision": first["revision"]},
        )
    )
    assert first["items"][0]["session_id"] == "a"
    assert second["items"][0]["session_id"] == "b"


def test_derived_retry_and_completion_cannot_change_origin(registry):
    parent = registry.submit(envelope("signal"), CONTEXT).results[0]
    origin = OperationOrigin(
        parent.message_id, parent.action_id, parent.operation_id, "policy"
    )
    context = replace(CONTEXT, origin=origin)
    child = registry.submit(envelope("derived"), context).results[0]
    assert child.origin == origin
    different = replace(origin, operation_id=origin.operation_id + 1)
    with pytest.raises(ControlError, match="different content"):
        registry.submit(envelope("derived"), replace(context, origin=different))
    with pytest.raises(ControlError, match="expired action"):
        registry.update(replace(child, origin=different), OperationStatus.COMPLETED)
    completed = registry.update(child, OperationStatus.COMPLETED)
    assert registry.submit(envelope("derived"), context).results == (completed,)
