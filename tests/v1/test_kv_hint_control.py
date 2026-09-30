# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Control correlation is independent of native or backend payload semantics."""

from dataclasses import FrozenInstanceError, replace
from typing import Any

import msgspec
import pytest

from vllm.v1.kv_hints import KvHintAction, KvHintsEnvelope
from vllm.v1.kv_hints.control import (
    ControlContext,
    ControlError,
    OperationRegistry,
    OperationStatus,
)
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


@pytest.mark.parametrize("changed", ["payload", "order", "consumer", "version"])
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
