# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from dataclasses import dataclass, field

import pytest

from vllm.v1.kv_hints.actions import KvHintResult, SetPriority
from vllm.v1.kv_hints.dispatch import KvHintDispatcher
from vllm.v1.kv_hints.protocol import KvHintAction, KvHintsEnvelope

SET_PRIORITY_PAYLOAD = {
    "target": "request_cached",
    "location": "local_g1",
    "claim_id": "router/claim",
    "revision": 1,
    "value": 10,
    "ttl_seconds": 60,
}


@dataclass
class FakeRequest:
    request_id: str
    kv_hints: KvHintsEnvelope | None = None
    seen: list = field(default_factory=list)


def make_action(
    action_id: str,
    action_type: str = "vllm.set_priority",
    action_version: str = "1.0",
    payload: dict | None = None,
) -> KvHintAction:
    return KvHintAction(
        action_id,
        action_type,
        action_version,
        SET_PRIORITY_PAYLOAD if payload is None else payload,
    )


def make_envelope(actions: list[KvHintAction], version: str = "0.1"):
    return KvHintsEnvelope(version, "router/message", actions)


def noop(payload: SetPriority, request: FakeRequest) -> KvHintResult:
    request.seen.append(payload)
    return KvHintResult("applied")


def test_duplicate_registration_rejected():
    dispatcher = KvHintDispatcher()
    dispatcher.register("vllm.set_priority", "1.0", SetPriority, noop, when="ingress")
    with pytest.raises(ValueError, match="already registered"):
        dispatcher.register(
            "vllm.set_priority", "1.0", SetPriority, noop, when="ingress"
        )


def test_no_hints_dispatches_nothing():
    assert KvHintDispatcher().dispatch(FakeRequest("r1", None)) == []


def test_backend_owned_actions_left_untouched():
    dispatcher = KvHintDispatcher()
    dispatcher.register("vllm.set_priority", "1.0", SetPriority, noop, when="ingress")
    request = FakeRequest(
        "r1",
        make_envelope([make_action("backend", "kv.fetch", "1.0", {"opaque": True})]),
    )
    assert dispatcher.dispatch(request) == []
    assert request.seen == []


@pytest.mark.parametrize(
    "action",
    [
        make_action("a1", "vllm.unknown", "1.0", SET_PRIORITY_PAYLOAD),
        make_action("a1", "vllm.set_priority", "2.0", SET_PRIORITY_PAYLOAD),
    ],
)
def test_unregistered_action_is_skipped(action):
    dispatcher = KvHintDispatcher()
    dispatcher.register("vllm.set_priority", "1.0", SetPriority, noop, when="ingress")
    request = FakeRequest("r1", make_envelope([action]))
    results = dispatcher.dispatch(request)
    assert [r.status for _, r in results] == ["unsupported"]
    assert request.seen == []


def test_wrong_protocol_version_skips_even_registered_action():
    dispatcher = KvHintDispatcher()
    dispatcher.register("vllm.set_priority", "1.0", SetPriority, noop, when="ingress")
    request = FakeRequest("r1", make_envelope([make_action("a1")], version="0.2"))
    results = dispatcher.dispatch(request)
    assert [r.status for _, r in results] == ["unsupported"]
    assert request.seen == []


@pytest.mark.parametrize(
    "payload",
    [
        {k: v for k, v in SET_PRIORITY_PAYLOAD.items() if k != "claim_id"},
        {**SET_PRIORITY_PAYLOAD, "unknown": 1},
        {**SET_PRIORITY_PAYLOAD, "value": None, "ttl_seconds": 10},
    ],
)
def test_bad_payload_rejected_without_failing_request(payload, caplog):
    dispatcher = KvHintDispatcher()
    dispatcher.register("vllm.set_priority", "1.0", SetPriority, noop, when="ingress")
    request = FakeRequest("r1", make_envelope([make_action("a1", payload=payload)]))
    with caplog.at_level(logging.WARNING, logger="vllm.v1.kv_hints.dispatch"):
        results = dispatcher.dispatch(request)
    assert [r.status for _, r in results] == ["rejected"]
    assert "Rejected KV hint" in caplog.text
    assert request.seen == []


def test_ingress_action_executes_immediately():
    dispatcher = KvHintDispatcher()
    dispatcher.register("vllm.set_priority", "1.0", SetPriority, noop, when="ingress")
    request = FakeRequest("r1", make_envelope([make_action("a1")]))
    results = dispatcher.dispatch(request)
    assert [r.status for _, r in results] == ["applied"]
    assert isinstance(request.seen[0], SetPriority)
    assert request.seen[0].claim_id == "router/claim"


def test_completion_action_deferred_until_successful_finish():
    dispatcher = KvHintDispatcher()
    dispatcher.register(
        "vllm.set_priority",
        "1.0",
        SetPriority,
        noop,
        when="successful_completion",
    )
    request = FakeRequest("r1", make_envelope([make_action("a1")]))
    assert [r.status for _, r in dispatcher.dispatch(request)] == ["deferred"]
    assert request.seen == []
    results = dispatcher.finish(request, successful=True)
    assert [r.status for _, r in results] == ["applied"]
    assert len(request.seen) == 1


def test_unsuccessful_finish_discards_deferred_actions():
    dispatcher = KvHintDispatcher()
    dispatcher.register(
        "vllm.set_priority", "1.0", SetPriority, noop, when="successful_completion"
    )
    request = FakeRequest("r1", make_envelope([make_action("a1")]))
    dispatcher.dispatch(request)
    assert dispatcher.finish(request, successful=False) == []
    assert request.seen == []
    assert dispatcher.finish(request, successful=True) == []


def test_finish_is_idempotent_after_pop():
    dispatcher = KvHintDispatcher()
    dispatcher.register(
        "vllm.set_priority", "1.0", SetPriority, noop, when="successful_completion"
    )
    request = FakeRequest("r1", make_envelope([make_action("a1")]))
    dispatcher.dispatch(request)
    dispatcher.finish(request, successful=True)
    assert dispatcher.finish(request, successful=True) == []
    assert len(request.seen) == 1


def test_pending_actions_isolated_per_request():
    dispatcher = KvHintDispatcher()
    dispatcher.register(
        "vllm.set_priority", "1.0", SetPriority, noop, when="successful_completion"
    )
    first = FakeRequest("r1", make_envelope([make_action("a1")]))
    second = FakeRequest("r2", make_envelope([make_action("a2")]))
    dispatcher.dispatch(first)
    dispatcher.dispatch(second)
    dispatcher.finish(first, successful=True)
    assert len(first.seen) == 1 and second.seen == []
    dispatcher.finish(second, successful=True)
    assert len(second.seen) == 1


def test_valid_sibling_survives_invalid_action():
    dispatcher = KvHintDispatcher()
    dispatcher.register("vllm.set_priority", "1.0", SetPriority, noop, when="ingress")
    request = FakeRequest(
        "r1",
        make_envelope(
            [
                make_action("backend", "kv.fetch", "1.0", {"opaque": True}),
                make_action("invalid", payload={}),
                make_action("valid"),
            ]
        ),
    )
    results = dispatcher.dispatch(request)
    assert [r.status for _, r in results] == ["rejected", "applied"]
    assert len(request.seen) == 1


def test_empty_envelope_dispatches_nothing():
    dispatcher = KvHintDispatcher()
    assert dispatcher.dispatch(FakeRequest("r1", make_envelope([]))) == []
