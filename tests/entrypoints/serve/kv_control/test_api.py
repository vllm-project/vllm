# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HTTP and Python must observe the same correlated engine control state."""

from types import SimpleNamespace

import msgspec
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from vllm.config import KVControlConfig
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.serve.kv_control.api_router import attach_router
from vllm.v1.kv_hints import KvHintAction, KvHintsEnvelope
from vllm.v1.kv_hints.api import KVControlService, unwrap_control_reply
from vllm.v1.kv_hints.control import OperationRegistry
from vllm.v1.kv_hints.controller import NativeKVController
from vllm.v1.kv_hints.retention import RetentionDeclarations
from vllm.v1.kv_hints.state import SessionPrefixIndex

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def control_client():
    registry = OperationRegistry("engine", engine_epoch="epoch")
    index = SessionPrefixIndex("namespace", registry.engine_epoch, max_events=2)
    service = KVControlService(
        NativeKVController(registry, index, RetentionDeclarations(index))
    )

    class Client:
        async def get_kv_control_capabilities(self):
            return [msgspec.to_builtins(service.controller.capabilities())]

        async def kv_control(self, method, invocation, payload):
            return unwrap_control_reply(service.call(method, invocation, payload))

    app = FastAPI()
    app.state.engine_client = Client()
    app.state.vllm_config = SimpleNamespace(kv_control_config=KVControlConfig())
    attach_router(app)
    with TestClient(app) as client:
        yield client, service


def body(method, payload, **invocation):
    return {
        "method": method,
        "invocation": {"engine_id": "engine", "engine_epoch": "epoch", **invocation},
        "payload": payload,
    }


def hint(message, kind, payload):
    return msgspec.to_builtins(
        KvHintsEnvelope("0.1", message, [KvHintAction("action", kind, "1", payload)])
    )


def test_http_submission_and_python_query_share_results_and_retries(control_client):
    client, service = control_client
    assert (
        client.get("/v1/kv/capabilities").json()["engines"][0]["engine_id"] == "engine"
    )
    envelope = hint(
        "message",
        "signal",
        {
            "target": {"kind": "session", "session_id": "s"},
            "signal": "tool_wait",
        },
    )
    response = client.post("/v1/kv/control", json=body("submit", envelope))
    assert response.status_code == 200
    submitted = response.json()
    query = unwrap_control_reply(
        service.call(
            "operation",
            body("operation", {})["invocation"],
            {"message_id": "message", "action_id": "action"},
        )
    )
    assert query == submitted["results"][0]
    assert query["status"] == "unconsumed"
    assert client.post("/v1/kv/control", json=body("submit", envelope)).json()[
        "replayed"
    ]
    events = client.post(
        "/v1/kv/control", json=body("events", {"stream": "operation"})
    ).json()
    assert events[-1] == query
    acknowledged = client.post(
        "/v1/kv/control", json=body("acknowledge", {"message_id": "message"})
    )
    assert acknowledged.json() == {"acknowledged": True}
    retry = client.post("/v1/kv/control", json=body("submit", envelope))
    assert retry.status_code == 409
    assert retry.json()["error"]["code"] == "acknowledged"
    capacity = client.post("/v1/kv/control", json=body("capacity", {})).json()
    assert capacity["records"] == 0
    assert capacity["acknowledged_messages"] == 1


def test_http_association_snapshot_supplies_cursor_after_real_history_loss(
    control_client,
):
    client, service = control_client

    def associate(session):
        response = client.post(
            "/v1/kv/control",
            json=body(
                "submit",
                hint(
                    session,
                    "associate",
                    {
                        "session_id": session,
                        "continuation_id": "c",
                        "expected_revision": 0,
                        "target": {
                            "kind": "hashes",
                            "namespace": "namespace",
                            "blocks": [
                                {
                                    "block_hash": "abcd",
                                    "group_id": 0,
                                    "start_token": 0,
                                    "end_token": 16,
                                }
                            ],
                        },
                    },
                ),
            ),
        )
        assert response.status_code == 200
        assert response.json()["results"][0]["status"] == "completed"

    associate("a")
    associate("b")
    gap = client.post("/v1/kv/control", json=body("events", {"stream": "association"}))
    assert gap.status_code == 409
    assert gap.json()["error"]["code"] == "event_gap"
    snapshot = client.post("/v1/kv/control", json=body("associations", {})).json()
    assert [item["session_id"] for item in snapshot["items"]] == ["a", "b"]
    assert snapshot["engine_epoch"] == "epoch"
    assert snapshot["namespace"] == "namespace"
    associate("c")
    deltas = client.post(
        "/v1/kv/control",
        json=body(
            "events",
            {
                "stream": "association",
                "after": snapshot["event_sequence"],
            },
        ),
    ).json()
    assert [event["kind"] for event in deltas] == ["associated", "updated"]
    assert {event["session_id"] for event in deltas} == {"c"}
    assert service.controller.index.members(0, "abcd") == (
        ("a", "c"),
        ("b", "c"),
        ("c", "c"),
    )


@pytest.mark.parametrize(
    "invocation,method,payload,status,code",
    [
        ({"engine_id": "other"}, "associations", {}, 404, "wrong_engine"),
        ({"engine_epoch": "obsolete"}, "associations", {}, 409, "stale_epoch"),
        ({"namespace": "old"}, "associations", {}, 400, "incompatible"),
        ({"consumer": "backend"}, "associations", {}, 501, "unsupported"),
        ({"binding_id": "inference"}, "associations", {}, 400, "invalid_payload"),
        ({}, "associations", {"limit": 0}, 400, "invalid_payload"),
        ({}, "associations", {"limit": True}, 400, "invalid_payload"),
        ({}, "associations", {"after": ["s", "c"]}, 400, "invalid_cursor"),
        ({}, "associations", {"unknown": 1}, 400, "invalid_payload"),
        ({}, "events", {"stream": "operation", "after": 100}, 409, "event_gap"),
        ({}, "unknown", {}, 400, "invalid_payload"),
        ({}, "operation", {"message_id": "absent", "action_id": "a"}, 404, "missing"),
    ],
)
def test_http_control_errors_preserve_engine_state(
    control_client, invocation, method, payload, status, code
):
    client, service = control_client
    response = client.post("/v1/kv/control", json=body(method, payload, **invocation))
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    if method == "unknown":
        assert (
            service.call(method, body(method, {})["invocation"], payload)["error"][
                "code"
            ]
            == "unsupported"
        )
    assert service.controller.index.query().items == ()


def test_http_rejects_invalid_json_unknown_fields_and_oversized_body(control_client):
    client, service = control_client
    invalid = client.post(
        "/v1/kv/control", content="{", headers={"Content-Type": "application/json"}
    )
    assert invalid.status_code == 400
    unknown = client.post(
        "/v1/kv/control", json={**body("associations", {}), "extra": 1}
    )
    assert unknown.status_code == 400
    client.app.state.vllm_config.kv_control_config.max_envelope_bytes = 32
    oversized = client.post(
        "/v1/kv/control",
        content=b" " * 2081,
        headers={"Content-Type": "application/json"},
    )
    assert oversized.status_code == 413
    assert service.controller.index.query().items == ()


def test_http_disabled_feature_reports_unsupported(control_client):
    client, _ = control_client
    client.app.state.vllm_config.kv_control_config = None
    response = client.post("/v1/kv/control", json=body("associations", {}))
    assert response.status_code == 501


@pytest.mark.parametrize(
    "request_type,fields",
    [
        (CompletionRequest, {"prompt": "hello"}),
        (ChatCompletionRequest, {"messages": [{"role": "user", "content": "hello"}]}),
    ],
)
def test_inference_protocol_preserves_the_merged_opaque_envelope(request_type, fields):
    hints = {
        "protocol_version": "backend-version",
        "message_id": "m",
        "actions": [
            {
                "action_id": "a",
                "action_type": "opaque",
                "action_version": "future",
                "payload": {"values": [None, True, 0.5, {"text": "值"}]},
            }
        ],
    }
    request = request_type(**fields, kv_hints=hints)
    assert isinstance(request.kv_hints, KvHintsEnvelope)
    assert request.model_dump()["kv_hints"] == hints
    assert request.model_json_schema()["properties"]["kv_hints"]
    for options in ({"n": 2}, {"use_beam_search": True}):
        with pytest.raises(ValidationError, match="n=1"):
            request_type(**fields, kv_hints=hints, **options)
    with pytest.raises(ValidationError, match="message_id"):
        request_type(**fields, kv_hints={**hints, "message_id": ""})
