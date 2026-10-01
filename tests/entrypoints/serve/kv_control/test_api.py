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
from vllm.v1.kv_hints import KvHintsEnvelope
from vllm.v1.kv_hints.api import KVControlService, unwrap_control_reply
from vllm.v1.kv_hints.control import OperationRegistry
from vllm.v1.kv_hints.controller import NativeKVController
from vllm.v1.kv_hints.retention import RetentionDeclarations
from vllm.v1.kv_hints.state import SessionPrefixIndex

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def control_client():
    registry = OperationRegistry("engine", engine_epoch="epoch")
    index = SessionPrefixIndex("namespace", registry.engine_epoch)
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


def test_http_submission_and_python_query_share_results_and_retries(control_client):
    client, service = control_client
    assert (
        client.get("/v1/kv/capabilities").json()["engines"][0]["engine_id"] == "engine"
    )
    envelope = {
        "protocol_version": "0.1",
        "message_id": "message",
        "actions": [
            {
                "action_id": "action",
                "action_type": "signal",
                "action_version": "1",
                "payload": {
                    "target": {"kind": "session", "session_id": "s"},
                    "signal": "tool_wait",
                },
            }
        ],
    }
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


def test_http_association_events_recover_from_gaps_without_residency_claims(
    control_client,
):
    client, service = control_client
    invocation = {
        "engine_id": "engine",
        "engine_epoch": "epoch",
        "namespace": "namespace",
    }
    envelope = {
        "protocol_version": "0.1",
        "message_id": "association",
        "actions": [
            {
                "action_id": "a",
                "action_type": "associate",
                "action_version": "1",
                "payload": {
                    "session_id": "s",
                    "continuation_id": "c",
                    "expected_revision": 0,
                    "target": {
                        "kind": "hashes",
                        "namespace": "namespace",
                        "blocks": [],
                    },
                },
            }
        ],
    }
    submitted = client.post(
        "/v1/kv/control",
        json={"method": "submit", "invocation": invocation, "payload": envelope},
    ).json()
    assert submitted["results"][0]["status"] == "completed"
    events = client.post(
        "/v1/kv/control",
        json={
            "method": "events",
            "invocation": invocation,
            "payload": {"stream": "association"},
        },
    ).json()
    assert {event["kind"] for event in events} == {"updated"}
    state = client.post(
        "/v1/kv/control",
        json={"method": "associations", "invocation": invocation, "payload": {}},
    ).json()
    assert state["items"][0]["session_id"] == "s"
    assert state["items"][0]["blocks"] == []
    stale = client.post(
        "/v1/kv/control",
        json={
            "method": "events",
            "invocation": {**invocation, "engine_epoch": "restarted"},
            "payload": {"stream": "association"},
        },
    )
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "stale_epoch"


@pytest.mark.parametrize(
    "invocation,status,code",
    [
        ({"engine_id": "other"}, 404, "wrong_engine"),
        ({"engine_epoch": "obsolete"}, 409, "stale_epoch"),
        ({"namespace": "old"}, 400, "incompatible"),
        ({"consumer": "backend"}, 501, "unsupported"),
    ],
)
def test_http_routing_errors_do_not_change_state(
    control_client, invocation, status, code
):
    client, service = control_client
    response = client.post(
        "/v1/kv/control", json=body("associations", {}, **invocation)
    )
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert service.controller.index.query().items == ()


@pytest.mark.parametrize(
    "payload", [{"limit": 0}, {"after": ["s", "c"]}, {"unknown": 1}]
)
def test_http_query_validation_is_the_engine_validation(control_client, payload):
    client, _ = control_client
    response = client.post("/v1/kv/control", json=body("associations", payload))
    assert response.status_code == 400


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
