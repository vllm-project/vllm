# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from argparse import Namespace
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock

import pytest
import schemathesis
from fastapi import FastAPI
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, settings
from jsonschema import Draft202012Validator
from schemathesis import GenerationMode
from schemathesis.checks import not_a_server_error
from schemathesis.config import (
    ChecksConfig,
    CoveragePhaseConfig,
    GenerationConfig,
    PhasesConfig,
    PositiveDataAcceptanceConfig,
    ProjectConfig,
    ProjectsConfig,
    SchemathesisConfig,
)
from schemathesis.core.failures import FailureGroup

from vllm.platforms import current_platform

from ...utils import RemoteOpenAIServer

MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"
MAXIMUM_IMAGES = 2
_ROCM_TIMEOUT_MULTIPLIER = 3 if current_platform.is_rocm() else 1
DEFAULT_TIMEOUT_SECONDS: Final[int] = 10 * _ROCM_TIMEOUT_MULTIPLIER
LONG_TIMEOUT_SECONDS: Final[int] = 60 * _ROCM_TIMEOUT_MULTIPLIER


@pytest.mark.parametrize("offline", [False, True])
def test_server_openapi_version(offline: bool) -> None:
    """Production app construction must declare the version required by itemSchema.

    Exercise build_app with online and offline docs selected by offline. The
    completion_contract fixture sets 3.2 itself, so its tests cannot catch a
    missing production version setting that leaves /openapi.json declaring 3.1.
    The /docs assertion only checks page availability, not browser rendering;
    this test does not validate the whole document against the OpenAPI spec.
    """
    from vllm.entrypoints.launchers.app import build_app

    args = Namespace(
        disable_fastapi_docs=False,
        enable_offline_docs=offline,
        root_path="",
        allowed_origins=["*"],
        allow_credentials=False,
        allowed_methods=["*"],
        allowed_headers=["*"],
        api_key=None,
        enable_request_id_headers=False,
        enable_fault_tolerance=False,
        middleware=[],
        log_error_stack=False,
    )
    app = build_app(args, supported_tasks=())
    # No lifespan: schema generation does not need an inference engine.
    client = TestClient(app)
    assert client.get("/openapi.json").json()["openapi"] == "3.2.0"
    assert client.get("/docs").status_code == 200


@pytest.fixture(params=["chat_completion", "completion"])
def completion_contract(request):
    """Supply an engine-free contract fixture for request.param's completion API.

    Return (app, path, models, choices), with models and choices ordered as unary
    then streaming. The app uses the real router but sets OpenAPI 3.2 explicitly;
    test_server_openapi_version separately checks production app construction.
    """
    endpoint = request.param
    if endpoint == "chat_completion":
        from vllm.entrypoints.openai.chat_completion.api_router import router
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionResponse,
            ChatCompletionStreamResponse,
        )

        path = "/v1/chat/completions"
        models = (ChatCompletionResponse, ChatCompletionStreamResponse)
        choices = (
            {"index": 0, "message": {"role": "assistant", "content": "hello"}},
            {"index": 0, "delta": {"content": "hello"}},
        )
    else:
        from vllm.entrypoints.openai.completion.api_router import router
        from vllm.entrypoints.openai.completion.protocol import (
            CompletionResponse,
            CompletionStreamResponse,
        )

        path = "/v1/completions"
        models = (CompletionResponse, CompletionStreamResponse)
        choices = ({"index": 0, "text": "hello"},) * 2

    app = FastAPI()
    app.openapi_version = "3.2.0"
    app.include_router(router)
    return app, path, models, choices


def test_openapi_response_payloads(completion_contract) -> None:
    """Export the intended unary/chunk refs and constrain their decoded payloads.

    For each API supplied by completion_contract, inspect /openapi.json and
    validate serialized model dictionaries against the referenced schemas,
    rejecting malformed choices. For SSE, extract contentSchema and validate
    the decoded chunk directly: no streaming request or SSE parsing occurs.
    test_openapi_sse_events checks that the surrounding event declaration also
    lets an SSE-aware consumer reach those payload constraints.
    """
    app, path, models, choices = completion_contract
    with TestClient(app) as client:
        document = client.get("/openapi.json").json()
    operation = document["paths"][path]["post"]
    assert document["openapi"] == "3.2.0"
    content = operation["responses"]["200"]["content"]
    for media, model, choice in zip(
        ("application/json", "text/event-stream"), models, choices
    ):
        if media == "application/json":
            schema = content[media]["schema"]
        else:
            assert "schema" not in content[media]
            event = content[media]["itemSchema"]
            assert event["required"] == ["data"]
            data = event["properties"]["data"]
            assert data["type"] == "string"
            assert data["anyOf"][0] == {"const": "[DONE]"}
            schema = data["anyOf"][1]["contentSchema"]
        assert schema["$ref"] == f"#/components/schemas/{model.__name__}"
        validator = Draft202012Validator(
            {**schema, "components": document["components"]}
        )
        payload = model(model="test", choices=[choice], usage={}).model_dump(
            mode="json"
        )
        validator.validate(payload)
        payload["choices"] = "not an array"
        assert not validator.is_valid(payload)


@pytest.mark.parametrize("payload_kind", ["chunk", "error", "invalid", "non_json"])
def test_openapi_sse_events(completion_contract, payload_kind: str) -> None:
    """Make the exported SSE contract usable through actual route responses.

    Unlike direct payload validation, Schemathesis must parse SSE framing and
    validate the event envelope and its JSON data using /openapi.json. The real
    route wraps a mocked serving handler's stream: chunk/error payload_kind
    cases must pass, while invalid choices and non-JSON data must fail. Valid
    streams also exercise acceptance of [DONE] and keep-alive comments.
    This catches a chunk schema attached to the event envelope instead of its
    data. It does not test inference, production event generation, or require
    that production emits [DONE] last.
    """
    from vllm.entrypoints.serve.engine.protocol import ErrorInfo, ErrorResponse

    app, path, models, choices = completion_contract
    payload = models[1](model="test", choices=[choices[1]], usage={}).model_dump(
        mode="json"
    )
    if payload_kind == "error":
        payload = ErrorResponse(
            error=ErrorInfo(message="failed", type="InternalServerError", code=500)
        ).model_dump(mode="json")
    elif payload_kind == "invalid":
        payload["choices"] = "not an array"
    data = "not JSON" if payload_kind == "non_json" else json.dumps(payload)

    async def events():
        """Emit one sample, a keep-alive comment and the terminal marker."""
        yield f": keep-alive\n\ndata: {data}\n\ndata: [DONE]\n\n"

    chat = "chat" in path
    method = "create_chat_completion" if chat else "create_completion"
    serving = SimpleNamespace(**{method: AsyncMock(return_value=events())})
    setattr(
        app.state,
        "openai_serving_chat" if chat else "openai_serving_completion",
        serving,
    )
    body = {"model": "test", "stream": True}
    body.update(
        {"messages": [{"role": "user", "content": "hello"}]}
        if chat
        else {"prompt": "hello"}
    )
    schema = schemathesis.openapi.from_asgi("/openapi.json", app)
    case = schema[path]["POST"].Case(body=body, media_type="application/json")
    expectation = (
        pytest.raises(FailureGroup, match="SSE")
        if payload_kind in {"invalid", "non_json"}
        else nullcontext()
    )
    with expectation:
        case.call_and_validate()


@pytest.fixture(scope="module")
def server():
    args = [
        "--runner",
        "generate",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "5",
        "--enforce-eager",
        "--trust-remote-code",
        "--limit-mm-per-prompt",
        json.dumps({"image": MAXIMUM_IMAGES}),
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def get_schema(server):
    # avoid generating null (\x00) bytes in strings during test case generation
    return schemathesis.openapi.from_url(
        f"{server.url_root}/openapi.json",
        config=SchemathesisConfig(
            projects=ProjectsConfig(
                default=ProjectConfig(
                    generation=GenerationConfig(
                        allow_x00=False,
                        modes=[GenerationMode.POSITIVE],
                    ),
                    checks=ChecksConfig(
                        positive_data_acceptance=PositiveDataAcceptanceConfig(
                            enabled=False,
                        ),
                    ),
                    phases=PhasesConfig(
                        coverage=CoveragePhaseConfig(enabled=False),
                    ),
                ),
            ),
        ),
    )


schema = schemathesis.pytest.from_fixture("get_schema")


@schemathesis.hook
def before_generate_case(context: schemathesis.HookContext, strategy):
    op = context.operation
    assert op is not None

    def no_invalid_types(case: schemathesis.Case):
        """Skips tool_calls with `"type": "custom"` which schemathesis incorrectly
        generates instead of the valid `"type": "function"`.

        Example test case that is skipped:
        curl -X POST -H 'Content-Type: application/json' \
            -d '{"messages": [{"role": "assistant", "tool_calls": [{"custom": {"input": "", "name": ""}, "id": "", "type": "custom"}]}]}' \
            http://localhost:8000/v1/chat/completions
        """  # noqa: E501
        if (
            hasattr(case, "body")
            and isinstance(case.body, dict)
            and "messages" in case.body
            and isinstance(case.body["messages"], list)
            and len(case.body["messages"]) > 0
        ):
            for message in case.body["messages"]:
                if not isinstance(message, dict):
                    continue

                tool_calls = message.get("tool_calls", [])
                if isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        if isinstance(tool_call, dict):
                            if tool_call.get("type") != "function":
                                return False
                            if "custom" in tool_call:
                                return False

        return True

    return strategy.filter(no_invalid_types)


@schema.parametrize()
@settings(
    deadline=LONG_TIMEOUT_SECONDS * 1000,
    max_examples=50,
    # Under CI's derandomized hypothesis seed, the schemathesis strategy
    # for /v1/chat/completions/batch's nested-message body, combined with
    # the no_invalid_types filter (notably the grammar=="" rule), exceeds
    # the default filtered-vs-good ratio. The filter is intentional, so
    # suppress the health check rather than drop the filter — dropping it
    # exposes pre-existing server bugs out of scope here.
    # The same nested schema can also trip Hypothesis' entropy budget while
    # generating large-but-valid request bodies before vLLM is called.
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
)
def test_openapi_stateless(case: schemathesis.Case):
    key = (
        case.operation.method.upper(),
        case.operation.path,
    )
    if case.operation.path.startswith("/v1/responses"):
        # Skip responses API as it is meant to be stateful.
        return

    # Skip weight transfer endpoints as they require special setup
    # (weight_transfer_config) and are meant to be stateful.
    if case.operation.path in (
        "/init_weight_transfer_engine",
        "/start_weight_update",
        "/start_draft_weight_update",
        "/update_weights",
        "/finish_weight_update",
        "/update_weight_version",
    ):
        return

    timeout = {
        # requires a longer timeout
        ("POST", "/v1/chat/completions"): LONG_TIMEOUT_SECONDS,
        ("POST", "/v1/chat/completions/batch"): LONG_TIMEOUT_SECONDS,
        ("POST", "/v1/completions"): LONG_TIMEOUT_SECONDS,
        ("POST", "/v1/messages"): LONG_TIMEOUT_SECONDS,
        ("POST", "/inference/v1/generate"): LONG_TIMEOUT_SECONDS,
    }.get(key, DEFAULT_TIMEOUT_SECONDS)

    # No need to verify SSL certificate for localhost
    systemone = case.operation.path == "/v1/systemone"
    response = case.call_and_validate(
        verify=False,
        timeout=timeout,
        headers={"Content-Type": "application/json"},
        excluded_checks=[not_a_server_error] if systemone else None,
    )
    if systemone:
        # SmolVLM does not support structured decisions.
        assert response.status_code < 500 or response.status_code == 501
