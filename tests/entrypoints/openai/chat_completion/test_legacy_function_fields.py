# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Legacy `functions`/`function_call` requests must be rejected, not ignored.

The legacy Chat Completions fields are not part of `ChatCompletionRequest`,
so requests carrying them used to validate cleanly (they landed in
`model_extra`), the fields were dropped, and the server answered a tool
request with ordinary text. These tests pin the explicit 400 instead, at the
protocol level and through the real HTTP route, while confirming that modern
`tools`/`tool_choice` and the supported assistant-message
`function_call` field are untouched.
"""

from argparse import Namespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from vllm.entrypoints.launchers.app import build_app
from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
)
from vllm.entrypoints.serve.engine.protocol import UsageInfo
from vllm.exceptions import VLLMValidationError

_MODEL = "test-model"


def _request(**extra: Any) -> dict[str, Any]:
    return {
        "model": _MODEL,
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        **extra,
    }


@pytest.mark.parametrize(
    "extra",
    [
        {"functions": [{"name": "get_weather"}]},
        {"function_call": {"name": "get_weather"}},
        {"function_call": "required"},
        {
            "functions": [{"name": "get_weather"}],
            "function_call": {"name": "get_weather"},
        },
    ],
)
def test_legacy_fields_rejected_at_protocol_level(extra):
    with pytest.raises(VLLMValidationError):
        ChatCompletionRequest.model_validate(_request(**extra))


def test_legacy_functions_error_names_field_and_replacement():
    with pytest.raises(VLLMValidationError, match=r"parameter=functions") as excinfo:
        ChatCompletionRequest.model_validate(
            _request(functions=[{"name": "get_weather"}])
        )
    assert "`tools`" in str(excinfo.value)


def test_legacy_function_call_error_names_field_and_replacement():
    with pytest.raises(
        VLLMValidationError, match=r"parameter=function_call"
    ) as excinfo:
        ChatCompletionRequest.model_validate(
            _request(function_call={"name": "get_weather"})
        )
    assert "`tool_choice`" in str(excinfo.value)


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"functions": None},
        {"functions": []},
        {"function_call": None},
        {"function_call": "none"},
        {"function_call": "auto"},
        {"functions": [], "function_call": "none"},
    ],
)
def test_omitted_and_no_op_legacy_values_still_validate(extra):
    """OpenAI treats `[]`/`null` as omitted and `"auto"` as the default.
    `"none"` is only a no-op without `tools`; with `tools` it is a real
    "do not call tools" instruction and is covered separately."""
    ChatCompletionRequest.model_validate(_request(**extra))


def test_function_call_none_with_tools_rejected():
    """`function_call: "none"` suppresses tool calls, and dropping it would
    let the caller's explicit instruction through as `tool_choice="auto"`."""
    with pytest.raises(
        VLLMValidationError, match=r"parameter=function_call"
    ) as excinfo:
        ChatCompletionRequest.model_validate(
            _request(
                tools=[
                    {
                        "type": "function",
                        "function": {"name": "get_weather", "parameters": {}},
                    }
                ],
                function_call="none",
            )
        )
    assert "tool_choice" in str(excinfo.value)


def test_function_call_none_with_empty_tools_still_validates():
    ChatCompletionRequest.model_validate(_request(tools=[], function_call="none"))


def test_modern_tools_unaffected():
    request = ChatCompletionRequest.model_validate(
        _request(
            tools=[
                {
                    "type": "function",
                    "function": {"name": "get_weather", "parameters": {}},
                }
            ],
            tool_choice="auto",
        )
    )
    assert request.tools is not None
    assert request.tool_choice == "auto"


def test_assistant_message_function_call_still_supported():
    """`function_call` inside an assistant message is the supported
    round-trip field and is not the legacy top-level request parameter."""
    request = ChatCompletionRequest.model_validate(
        {
            "model": _MODEL,
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": None,
                    "function_call": {"name": "get_weather", "arguments": "{}"},
                },
            ],
        }
    )
    dumped = request.model_dump()
    assert dumped["messages"][1]["function_call"]["name"] == "get_weather"


def test_general_extra_fields_still_allowed():
    """The rejection is narrow: unknown, non-legacy fields keep working."""
    request = ChatCompletionRequest.model_validate(_request(provider_field="value"))
    assert request.model_extra is not None
    assert request.model_extra["provider_field"] == "value"


@pytest.mark.parametrize("field", ["functions", "function_call"])
def test_batch_request_rejects_legacy_fields(field):
    value = [{"name": "get_weather"}] if field == "functions" else "required"
    with pytest.raises(VLLMValidationError, match=f"parameter={field}"):
        BatchChatCompletionRequest.model_validate(
            {
                "model": _MODEL,
                "messages": [[{"role": "user", "content": "hi"}]],
                field: value,
            }
        )


def test_non_object_body_still_rejected():
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate("not an object")


class _StubServingChat:
    """Echoes a minimal valid response so the route can be exercised without a
    model; request validation happens before this is ever called."""

    async def create_chat_completion(self, request, raw_request):
        return ChatCompletionResponse(
            model=_MODEL,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="ok"),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(prompt_tokens=1, total_tokens=2, completion_tokens=1),
        )


@pytest.fixture(scope="module")
def client():
    args = Namespace(
        disable_fastapi_docs=True,
        enable_offline_docs=False,
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
    app = build_app(args, supported_tasks=("generate",))
    app.state.openai_serving_chat = _StubServingChat()
    # No context manager: skipping the lifespan avoids needing server state
    # (log_stats, metrics) unrelated to request validation.
    yield TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize(
    "extra, param",
    [
        ({"functions": [{"name": "get_weather"}]}, "functions"),
        ({"function_call": {"name": "get_weather"}}, "function_call"),
        ({"function_call": "required"}, "function_call"),
        (
            {
                "functions": [{"name": "get_weather"}],
                "function_call": {"name": "get_weather"},
            },
            "functions",
        ),
        (
            {
                "tools": [{"type": "function", "function": {"name": "get_weather"}}],
                "function_call": "none",
            },
            "function_call",
        ),
    ],
)
def test_route_rejects_legacy_fields(client, extra, param):
    response = client.post("/v1/chat/completions", json=_request(**extra))
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["param"] == param
    assert "legacy" in error["message"]


def test_route_accepts_no_op_legacy_values(client):
    response = client.post(
        "/v1/chat/completions", json=_request(functions=[], function_call="none")
    )
    assert response.status_code == 200


def test_route_accepts_modern_tools(client):
    response = client.post(
        "/v1/chat/completions",
        json=_request(
            tools=[
                {
                    "type": "function",
                    "function": {"name": "get_weather", "parameters": {}},
                }
            ],
            tool_choice="auto",
        ),
    )
    assert response.status_code == 200
