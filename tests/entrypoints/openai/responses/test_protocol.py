# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
import json
import pathlib
import typing
from argparse import Namespace

import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator
from openai.types.responses import (
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseOutputText,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai_harmony import (
    Message,
)
from pydantic import ValidationError

from vllm.entrypoints.launchers.app import build_app
from vllm.entrypoints.openai.responses.protocol import (
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponsesRequest,
    ResponsesResponse,
    StreamingResponsesResponse,
    serialize_message,
    serialize_messages,
)
from vllm.entrypoints.serve.engine.protocol import ErrorInfo, ErrorResponse
from vllm.exceptions import VLLMValidationError

pytestmark = pytest.mark.skip_global_cleanup


def test_serialize_message() -> None:
    dict_value = {"a": 1, "b": "2"}
    assert serialize_message(dict_value) == dict_value

    msg_value = {
        "role": "assistant",
        "name": None,
        "content": [{"type": "text", "text": "Test 1"}],
        "channel": "analysis",
    }
    msg = Message.from_dict(msg_value)
    assert serialize_message(msg) == msg_value


def test_serialize_messages() -> None:
    assert serialize_messages(None) is None
    assert serialize_messages([]) is None

    dict_value = {"a": 3, "b": "4"}
    msg_value = {
        "role": "assistant",
        "name": None,
        "content": [{"type": "text", "text": "Test 2"}],
        "channel": "analysis",
    }
    msg = Message.from_dict(msg_value)
    assert serialize_messages([msg, dict_value]) == [msg_value, dict_value]


def test_normalize_openresponses_input_item_stamps_type() -> None:
    """A typeless assistant item must keep its role and content.

    `type` is optional in the OpenResponses schema, but filling in
    `id`/`status` without it makes the item match `ItemReference` in the
    input union, which collapses it to `{"id": ...}` and silently drops the
    assistant turn from the conversation. Regression test for that.
    """
    item = {
        "role": "assistant",
        "content": [{"type": "output_text", "text": "hi"}],
    }

    normalized = ResponsesRequest._normalize_openresponses_input_item(item)

    assert normalized["type"] == "message"
    assert normalized["role"] == "assistant"
    assert normalized["id"].startswith("msg_")
    assert normalized["status"] == "completed"
    # annotations are filled in for output_text content
    assert normalized["content"][0]["annotations"] == []
    # the caller's dict is not mutated
    assert "type" not in item


def test_normalize_openresponses_input_item_reasoning_type() -> None:
    """An explicit reasoning item keeps its type and gets a summary default."""
    normalized = ResponsesRequest._normalize_openresponses_input_item(
        {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "why"}]}
    )

    assert normalized["type"] == "reasoning"
    assert normalized["id"].startswith("rs_")
    assert normalized["status"] == "completed"
    assert normalized["summary"] == []


def test_input_item_parsing_typeless_assistant_message() -> None:
    """End to end: a typeless assistant item survives as a real message."""
    request = ResponsesRequest.model_validate(
        {
            "input": [
                {
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hi"}],
                }
            ]
        }
    )

    assert isinstance(request.input, list)
    (parsed,) = request.input
    assert isinstance(parsed, ResponseOutputMessage)
    assert parsed.role == "assistant"
    assert parsed.content[0].text == "hi"


def test_input_item_parsing_leaves_non_assistant_items_alone() -> None:
    """User messages are untouched — normalization only targets assistant items."""
    item = {"role": "user", "content": "hello"}
    request = ResponsesRequest.model_validate({"input": [item]})

    assert request.input == [item]


def test_input_list_validation_stops_at_first_invalid_item() -> None:
    valid_item = {"role": "user", "content": "hello"}
    invalid_item = {"type": "message", "role": "user"}
    with pytest.raises(ValidationError) as first:
        ResponsesRequest(input=[valid_item, invalid_item])
    with pytest.raises(ValidationError) as multiple:
        ResponsesRequest(input=[valid_item, invalid_item, invalid_item])

    assert multiple.value.error_count() == first.value.error_count()


def test_custom_tool_grammar_format_rejected() -> None:
    grammar = {"type": "grammar", "syntax": "lark", "definition": 'start: "pwd"'}
    with pytest.raises(VLLMValidationError) as exc_info:
        ResponsesRequest(
            input="hi", tools=[{"type": "custom", "name": "emit", "format": grammar}]
        )
    assert exc_info.value.parameter == "tools"

    request = ResponsesRequest(
        input="hi",
        tools=[{"type": "custom", "name": "emit", "format": {"type": "text"}}],
    )
    assert request.tools is not None and request.tools[0].type == "custom"


@pytest.mark.parametrize(
    "tools",
    [
        "abc",
        {"type": "custom"},
        [None],
        [{"type": "custom", "name": "x", "format": "text"}],
    ],
)
def test_malformed_tools_rejected_by_field_validation(tools) -> None:
    """Malformed tools bodies fail pydantic field validation (4xx), not an
    AttributeError inside the custom-format validator (500)."""
    with pytest.raises(ValidationError):
        ResponsesRequest(input="hi", tools=tools)


def _codex_additional_tools_item() -> dict:
    """Shape sent by Codex >= 0.169 as an input item instead of `tools`."""
    return {
        "type": "additional_tools",
        "id": "at_123",
        "role": "developer",
        "tools": [
            {
                "type": "namespace",
                "name": "clock",
                "description": "",
                "tools": [
                    {
                        "type": "function",
                        "name": "sleep",
                        "description": "Sleep for a while.",
                        "parameters": {"type": "object", "properties": {}},
                        "strict": False,
                    }
                ],
            }
        ],
    }


def test_additional_tools_item_hoisted_into_tools() -> None:
    top_level_tool = {
        "type": "function",
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {}},
        "strict": False,
    }
    user_message = {"role": "user", "content": "Reply with OK"}
    request = ResponsesRequest.model_validate(
        {
            "input": [_codex_additional_tools_item(), user_message],
            "tools": [top_level_tool],
        }
    )

    assert request.input == [user_message]
    assert [tool.type for tool in request.tools] == ["function", "namespace"]
    assert request.tools[0].name == "get_weather"
    assert request.tools[1].name == "clock"
    assert request.tools[1].tools[0].name == "sleep"


def test_additional_tools_items_merged_without_top_level_tools() -> None:
    first = _codex_additional_tools_item()
    second = _codex_additional_tools_item()
    second["tools"][0]["name"] = "timer"
    request = ResponsesRequest.model_validate(
        {"input": [first, {"role": "user", "content": "hi"}, second]}
    )

    assert len(request.input) == 1
    assert [tool.name for tool in request.tools] == ["clock", "timer"]
    # Tool-choice validation must see the hoisted tools, or "auto" becomes "none".
    assert request.tool_choice == "auto"


@pytest.mark.parametrize(
    "tool_choice,expected",
    [
        ("required", "required"),
        ({"type": "function", "name": "sleep"}, "clock__sleep"),
    ],
)
def test_additional_tools_satisfy_tool_choice(tool_choice, expected) -> None:
    request = ResponsesRequest.model_validate(
        {
            "input": [
                _codex_additional_tools_item(),
                {"role": "user", "content": "hi"},
            ],
            "tool_choice": tool_choice,
        }
    )

    resolved = request.tool_choice
    assert (resolved if isinstance(resolved, str) else resolved.name) == expected


def test_additional_tools_custom_grammar_rejected() -> None:
    grammar = {"type": "grammar", "syntax": "lark", "definition": 'start: "pwd"'}
    item = {
        "type": "additional_tools",
        "role": "developer",
        "tools": [{"type": "custom", "name": "emit", "format": grammar}],
    }
    with pytest.raises(VLLMValidationError) as exc_info:
        ResponsesRequest.model_validate({"input": [item]})
    assert exc_info.value.parameter == "tools"


def test_additional_tools_item_without_tools_list_rejected() -> None:
    item = {"type": "additional_tools", "role": "developer", "tools": "oops"}
    with pytest.raises(VLLMValidationError) as exc_info:
        ResponsesRequest.model_validate({"input": [item]})
    assert exc_info.value.parameter == "input"


def _function_tool(name: str, description: str = "") -> dict:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": {}},
    }


def test_additional_tools_deduplicated() -> None:
    """Repeated definitions are offered once; a namespace split across items
    is merged, so a named choice of a repeated namespaced tool is unambiguous."""
    clock = _codex_additional_tools_item()
    timer = _codex_additional_tools_item()
    timer["tools"][0]["tools"][0]["name"] = "timer"
    weather = _function_tool("get_weather")
    request = ResponsesRequest.model_validate(
        {
            "input": [
                clock,
                {**_codex_additional_tools_item(), "tools": [weather]},
                {"role": "user", "content": "hi"},
                _codex_additional_tools_item(),
                timer,
            ],
            "tools": [weather],
            "tool_choice": {"type": "function", "name": "sleep"},
        }
    )

    assert [tool.name for tool in request.tools] == ["get_weather", "clock"]
    assert [tool.name for tool in request.tools[1].tools] == ["sleep", "timer"]
    assert request.tool_choice.name == "clock__sleep"


def test_additional_tools_redefinition_last_wins_in_place() -> None:
    def item(*tools: dict) -> dict:
        return {"type": "additional_tools", "role": "developer", "tools": list(tools)}

    def clock(*tools: dict) -> dict:
        return {
            "type": "namespace",
            "name": "clock",
            "description": "",
            "tools": list(tools),
        }

    request = ResponsesRequest.model_validate(
        {
            "input": [
                item(clock(_function_tool("sleep", "v1"), _function_tool("timer"))),
                item(_function_tool("get_weather", "v2")),
                {"role": "user", "content": "hi"},
                item(clock(_function_tool("sleep", "v2"))),
            ],
            "tools": [_function_tool("get_weather", "v1"), _function_tool("get_time")],
        }
    )

    assert [tool.name for tool in request.tools] == ["get_weather", "get_time", "clock"]
    assert request.tools[0].description == "v2"
    assert [(t.name, t.description) for t in request.tools[2].tools] == [
        ("sleep", "v2"),
        ("timer", ""),
    ]


@pytest.mark.parametrize("with_item", [True, False])
@pytest.mark.parametrize("tools", [5, "abc", {"type": "function"}])
def test_non_list_tools_rejected_by_field_validation(tools, with_item) -> None:
    items = [_codex_additional_tools_item()] if with_item else []
    with pytest.raises(ValidationError) as exc_info:
        ResponsesRequest.model_validate(
            {"input": [*items, {"role": "user", "content": "hi"}], "tools": tools}
        )
    (error,) = exc_info.value.errors()
    assert error["loc"] == ("tools",) and error["type"] == "list_type"


def test_streaming_responses_response_covers_all_emitted_events() -> None:
    """StreamingResponsesResponse must include every event class emitted by
    streaming_events.py and serving.py."""
    import vllm.entrypoints.openai.responses.serving as serving_mod
    import vllm.entrypoints.openai.responses.streaming_events as events_mod

    emitted: set[str] = set()
    for mod in (events_mod, serving_mod):
        assert mod.__file__ is not None
        tree = ast.parse(pathlib.Path(mod.__file__).read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id.endswith("Event")
            ):
                emitted.add(node.func.id)

    union_names = {cls.__name__ for cls in typing.get_args(StreamingResponsesResponse)}
    assert emitted <= union_names


def _sample_responses_response() -> ResponsesResponse:
    return ResponsesResponse(
        id="resp_test_1",
        model="test-model",
        output=[
            ResponseOutputMessage(
                id="msg_1",
                type="message",
                role="assistant",
                content=[
                    ResponseOutputText(type="output_text", text="hi", annotations=[])
                ],
                status="completed",
            )
        ],
        parallel_tool_calls=True,
        temperature=0.0,
        tool_choice="auto",
        tools=[],
        top_p=1.0,
        background=False,
        max_output_tokens=16,
        service_tier="auto",
        status="completed",
        truncation="disabled",
    )


def _sample_streaming_events(response: ResponsesResponse) -> list:
    return [
        ResponseCreatedEvent(
            type="response.created",
            sequence_number=0,
            response=response,
        ),
        ResponseTextDeltaEvent(
            type="response.output_text.delta",
            sequence_number=1,
            content_index=0,
            output_index=0,
            item_id="msg_1",
            delta="hi",
            logprobs=[],
        ),
        ResponseTextDoneEvent(
            type="response.output_text.done",
            sequence_number=2,
            content_index=0,
            output_index=0,
            item_id="msg_1",
            text="hi",
            logprobs=[],
        ),
        ResponseFunctionCallArgumentsDeltaEvent(
            type="response.function_call_arguments.delta",
            sequence_number=3,
            output_index=0,
            item_id="fc_1",
            delta="{}",
        ),
        ResponseFunctionCallArgumentsDoneEvent(
            type="response.function_call_arguments.done",
            sequence_number=4,
            output_index=0,
            item_id="fc_1",
            arguments="{}",
            name="fn",
        ),
        ResponseCompletedEvent(
            type="response.completed",
            sequence_number=5,
            response=response,
        ),
    ]


def _parse_sse_frames(body: str) -> list[tuple[str | None, str]]:
    frames: list[tuple[str | None, str]] = []
    for raw_frame in body.strip().split("\n\n"):
        if not raw_frame.strip():
            continue
        event_name: str | None = None
        data_lines: list[str] = []
        for line in raw_frame.splitlines():
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
        frames.append((event_name, "\n".join(data_lines)))
    return frames


def test_responses_openapi_and_sse_conformance() -> None:
    """/v1/responses* OpenAPI contracts match actual ASGI JSON and SSE output."""
    args = Namespace(
        disable_fastapi_docs=False,
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
    sample_response = _sample_responses_response()
    sample_events = _sample_streaming_events(sample_response)

    class _StubServingResponses:
        async def create_responses(self, request, raw_request):
            if request.stream:

                async def _gen():
                    for event in sample_events:
                        yield event

                return _gen()
            return sample_response

        async def retrieve_responses(self, response_id, starting_after, stream):
            if response_id == "missing":
                return ErrorResponse(
                    error=ErrorInfo(
                        message="not found", type="invalid_request_error", code=404
                    )
                )
            if stream:

                async def _gen():
                    for event in sample_events:
                        yield event

                return _gen()
            return sample_response

        async def cancel_responses(self, response_id):
            return sample_response

    app.state.openai_serving_responses = _StubServingResponses()
    client = TestClient(app)

    document = client.get("/openapi.json").json()
    unary_ref = {"$ref": f"#/components/schemas/{ResponsesResponse.__name__}"}

    cancel_200 = document["paths"]["/v1/responses/{response_id}/cancel"]["post"][
        "responses"
    ]["200"]["content"]
    assert cancel_200["application/json"]["schema"] == unary_ref
    cancel_resp = client.post("/v1/responses/resp_test_1/cancel")
    assert cancel_resp.status_code == 200
    Draft202012Validator(
        {
            **cancel_200["application/json"]["schema"],
            "components": document["components"],
        }
    ).validate(cancel_resp.json())

    for path, method, call_fn in (
        (
            "/v1/responses",
            "post",
            lambda stream: client.post(
                "/v1/responses", json={"input": "hi", "stream": stream}
            ),
        ),
        (
            "/v1/responses/{response_id}",
            "get",
            lambda stream: client.get(
                f"/v1/responses/resp_test_1?stream={str(stream).lower()}"
            ),
        ),
    ):
        content = document["paths"][path][method]["responses"]["200"]["content"]
        assert content["application/json"]["schema"] == unary_ref

        unary_validator = Draft202012Validator(
            {
                **content["application/json"]["schema"],
                "components": document["components"],
            }
        )
        unary_http = call_fn(False)
        assert unary_http.status_code == 200
        unary_validator.validate(unary_http.json())

        data_prop = content["text/event-stream"]["schema"]["properties"]["data"]
        assert all("const" not in item for item in data_prop["anyOf"])
        chunk_validator = Draft202012Validator(
            {
                **data_prop["anyOf"][0]["contentSchema"],
                "components": document["components"],
            }
        )

        sse_http = call_fn(True)
        assert sse_http.status_code == 200
        assert sse_http.headers["content-type"].startswith("text/event-stream")
        frames = _parse_sse_frames(sse_http.text)
        assert len(frames) == len(sample_events)
        assert all(raw_data != "[DONE]" for _, raw_data in frames)
        for event_name, raw_data in frames:
            payload = json.loads(raw_data)
            assert event_name == payload["type"]
            chunk_validator.validate(payload)
        assert frames[-1][0] == "response.completed"

    not_found = client.get("/v1/responses/missing")
    assert not_found.status_code == 404
    err_404_schema = document["paths"]["/v1/responses/{response_id}"]["get"][
        "responses"
    ]["404"]["content"]["application/json"]["schema"]
    Draft202012Validator(
        {**err_404_schema, "components": document["components"]}
    ).validate(not_found.json())
