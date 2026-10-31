# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
from openai_harmony import (
    Message,
)
from pydantic import ValidationError

from vllm.entrypoints.generate.base.protocol import FunctionCall
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
    ResponsesResponse,
    serialize_message,
    serialize_messages,
)
from vllm.entrypoints.openai.responses.utils import build_response_output_items
from vllm.exceptions import VLLMValidationError


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


def test_response_json_omits_unset_item_and_tool_fields() -> None:
    """Unset optional fields stay off the wire; what the client sent doesn't."""
    parameters = {"type": "object", "properties": {"a": {"default": None}}}
    request = ResponsesRequest(
        input="hi",
        tools=[
            {"type": "function", "name": "f", "parameters": parameters},
            {"type": "function", "name": "g", "strict": True, "description": None},
        ],
    )
    response = ResponsesResponse.from_request(
        request,
        request.to_sampling_params(default_max_tokens=8),
        model_name="m",
        created_time=0,
        output=build_response_output_items(
            reasoning="r",
            content=None,
            tool_calls=[FunctionCall(id="call_1", name="f", arguments="{}")],
            tools=request.tools,
        ),
        status="completed",
    )

    data = response.model_dump(mode="json", by_alias=True)

    assert data["output"][0].keys() == {"id", "summary", "type", "content"}
    assert data["output"][1].keys() == {
        "id",
        "call_id",
        "name",
        "arguments",
        "type",
        "status",
    }
    assert data["tools"] == [
        {"type": "function", "name": "f", "parameters": parameters, "strict": None},
        {
            "type": "function",
            "name": "g",
            "strict": True,
            "description": None,
            "parameters": None,
        },
    ]
    # Defaulted fields on the response itself are unaffected.
    assert data["object"] == "response"
