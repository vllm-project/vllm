# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers.abstract_tool_parser import ToolParser


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Adds two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        },
    }
]


def _make_request(
    *,
    tool_choice="auto",
    response_format=None,
) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "What is 42 + 58?"}],
        tools=TOOLS,
        tool_choice=tool_choice,
        response_format=response_format,
    )


def _make_parser() -> ToolParser:
    parser = ToolParser.__new__(ToolParser)
    parser.tools = TOOLS
    return parser


def test_auto_tool_choice_drops_response_format_constraint() -> None:
    request = _make_request(
        tool_choice="auto",
        response_format={"type": "json_object"},
    )

    _make_parser().adjust_request(request)

    assert request.response_format is None
    assert request.structured_outputs is None


def test_required_tool_choice_keeps_forced_tool_schema() -> None:
    request = _make_request(
        tool_choice="required",
        response_format={"type": "json_object"},
    )

    _make_parser().adjust_request(request)

    assert request.response_format is None
    assert request.structured_outputs is not None
    assert request.structured_outputs.json is not None


def test_tool_choice_none_keeps_response_format_constraint() -> None:
    request = _make_request(
        tool_choice="none",
        response_format={"type": "json_object"},
    )

    _make_parser().adjust_request(request)

    assert request.response_format is not None
    assert request.response_format.type == "json_object"
    assert request.structured_outputs is None
