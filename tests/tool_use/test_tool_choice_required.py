# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from copy import deepcopy

import pytest
import regex as re
from pydantic import TypeAdapter

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.parser.abstract_parser import _WrappedParser
from vllm.tool_parsers.abstract_tool_parser import ToolParser
from vllm.tool_parsers.streaming import extract_required_tool_call_streaming
from vllm.tool_parsers.utils import get_json_schema_from_tools

pytestmark = pytest.mark.cpu_test

EXAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to find the weather for"
                        ", e.g. 'San Francisco'",
                    },
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        },
        "strict": True,
    },
    {
        "type": "function",
        "function": {
            "name": "get_forecast",
            "description": "Get the weather forecast for a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to get the forecast for, e.g. "
                        "'New York'",
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days to get the forecast for (1-7)",
                    },
                },
                "required": ["city", "days"],
                "additionalProperties": False,
            },
        },
        "strict": True,
    },
]


def _compile_and_check(
    tools: list[ChatCompletionToolsParam], sample_output, should_match: bool
):
    # self = MagicMock(tool_choice="required", tools=tools)
    # schema = ChatCompletionRequest._get_json_schema_from_tool(self)
    schema = get_json_schema_from_tools(tools=tools, tool_choice="required")
    assert isinstance(schema, dict)

    # use build_regex_from_schema used in JSONLogitsProcessor to create Guide
    from outlines_core.json_schema import build_regex_from_schema

    regex = build_regex_from_schema(json.dumps(schema))
    compiled = re.compile(regex)
    matches = compiled.fullmatch(json.dumps(sample_output)) is not None

    assert matches == should_match


VALID_TOOL_OUTPUTS = [
    ([{"name": "get_current_weather", "parameters": {"city": "Vienna"}}], True),
    (
        [
            {"name": "get_current_weather", "parameters": {"city": "Vienna"}},
            {"name": "get_current_weather", "parameters": {"city": "Berlin"}},
        ],
        True,
    ),
    ([{"name": "get_forecast", "parameters": {"city": "Vienna", "days": 7}}], True),
    (
        [
            {"name": "get_forecast", "parameters": {"city": "Vienna", "days": 7}},
            {"name": "get_current_weather", "parameters": {"city": "Vienna"}},
        ],
        True,
    ),
    (
        [
            {"name": "get_forecast", "parameters": {"city": "Vienna", "days": 7}},
            {"name": "get_current_weather", "parameters": {"city": "Vienna"}},
            {"name": "get_forecast", "parameters": {"city": "Berlin", "days": 7}},
            {"name": "get_current_weather", "parameters": {"city": "Berlin"}},
        ],
        True,
    ),
]

VALID_TOOLS = [t[0] for t in VALID_TOOL_OUTPUTS]


@pytest.mark.parametrize(
    "sample_output, should_match",
    VALID_TOOL_OUTPUTS
    + [
        (None, False),
        ([], False),  # empty list cannot be generated
        ({}, False),  # empty object cannot be generated
        ([{}], False),  # list with empty object cannot be generated
        (
            [
                {  # function without required parameters cannot be generated
                    "name": "get_current_weather"
                }
            ],
            False,
        ),
        (
            [
                {  # function without required parameters cannot be generated
                    "name": "get_current_weather",
                    "parameters": {},
                }
            ],
            False,
        ),
        (
            [
                {  # function without required parameters cannot be generated
                    "name": "get_current_weather",
                    "parameters": None,
                }
            ],
            False,
        ),
        (
            {  # tool call without lists cannot be generated
                "name": "get_current_weather",
                "parameters": {"city": "Vienna"},
            },
            False,
        ),
        (
            [
                {  # tool call with extra parameters cannot be generated
                    "name": "get_current_weather",
                    "parameters": {"city": "Vienna", "extra": "value"},
                }
            ],
            False,
        ),
        (
            [
                {  # tool call where parameters are first cannot be generated
                    "parameters": {"city": "Vienna"},
                    "name": "get_current_weather",
                }
            ],
            False,
        ),
        (
            [
                {  # tool call without all required parameters cannot be generated
                    "name": "get_forecast",
                    "parameters": {"city": "Vienna"},
                }
            ],
            False,
        ),
        (  # tool call with incorrect name/parameters cannot be generated
            [{"name": "get_weather", "parameters": {"city": "Vienna", "days": 7}}],
            False,
        ),
        (  #  tool call with both valid and empty function cannot be generated
            [{"name": "get_current_weather", "parameters": {"city": "Vienna"}}, {}],
            False,
        ),
    ],
)
def test_structured_outputs_json(sample_output, should_match):
    _compile_and_check(
        tools=TypeAdapter(list[ChatCompletionToolsParam]).validate_python(
            EXAMPLE_TOOLS
        ),
        sample_output=sample_output,
        should_match=should_match,
    )


def update_parameters_none(tool: ChatCompletionToolsParam) -> ChatCompletionToolsParam:
    tool.function.parameters = None
    return tool


def update_parameters_empty_dict(
    tool: ChatCompletionToolsParam,
) -> ChatCompletionToolsParam:
    tool.function.parameters = {}
    return tool


@pytest.mark.parametrize(
    "sample_output, should_match",
    [
        (None, False),
        ([], False),  # empty list cannot be generated
        ({}, False),  # empty object cannot be generated
        ([{}], False),  # list with empty object cannot be generated
        (
            [
                {  # function without required parameters cannot be generated
                    "name": "get_current_weather"
                }
            ],
            False,
        ),
        (
            [
                {  # function without required parameters cannot be generated
                    "name": "get_current_weather",
                    "parameters": None,
                }
            ],
            False,
        ),
        (
            [
                {  # function with extra parameters cannot be generated
                    "name": "get_current_weather",
                    "parameters": {"extra": "value"},
                }
            ],
            False,
        ),
        (
            [
                {  # only function with empty parameters object is valid
                    "name": "get_current_weather",
                    "parameters": {},
                }
            ],
            True,
        ),
    ],
)
@pytest.mark.parametrize(
    "update_parameters", [update_parameters_none, update_parameters_empty_dict]
)
def test_structured_outputs_json_without_parameters(
    sample_output, should_match, update_parameters
):
    updated_tools = [deepcopy(EXAMPLE_TOOLS[0])]
    tools = TypeAdapter(list[ChatCompletionToolsParam]).validate_python(updated_tools)
    tools = list(map(update_parameters, tools))
    assert all(
        [
            tool.function.parameters is None or tool.function.parameters == {}
            for tool in tools
        ]
    )
    _compile_and_check(
        tools=tools, sample_output=sample_output, should_match=should_match
    )


@pytest.mark.parametrize("output", VALID_TOOLS)
@pytest.mark.parametrize("empty_params", [False, True])
@pytest.mark.parametrize("delta_len", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_streaming_output_valid(output, empty_params, delta_len):
    output = deepcopy(output)
    if empty_params:
        output = [{"name": o["name"], "parameters": {}} for o in output]
    output_json = json.dumps(output)

    previous_text = ""
    function_name_returned = False
    messages = []
    for i in range(0, len(output_json), delta_len):
        delta_text = output_json[i : i + delta_len]
        current_text = previous_text + delta_text

        delta_message, function_name_returned = extract_required_tool_call_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            function_name_returned=function_name_returned,
            tool_call_idx=None,
            tool_call_id_type="random",
        )

        if delta_message:
            messages.append(delta_message)

        previous_text = current_text

    assert len(messages) > 0

    combined_messages = "["
    for message in messages:
        if message.tool_calls[0].function.name:
            if len(combined_messages) > 1:
                combined_messages += "},"

            combined_messages += (
                '{"name": "'
                + message.tool_calls[0].function.name
                + '", "parameters": '
                + message.tool_calls[0].function.arguments
            )
        else:
            combined_messages += message.tool_calls[0].function.arguments
    combined_messages += "}]"
    assert json.loads(combined_messages) == output
    assert json.dumps(json.loads(combined_messages)) == output_json


@pytest.mark.parametrize("delta_len", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_streaming_output_emits_header_for_each_tool_call(delta_len):
    output = [
        {"name": "get_current_weather", "parameters": {"city": "Vienna"}},
        {"name": "get_forecast", "parameters": {"city": "Berlin", "days": 3}},
    ]
    output_json = json.dumps(output)

    previous_text = ""
    function_name_returned = False
    headers: dict[int, str] = {}
    arguments: dict[int, str] = {}

    for i in range(0, len(output_json), delta_len):
        delta_text = output_json[i : i + delta_len]
        current_text = previous_text + delta_text

        delta_message, function_name_returned = extract_required_tool_call_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            function_name_returned=function_name_returned,
            tool_call_idx=None,
            tool_call_id_type="random",
        )

        if delta_message:
            tool_call = delta_message.tool_calls[0]
            index = tool_call.index
            if tool_call.function.name:
                headers[index] = tool_call.function.name
            if tool_call.function.arguments:
                arguments[index] = (
                    arguments.get(index, "") + tool_call.function.arguments
                )

        previous_text = current_text

    assert headers == {0: "get_current_weather", 1: "get_forecast"}
    assert json.loads(arguments[0]) == output[0]["parameters"]
    assert json.loads(arguments[1]) == output[1]["parameters"]


class RequiredToolChoiceParser(ToolParser):
    supports_required_and_named = True


def test_required_tool_streaming_uses_parser_state_for_each_tool_call():
    old_reasoning_parser_cls = _WrappedParser.reasoning_parser_cls
    old_tool_parser_cls = _WrappedParser.tool_parser_cls
    _WrappedParser.reasoning_parser_cls = None
    _WrappedParser.tool_parser_cls = RequiredToolChoiceParser
    parser = _WrappedParser(tokenizer=object())
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "weather"}],
        tools=EXAMPLE_TOOLS,
        tool_choice="required",
    )
    output = [
        {"name": "get_current_weather", "parameters": {"city": "Vienna"}},
        {"name": "get_forecast", "parameters": {"city": "Berlin", "days": 3}},
    ]
    output_json = json.dumps(output)

    headers: dict[int, str] = {}
    arguments: dict[int, str] = {}
    prompt_token_ids: list[int] | None = []

    try:
        for i in range(0, len(output_json), 4):
            delta_text = output_json[i : i + 4]
            delta_message = parser.parse_delta(
                delta_text=delta_text,
                delta_token_ids=[i],
                request=request,
                prompt_token_ids=prompt_token_ids,
            )
            prompt_token_ids = None

            if delta_message and delta_message.tool_calls:
                tool_call = delta_message.tool_calls[0]
                index = tool_call.index
                if tool_call.function.name:
                    headers[index] = tool_call.function.name
                if tool_call.function.arguments:
                    arguments[index] = (
                        arguments.get(index, "") + tool_call.function.arguments
                    )
    finally:
        _WrappedParser.reasoning_parser_cls = old_reasoning_parser_cls
        _WrappedParser.tool_parser_cls = old_tool_parser_cls

    assert headers == {0: "get_current_weather", 1: "get_forecast"}
    assert json.loads(arguments[0]) == output[0]["parameters"]
    assert json.loads(arguments[1]) == output[1]["parameters"]


def test_streaming_output_valid_with_trailing_extra_data():
    output = [{"name": "get_current_weather", "parameters": {"city": "Vienna"}}]
    output_json = json.dumps(output) + "\nDONE"

    previous_text = ""
    function_name_returned = False
    messages = []
    delta_len = 3
    for i in range(0, len(output_json), delta_len):
        delta_text = output_json[i : i + delta_len]
        current_text = previous_text + delta_text

        delta_message, function_name_returned = extract_required_tool_call_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            function_name_returned=function_name_returned,
            tool_call_idx=None,
            tool_call_id_type="random",
        )

        if delta_message:
            messages.append(delta_message)

        previous_text = current_text

    assert len(messages) > 0
