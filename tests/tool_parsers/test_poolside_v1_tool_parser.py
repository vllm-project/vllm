# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from unittest.mock import Mock

import pytest

from tests.tool_parsers.utils import run_tool_extraction
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParserManager

# PoolsideV1ToolParser.extract_tool_calls_streaming short-circuits to content
# pass-through when request.tools is None. All tests pass a mock_request so
# the streaming path activates; non-streaming ignores request.tools entirely.


@pytest.fixture
def mock_request():
    request = Mock(spec=ChatCompletionRequest)
    request.tools = [
        ChatCompletionToolsParam(function=FunctionDefinition(name="get_weather")),
        ChatCompletionToolsParam(function=FunctionDefinition(name="get_time")),
    ]
    request.logprobs = None
    return request


# Poolside format:
# <tool_call>func_name\n<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>

NO_TOOL_OUTPUT = "How can I help you today?"

SINGLE_TOOL_OUTPUT = (
    "<tool_call>get_weather\n"
    "<arg_key>city</arg_key><arg_value>Tokyo</arg_value>"
    "</tool_call>"
)

PARALLEL_TOOLS_OUTPUT = (
    "<tool_call>get_weather\n"
    "<arg_key>city</arg_key><arg_value>Tokyo</arg_value>"
    "</tool_call>"
    "<tool_call>get_time\n"
    "<arg_key>city</arg_key><arg_value>London</arg_value>"
    "</tool_call>"
)

EMPTY_ARGS_OUTPUT = "<tool_call>get_weather\n</tool_call>"

SURROUNDING_TEXT_OUTPUT = (
    "Sure, let me check the weather.\n"
    "<tool_call>get_weather\n"
    "<arg_key>city</arg_key><arg_value>Tokyo</arg_value>"
    "</tool_call>"
)

VARIOUS_TYPES_OUTPUT = (
    "<tool_call>register_entity\n"
    "<arg_key>string_field</arg_key><arg_value>hello</arg_value>"
    "<arg_key>int_field</arg_key><arg_value>42</arg_value>"
    "<arg_key>float_field</arg_key><arg_value>3.14</arg_value>"
    "<arg_key>bool_field</arg_key><arg_value>true</arg_value>"
    "<arg_key>null_field</arg_key><arg_value>null</arg_value>"
    '<arg_key>array_field</arg_key><arg_value>["a", "b"]</arg_value>'
    '<arg_key>object_field</arg_key><arg_value>{"key": "val"}</arg_value>'
    "</tool_call>"
)

MALFORMED_INPUTS = [
    "<tool_call>unclosed with no end tag",
    "<tool_call>func\n<arg_key>city</arg_key>missing value tag</tool_call>",
]


@pytest.mark.parametrize("streaming", [True, False])
def test_no_tool_calls(
    streaming: bool,
    default_tokenizer: TokenizerLike,
    mock_request: Mock,
) -> None:
    tool_parser = ToolParserManager.get_tool_parser("poolside_v1")(default_tokenizer)
    content, tool_calls = run_tool_extraction(
        tool_parser, NO_TOOL_OUTPUT, request=mock_request, streaming=streaming
    )
    assert content == NO_TOOL_OUTPUT
    assert len(tool_calls) == 0


@pytest.mark.parametrize("streaming", [True, False])
def test_single_tool_call(
    streaming: bool,
    default_tokenizer: TokenizerLike,
    mock_request: Mock,
) -> None:
    tool_parser = ToolParserManager.get_tool_parser("poolside_v1")(default_tokenizer)
    content, tool_calls = run_tool_extraction(
        tool_parser, SINGLE_TOOL_OUTPUT, request=mock_request, streaming=streaming
    )
    assert len(tool_calls) == 1
    assert tool_calls[0].type == "function"
    assert tool_calls[0].function.name == "get_weather"
    args = json.loads(tool_calls[0].function.arguments)
    assert args["city"] == "Tokyo"


@pytest.mark.parametrize("streaming", [True, False])
def test_parallel_tool_calls(
    streaming: bool,
    default_tokenizer: TokenizerLike,
    mock_request: Mock,
) -> None:
    tool_parser = ToolParserManager.get_tool_parser("poolside_v1")(default_tokenizer)
    content, tool_calls = run_tool_extraction(
        tool_parser,
        PARALLEL_TOOLS_OUTPUT,
        request=mock_request,
        streaming=streaming,
        assert_one_tool_per_delta=not streaming,
    )
    assert len(tool_calls) == 2
    assert tool_calls[0].function.name == "get_weather"
    assert tool_calls[1].function.name == "get_time"
    assert tool_calls[0].id != tool_calls[1].id
    assert json.loads(tool_calls[0].function.arguments)["city"] == "Tokyo"
    assert json.loads(tool_calls[1].function.arguments)["city"] == "London"


@pytest.mark.parametrize("streaming", [True, False])
def test_empty_arguments(
    streaming: bool,
    default_tokenizer: TokenizerLike,
    mock_request: Mock,
) -> None:
    tool_parser = ToolParserManager.get_tool_parser("poolside_v1")(default_tokenizer)
    content, tool_calls = run_tool_extraction(
        tool_parser, EMPTY_ARGS_OUTPUT, request=mock_request, streaming=streaming
    )
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "get_weather"
    assert tool_calls[0].function.arguments == "{}"


@pytest.mark.parametrize("streaming", [True, False])
def test_surrounding_text(
    streaming: bool,
    default_tokenizer: TokenizerLike,
    mock_request: Mock,
) -> None:
    tool_parser = ToolParserManager.get_tool_parser("poolside_v1")(default_tokenizer)
    content, tool_calls = run_tool_extraction(
        tool_parser, SURROUNDING_TEXT_OUTPUT, request=mock_request, streaming=streaming
    )
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "get_weather"
    assert content is not None
    assert "Sure" in content


def test_various_data_types(
    default_tokenizer: TokenizerLike,
    mock_request: Mock,
) -> None:
    """Non-streaming: _deserialize must handle all JSON-compatible types."""
    tool_parser = ToolParserManager.get_tool_parser("poolside_v1")(default_tokenizer)
    _, tool_calls = run_tool_extraction(
        tool_parser, VARIOUS_TYPES_OUTPUT, request=mock_request, streaming=False
    )
    assert len(tool_calls) == 1
    args = json.loads(tool_calls[0].function.arguments)
    assert isinstance(args["string_field"], str)
    assert args["int_field"] == 42
    assert isinstance(args["int_field"], int)
    assert abs(args["float_field"] - 3.14) < 0.001
    assert args["bool_field"] is True
    assert args["null_field"] is None
    assert isinstance(args["array_field"], list)
    assert isinstance(args["object_field"], dict)


@pytest.mark.parametrize("streaming", [True, False])
def test_malformed_input(
    streaming: bool,
    default_tokenizer: TokenizerLike,
    mock_request: Mock,
) -> None:
    """Parser must not raise on malformed inputs."""
    tool_parser = ToolParserManager.get_tool_parser("poolside_v1")(default_tokenizer)
    for malformed in MALFORMED_INPUTS:
        run_tool_extraction(
            tool_parser, malformed, request=mock_request, streaming=streaming
        )
