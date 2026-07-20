# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
from openai.types.responses import FunctionTool, NamespaceTool
from transformers import AutoTokenizer

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.tool_parsers.iquest_coder_v2_tool_parser import IquestCoderV2ToolParser

# The v2 tool parser is tokenizer-format-specific: it only requires that the
# call-boundary tokens exist in the vocabulary. We reuse a small, widely
# available tokenizer and inject the iQuest Coder V2 special tokens.
MODEL = "Qwen/Qwen3-0.6B"

CALL_START = "<iquestcoder_tool_call>"
CALL_END = "</iquestcoder_tool_call>"
KEY_START = "<arg_key>"
KEY_END = "</arg_key>"
VALUE_START = "<arg_value>"
VALUE_END = "</arg_value>"


@pytest.fixture(scope="module")
def iquest_v2_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    required = [CALL_START, CALL_END, KEY_START, KEY_END, VALUE_START, VALUE_END]
    existing = set(tokenizer.get_vocab().keys())
    missing = [token for token in required if token not in existing]
    if missing:
        tokenizer.add_tokens(missing)
    return tokenizer


@pytest.fixture
def iquest_v2_tool_parser(iquest_v2_tokenizer):
    return IquestCoderV2ToolParser(iquest_v2_tokenizer)


@pytest.fixture
def sample_tools():
    return [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "The city name"},
                        "state": {"type": "string", "description": "The state code"},
                        "days": {"type": "integer", "description": "Forecast days"},
                    },
                    "required": ["city", "state"],
                },
            },
        ),
    ]


def _call(tool_name: str, *pairs: tuple[str, str]) -> str:
    """Build a single XML tool-call block in the iQuest Coder V2 format."""
    body = tool_name
    for key, value in pairs:
        body += f"{KEY_START}{key}{KEY_END}{VALUE_START}{value}{VALUE_END}"
    return f"{CALL_START}{body}{CALL_END}"


def _request(tools):
    return ChatCompletionRequest(model=MODEL, messages=[], tools=tools)


def test_missing_tokens_raise(iquest_v2_tokenizer):
    """A tokenizer without the call-boundary tokens is rejected."""
    plain = AutoTokenizer.from_pretrained(MODEL)
    # Ensure the guard triggers even if a prior test added the tokens: the
    # base Qwen3 vocab does not contain the iQuest Coder call tokens.
    if CALL_START not in plain.get_vocab():
        with pytest.raises(ValueError, match="missing required"):
            IquestCoderV2ToolParser(plain)


def test_adjust_request_disables_skip_special_tokens(
    iquest_v2_tool_parser, sample_tools
):
    request = _request(sample_tools)
    adjusted = iquest_v2_tool_parser.adjust_request(request)
    assert adjusted.skip_special_tokens is False


def test_adjust_request_leaves_skip_special_tokens_when_no_tools(
    iquest_v2_tool_parser,
):
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=None)
    adjusted = iquest_v2_tool_parser.adjust_request(request)
    # Default behavior is preserved (parser must not force it off).
    assert adjusted.skip_special_tokens is not False


def test_no_tool_call_returns_content(iquest_v2_tool_parser, sample_tools):
    request = _request(sample_tools)
    model_output = "Just a plain assistant reply with no tool call."
    extracted = iquest_v2_tool_parser.extract_tool_calls(model_output, request=request)

    assert extracted.tools_called is False
    assert extracted.tool_calls == []
    assert extracted.content == model_output


def test_single_tool_call_typed_arguments(iquest_v2_tool_parser, sample_tools):
    request = _request(sample_tools)
    model_output = _call(
        "get_current_weather",
        ("city", "Dallas"),
        ("state", "TX"),
        ("days", "3"),
    )
    extracted = iquest_v2_tool_parser.extract_tool_calls(model_output, request=request)

    assert extracted.tools_called is True
    assert len(extracted.tool_calls) == 1
    assert extracted.tool_calls[0].function.name == "get_current_weather"
    # `city`/`state` are declared string -> kept verbatim; `days` is integer
    # -> JSON-decoded to an int.
    assert json.loads(extracted.tool_calls[0].function.arguments) == {
        "city": "Dallas",
        "state": "TX",
        "days": 3,
    }


def test_string_parameter_is_not_json_decoded(iquest_v2_tool_parser, sample_tools):
    """A numeric-looking value for a string parameter stays a string."""
    request = _request(sample_tools)
    model_output = _call("get_current_weather", ("city", "12345"))
    extracted = iquest_v2_tool_parser.extract_tool_calls(model_output, request=request)

    args = json.loads(extracted.tool_calls[0].function.arguments)
    assert args["city"] == "12345"
    assert isinstance(args["city"], str)


def test_tool_call_without_arguments(iquest_v2_tool_parser, sample_tools):
    request = _request(sample_tools)
    model_output = f"{CALL_START}get_current_weather{CALL_END}"
    extracted = iquest_v2_tool_parser.extract_tool_calls(model_output, request=request)

    assert extracted.tools_called is True
    assert extracted.tool_calls[0].function.name == "get_current_weather"
    assert json.loads(extracted.tool_calls[0].function.arguments) == {}


def test_leading_content_preserved(iquest_v2_tool_parser, sample_tools):
    request = _request(sample_tools)
    prefix = "Let me look that up for you.\n"
    model_output = prefix + _call("get_current_weather", ("city", "Dallas"))
    extracted = iquest_v2_tool_parser.extract_tool_calls(model_output, request=request)

    assert extracted.tools_called is True
    assert extracted.content == prefix


def test_parallel_tool_calls(iquest_v2_tool_parser, sample_tools):
    request = _request(sample_tools)
    model_output = _call("get_current_weather", ("city", "Dallas")) + _call(
        "get_current_weather", ("city", "Orlando")
    )
    extracted = iquest_v2_tool_parser.extract_tool_calls(model_output, request=request)

    assert extracted.tools_called is True
    assert len(extracted.tool_calls) == 2
    assert json.loads(extracted.tool_calls[0].function.arguments) == {"city": "Dallas"}
    assert json.loads(extracted.tool_calls[1].function.arguments) == {"city": "Orlando"}


def test_whitespace_between_name_and_args(iquest_v2_tool_parser, sample_tools):
    """Newlines around the name and between key/value tags are tolerated."""
    request = _request(sample_tools)
    model_output = (
        f"{CALL_START}get_current_weather\n"
        f"{KEY_START}city{KEY_END}\n{VALUE_START}Dallas{VALUE_END}\n"
        f"{CALL_END}"
    )
    extracted = iquest_v2_tool_parser.extract_tool_calls(model_output, request=request)

    assert extracted.tools_called is True
    assert extracted.tool_calls[0].function.name == "get_current_weather"
    assert json.loads(extracted.tool_calls[0].function.arguments) == {"city": "Dallas"}


def test_malformed_call_missing_value_end_is_skipped(
    iquest_v2_tool_parser, sample_tools
):
    """A call whose <arg_value> is never closed is dropped, not crashed."""
    request = _request(sample_tools)
    model_output = (
        f"{CALL_START}get_current_weather{KEY_START}city{KEY_END}"
        f"{VALUE_START}Dallas{CALL_END}"
    )
    extracted = iquest_v2_tool_parser.extract_tool_calls(model_output, request=request)

    assert extracted.tools_called is False
    assert extracted.tool_calls == []


def _feed_deltas(parser, deltas, request):
    """Drive the streaming parser with explicit delta chunks.

    This bypasses the tokenizer so the test controls exactly how the text is
    split across deltas (e.g. a start token straddling two deltas).
    """
    previous_text = ""
    for delta_text in deltas:
        current_text = previous_text + delta_text
        delta_message = parser.extract_tool_calls_streaming(
            previous_text,
            current_text,
            delta_text,
            [],
            [],
            [],
            request=request,
        )
        if delta_message is not None:
            yield delta_message
        previous_text = current_text


def test_streaming_passthrough_when_tools_disabled(iquest_v2_tool_parser):
    """With no tools, deltas are emitted verbatim as content."""
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=None)
    deltas = ["Hello ", "world"]

    content = "".join(
        dm.content or "" for dm in _feed_deltas(iquest_v2_tool_parser, deltas, request)
    )
    assert content == "Hello world"


def test_streaming_content_then_tool_call(iquest_v2_tool_parser, sample_tools):
    request = _request(sample_tools)
    deltas = [
        "Checking the weather.",
        CALL_START,
        "get_current_weather",
        f"{KEY_START}city{KEY_END}",
        f"{VALUE_START}Dallas{VALUE_END}",
        CALL_END,
    ]

    content = ""
    tool_names: list[str] = []
    tool_args: list[str] = []
    for dm in _feed_deltas(iquest_v2_tool_parser, deltas, request):
        assert not dm.role
        if dm.content:
            content += dm.content
        for tc in dm.tool_calls:
            if tc.function and tc.function.name:
                tool_names.append(tc.function.name)
            if tc.function and tc.function.arguments:
                tool_args.append(tc.function.arguments)

    assert content == "Checking the weather."
    assert tool_names == ["get_current_weather"]
    assert json.loads(tool_args[0]) == {"city": "Dallas"}


def test_streaming_start_token_split_across_deltas(iquest_v2_tool_parser, sample_tools):
    """A start token split mid-token must not leak as content."""
    request = _request(sample_tools)
    head, tail = CALL_START[:5], CALL_START[5:]
    deltas = [
        head,
        tail,
        "get_current_weather",
        f"{KEY_START}city{KEY_END}{VALUE_START}Dallas{VALUE_END}",
        CALL_END,
    ]

    content = ""
    tool_indices = set()
    for dm in _feed_deltas(iquest_v2_tool_parser, deltas, request):
        if dm.content:
            content += dm.content
        for tc in dm.tool_calls:
            tool_indices.add(tc.index)

    assert content == ""
    assert tool_indices == {0}


def test_streaming_parallel_tool_calls_indices(iquest_v2_tool_parser, sample_tools):
    request = _request(sample_tools)
    deltas = [
        _call("get_current_weather", ("city", "Dallas")),
        _call("get_current_weather", ("city", "Orlando")),
    ]

    tool_indices = []
    for dm in _feed_deltas(iquest_v2_tool_parser, deltas, request):
        assert not dm.content
        for tc in dm.tool_calls:
            tool_indices.append(tc.index)

    assert tool_indices == [0, 1]


# _parameter_is_string across tool shapes.
#
# The parser is invoked from both the Chat Completions path (tools are
# ChatCompletionToolsParam, params nested under `.function`) and the Responses
# API path (tools are FunctionTool / NamespaceTool with `.name`/`.parameters`
# on the tool itself). Directly reading `tool.function` used to crash on the
# Responses shapes with AttributeError; these lock in support for all three.
def _weather_properties():
    return {
        "city": {"type": "string"},
        "days": {"type": "integer"},
    }


def test_parameter_is_string_responses_function_tool():
    """Responses API FunctionTool has name/parameters directly (no `.function`)."""
    ft = FunctionTool(
        type="function",
        name="get_current_weather",
        description="w",
        strict=False,
        parameters={"type": "object", "properties": _weather_properties()},
    )
    request = SimpleNamespace(tools=[ft], tool_choice="auto")

    assert IquestCoderV2ToolParser._parameter_is_string(
        request, "get_current_weather", "city"
    )
    assert not IquestCoderV2ToolParser._parameter_is_string(
        request, "get_current_weather", "days"
    )


def test_parameter_is_string_chat_completion_tool():
    """Chat Completions tool nests name/parameters under `.function`."""
    tool = ChatCompletionToolsParam(
        type="function",
        function={
            "name": "get_current_weather",
            "parameters": {"type": "object", "properties": _weather_properties()},
        },
    )
    request = SimpleNamespace(tools=[tool], tool_choice="auto")

    assert IquestCoderV2ToolParser._parameter_is_string(
        request, "get_current_weather", "city"
    )
    assert not IquestCoderV2ToolParser._parameter_is_string(
        request, "get_current_weather", "days"
    )


def test_parameter_is_string_namespace_tool():
    """Namespace tool children match against the flattened `namespace__name`."""
    ns = NamespaceTool(
        type="namespace",
        name="multi_agent_v1",
        description="agents",
        tools=[
            {
                "type": "function",
                "name": "spawn_agent",
                "description": "spawn",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                        "count": {"type": "integer"},
                    },
                },
            }
        ],
    )
    request = SimpleNamespace(tools=[ns], tool_choice="auto")

    assert IquestCoderV2ToolParser._parameter_is_string(
        request, "multi_agent_v1__spawn_agent", "message"
    )
    assert not IquestCoderV2ToolParser._parameter_is_string(
        request, "multi_agent_v1__spawn_agent", "count"
    )


def test_parameter_is_string_unknown_tool_or_param():
    ft = FunctionTool(
        type="function",
        name="get_current_weather",
        description="w",
        strict=False,
        parameters={"type": "object", "properties": _weather_properties()},
    )
    request = SimpleNamespace(tools=[ft], tool_choice="auto")

    assert not IquestCoderV2ToolParser._parameter_is_string(
        request, "unknown_tool", "city"
    )
    assert not IquestCoderV2ToolParser._parameter_is_string(
        request, "get_current_weather", "unknown_param"
    )


def test_parameter_is_string_no_tools():
    request = SimpleNamespace(tools=None, tool_choice="auto")
    assert not IquestCoderV2ToolParser._parameter_is_string(request, "x", "y")
