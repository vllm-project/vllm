# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.tool_parsers.iquest_coder_tool_parser import IquestCoderToolParser
from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

# iquest_coder shares the qwen3_coder XML tool-call format and tokenizer.
MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"


@pytest.fixture(scope="module")
def iquest_tokenizer():
    from vllm.tokenizers import get_tokenizer

    return get_tokenizer(tokenizer_name=MODEL)


@pytest.fixture
def iquest_tool_parser(iquest_tokenizer):
    return IquestCoderToolParser(iquest_tokenizer)


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
                        "unit": {"type": "string", "enum": ["fahrenheit", "celsius"]},
                    },
                    "required": ["city", "state"],
                },
            },
        ),
    ]


def test_is_subclass_of_qwen3_coder():
    assert issubclass(IquestCoderToolParser, Qwen3CoderToolParser)


def _feed_deltas(parser, deltas, request):
    """Drive the streaming parser with an explicit list of delta chunks.

    This does not go through the tokenizer, so the test can control exactly
    how text is chunked across deltas (e.g. fusing separator whitespace with
    the following <tool_call> start token).
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
        if delta_message:
            yield delta_message
        previous_text = current_text


def test_streaming_no_content_between_parallel_tools(iquest_tool_parser, sample_tools):
    """Whitespace between parallel tool calls must not leak as content.

    When the separator "\\n" between </tool_call> and the next <tool_call>
    is fused into a single delta with the start token, the base qwen3_coder
    parser emits that whitespace as a stray content block sitting between the
    two tool_use blocks. IquestCoderToolParser drops it.
    """
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=sample_tools)

    # Deltas crafted so the inter-call "\n" arrives fused with the next
    # <tool_call> start token, mimicking a real tokenizer's "\n<tool_call>".
    deltas = [
        "<tool_call>\n",
        "<function=get_current_weather>\n",
        "<parameter=city>\nDallas\n</parameter>\n",
        "<parameter=state>\nTX\n</parameter>\n",
        "<parameter=unit>\nfahrenheit\n</parameter>\n",
        "</function>\n",
        "</tool_call>",
        "\n<tool_call>\n",  # separator whitespace fused with start token
        "<function=get_current_weather>\n",
        "<parameter=city>\nOrlando\n</parameter>\n",
        "<parameter=state>\nFL\n</parameter>\n",
        "<parameter=unit>\ncelsius\n</parameter>\n",
        "</function>\n",
        "</tool_call>",
    ]

    other_content = ""
    tool_indices = set()

    for delta_message in _feed_deltas(iquest_tool_parser, deltas, request):
        assert not delta_message.role
        if delta_message.content:
            other_content += delta_message.content
        if delta_message.tool_calls:
            for tool_call in delta_message.tool_calls:
                tool_indices.add(tool_call.index)

    # No stray whitespace content block should have been emitted.
    assert other_content == ""
    # Both parallel tool calls should have been streamed.
    assert tool_indices == {0, 1}


def test_streaming_preserves_real_leading_content(iquest_tool_parser, sample_tools):
    """Genuine text before the first tool call is still streamed."""
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=sample_tools)

    deltas = [
        "Let me check the weather.",
        "<tool_call>\n",
        "<function=get_current_weather>\n",
        "<parameter=city>\nDallas\n</parameter>\n",
        "<parameter=state>\nTX\n</parameter>\n",
        "<parameter=unit>\nfahrenheit\n</parameter>\n",
        "</function>\n",
        "</tool_call>",
    ]

    other_content = ""
    tool_indices = set()

    for delta_message in _feed_deltas(iquest_tool_parser, deltas, request):
        if delta_message.content:
            other_content += delta_message.content
        if delta_message.tool_calls:
            for tool_call in delta_message.tool_calls:
                tool_indices.add(tool_call.index)

    assert other_content == "Let me check the weather."
    assert tool_indices == {0}


def test_non_streaming_parallel_tools(iquest_tool_parser, sample_tools):
    """Non-streaming extraction is inherited unchanged from qwen3_coder."""
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=sample_tools)
    model_output = (
        "<tool_call>\n<function=get_current_weather>\n"
        "<parameter=city>\nDallas\n</parameter>\n"
        "<parameter=state>\nTX\n</parameter>\n"
        "<parameter=unit>\nfahrenheit\n</parameter>\n"
        "</function>\n</tool_call>\n"
        "<tool_call>\n<function=get_current_weather>\n"
        "<parameter=city>\nOrlando\n</parameter>\n"
        "<parameter=state>\nFL\n</parameter>\n"
        "<parameter=unit>\ncelsius\n</parameter>\n"
        "</function>\n</tool_call>"
    )

    extracted = iquest_tool_parser.extract_tool_calls(model_output, request=request)

    assert extracted.tools_called
    assert len(extracted.tool_calls) == 2
    assert json.loads(extracted.tool_calls[0].function.arguments) == {
        "city": "Dallas",
        "state": "TX",
        "unit": "fahrenheit",
    }
    assert json.loads(extracted.tool_calls[1].function.arguments) == {
        "city": "Orlando",
        "state": "FL",
        "unit": "celsius",
    }


# Argument type conversion across tool shapes.
#
# The argument-config lookup drives type conversion (e.g. an integer parameter
# is JSON-decoded rather than kept as a string). The base qwen3_coder parser
# only understands the Chat Completions shape (`.function.parameters`); on the
# /v1/responses path tools are Responses API FunctionTool / NamespaceTool with
# `name`/`parameters` on the tool itself. iquest_coder overrides
# `_get_arguments_config` to resolve all three shapes.

_TYPED_OUTPUT = (
    "<tool_call>\n<function={name}>\n"
    "<parameter=command>\napt-get update\n</parameter>\n"
    "<parameter=timeout_ms>\n300000\n</parameter>\n"
    "</function>\n</tool_call>"
)

_TYPED_PROPERTIES = {
    "type": "object",
    "properties": {
        "command": {"type": "string"},
        "timeout_ms": {"type": "integer"},
    },
}


def test_type_conversion_responses_function_tool(iquest_tool_parser):
    """Responses API FunctionTool: integer param is decoded, not left a string."""
    from types import SimpleNamespace

    from openai.types.responses import FunctionTool

    ft = FunctionTool(
        type="function",
        name="shell_command",
        description="run a shell command",
        strict=False,
        parameters=_TYPED_PROPERTIES,
    )
    request = SimpleNamespace(tools=[ft], tool_choice="auto")
    extracted = iquest_tool_parser.extract_tool_calls(
        _TYPED_OUTPUT.format(name="shell_command"), request=request
    )

    args = json.loads(extracted.tool_calls[0].function.arguments)
    assert args["command"] == "apt-get update"
    assert args["timeout_ms"] == 300000
    assert isinstance(args["timeout_ms"], int)


def test_type_conversion_responses_namespace_tool(iquest_tool_parser):
    """Namespace children resolve via the flattened `namespace__name`."""
    from types import SimpleNamespace

    from openai.types.responses import NamespaceTool

    ns = NamespaceTool(
        type="namespace",
        name="agents",
        description="agent bundle",
        tools=[
            {
                "type": "function",
                "name": "shell_command",
                "description": "run a shell command",
                "parameters": _TYPED_PROPERTIES,
            }
        ],
    )
    request = SimpleNamespace(tools=[ns], tool_choice="auto")
    extracted = iquest_tool_parser.extract_tool_calls(
        _TYPED_OUTPUT.format(name="agents__shell_command"), request=request
    )

    args = json.loads(extracted.tool_calls[0].function.arguments)
    assert args["timeout_ms"] == 300000
    assert isinstance(args["timeout_ms"], int)


def test_type_conversion_chat_completion_tool_fallback(iquest_tool_parser):
    """Chat Completions shape still works via the base-class lookup."""
    tool = ChatCompletionToolsParam(
        type="function",
        function={"name": "shell_command", "parameters": _TYPED_PROPERTIES},
    )
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=[tool])
    extracted = iquest_tool_parser.extract_tool_calls(
        _TYPED_OUTPUT.format(name="shell_command"), request=request
    )

    args = json.loads(extracted.tool_calls[0].function.arguments)
    assert args["timeout_ms"] == 300000
    assert isinstance(args["timeout_ms"], int)


def test_string_param_not_decoded_responses_shape(iquest_tool_parser):
    """A numeric-looking value for a string param stays a string."""
    from types import SimpleNamespace

    from openai.types.responses import FunctionTool

    ft = FunctionTool(
        type="function",
        name="shell_command",
        description="s",
        strict=False,
        parameters={
            "type": "object",
            "properties": {"command": {"type": "string"}},
        },
    )
    request = SimpleNamespace(tools=[ft], tool_choice="auto")
    model_output = (
        "<tool_call>\n<function=shell_command>\n"
        "<parameter=command>\n12345\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    extracted = iquest_tool_parser.extract_tool_calls(model_output, request=request)

    args = json.loads(extracted.tool_calls[0].function.arguments)
    assert args["command"] == "12345"
    assert isinstance(args["command"], str)
