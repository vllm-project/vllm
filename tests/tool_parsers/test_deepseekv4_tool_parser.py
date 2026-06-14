# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for DeepSeekV4ToolParser."""

import json
from unittest.mock import MagicMock

import pytest
from xgrammar import StructuralTag

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedFunction,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.parser.abstract_parser import DelegatingParser
from vllm.reasoning.deepseek_v3_reasoning_parser import (
    DeepSeekV3ReasoningWithThinkingParser,
)
from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.deepseekv4_tool_parser import DeepSeekV4ToolParser

MOCK_TOKENIZER = MagicMock()
MOCK_TOKENIZER.get_vocab.return_value = {}

TC_START = "<｜DSML｜tool_calls>"
TC_END = "</｜DSML｜tool_calls>"
INV_START = '<｜DSML｜invoke name="'
INV_END = "</｜DSML｜invoke>"
PARAM_START = '<｜DSML｜parameter name="'
PARAM_END = "</｜DSML｜parameter>"


@pytest.fixture
def sample_tools() -> list[ChatCompletionToolsParam]:
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
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "calculate_area",
                "description": "Calculate area of a shape",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "shape": {"type": "string"},
                        "dimensions": {"type": "object"},
                        "precision": {"type": "integer"},
                    },
                },
            },
        ),
    ]


def make_parser(tools=None) -> DeepSeekV4ToolParser:
    return DeepSeekV4ToolParser(MOCK_TOKENIZER, tools=tools)


def make_request(tools=None) -> MagicMock:
    req = MagicMock()
    req.tools = tools
    return req


def build_tool_call(func_name: str, params: dict[str, str]) -> str:
    param_strs = "".join(
        f'{PARAM_START}{k}" string="true">{v}{PARAM_END}\n' for k, v in params.items()
    )
    return f'{TC_START}\n{INV_START}{func_name}">\n{param_strs}{INV_END}\n{TC_END}'


def stream(parser: DeepSeekV4ToolParser, full_text: str, chunk_size: int = 7):
    deltas = []
    previous_text = ""
    for start in range(0, len(full_text), chunk_size):
        delta_text = full_text[start : start + chunk_size]
        current_text = previous_text + delta_text
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[1],
            request=make_request(),
        )
        previous_text = current_text
        if delta is not None:
            deltas.append(delta)
    return deltas


def reconstruct_args(deltas, tool_index: int = 0) -> str:
    fragments = []
    for delta in deltas:
        if delta.tool_calls:
            for tool_call in delta.tool_calls:
                if (
                    tool_call.index == tool_index
                    and tool_call.function
                    and tool_call.function.arguments
                ):
                    fragments.append(tool_call.function.arguments)
    return "".join(fragments)


def test_registered():
    assert ToolParserManager.get_tool_parser("deepseek_v4") is DeepSeekV4ToolParser


def test_extract_tool_calls():
    parser = make_parser()
    model_output = "Let me check. " + build_tool_call(
        "get_weather", {"location": "Beijing", "unit": "celsius"}
    )

    result = parser.extract_tool_calls(model_output, make_request())

    assert result.tools_called
    assert result.content == "Let me check. "
    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.function.name == "get_weather"
    assert json.loads(tool_call.function.arguments) == {
        "location": "Beijing",
        "unit": "celsius",
    }


def test_function_calls_block_is_not_accepted():
    parser = make_parser()
    model_output = build_tool_call("search", {"query": "vllm"}).replace(
        "tool_calls", "function_calls"
    )

    result = parser.extract_tool_calls(model_output, make_request())

    assert not result.tools_called
    assert result.content == model_output


def test_streaming_extracts_complete_invokes():
    parser = make_parser()
    full_text = build_tool_call("search", {"query": "deepseek v4"})

    deltas = stream(parser, full_text, chunk_size=5)

    names = [
        tool_call.function.name
        for delta in deltas
        if delta.tool_calls
        for tool_call in delta.tool_calls
        if tool_call.function.name
    ]
    assert names == ["search"]
    assert json.loads(reconstruct_args(deltas)) == {"query": "deepseek v4"}


def test_streaming_emits_incremental_argument_chunks():
    tool = ChatCompletionToolsParam(
        function=FunctionDefinition(
            name="plan_trip",
            parameters={
                "type": "object",
                "properties": {
                    "days": {"type": "integer"},
                    "flexible": {"type": "boolean"},
                    "cities": {"type": "array", "items": {"type": "string"}},
                    "notes": {"type": "string"},
                },
            },
        ),
    )
    parser = make_parser(tools=[tool])
    full_text = (
        f"{TC_START}\n"
        f'{INV_START}plan_trip">\n'
        f'{PARAM_START}days" string="false">3{PARAM_END}\n'
        f'{PARAM_START}flexible" string="false">false{PARAM_END}\n'
        f'{PARAM_START}cities" string="false">'
        f'["Beijing","Shanghai","Tokyo","New York"]{PARAM_END}\n'
        f'{PARAM_START}notes" string="true">靠窗座位{PARAM_END}\n'
        f"{INV_END}\n"
        f"{TC_END}"
    )

    deltas = stream(parser, full_text, chunk_size=4)
    arg_chunks = [
        tool_call.function.arguments
        for delta in deltas
        for tool_call in delta.tool_calls or []
        if tool_call.function and tool_call.function.arguments is not None
    ]

    assert len([chunk for chunk in arg_chunks if chunk]) > 2
    assert json.loads("".join(arg_chunks)) == {
        "days": 3,
        "flexible": False,
        "cities": ["Beijing", "Shanghai", "Tokyo", "New York"],
        "notes": "靠窗座位",
    }


def test_get_vllm_registry_structural_tag_returns_structural_tag(
    sample_tools: list[ChatCompletionToolsParam],
) -> None:
    parser = make_parser()
    req = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=sample_tools,
        tool_choice="auto",
    )
    tag = parser.get_structural_tag(req)
    assert isinstance(tag, StructuralTag)

    req = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=sample_tools,
        tool_choice="required",
    )
    tag = parser.get_structural_tag(req)
    assert isinstance(tag, StructuralTag)

    if sample_tools:
        tool = sample_tools[0]
        req = ChatCompletionRequest(
            messages=[],
            model="m",
            tools=sample_tools,
        )
        req.tool_choice = ChatCompletionNamedToolChoiceParam(
            function=ChatCompletionNamedFunction(name=tool.function.name)
        )
        tag = parser.get_structural_tag(req)
        assert isinstance(tag, StructuralTag)


def test_extract_tool_calls_arguments_wrapper():
    mock_tokenizer = MagicMock()
    mock_tokenizer.get_vocab.return_value = {}

    tool = ChatCompletionToolsParam(
        type="function",
        function={
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        },
    )

    parser = DeepSeekV4ToolParser(mock_tokenizer, tools=[tool])
    request = MagicMock()
    request.tools = [tool]

    model_output = (
        f"{TC_START}"
        f'{INV_START}get_weather">'
        f'{PARAM_START}arguments" string="false">{{"location":"Beijing"}}{PARAM_END}'
        f"{INV_END}"
        f"{TC_END}"
    )

    result = parser.extract_tool_calls(model_output, request)
    assert result.tools_called
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args == {"location": "Beijing"}


@pytest.mark.skip_global_cleanup
def test_composed_schema_converts_object_and_array_params():
    tool = ChatCompletionToolsParam(
        type="function",
        function={
            "name": "set_timer",
            "parameters": {
                "type": "object",
                "properties": {
                    "wait": {
                        "anyOf": [
                            {"type": "object"},
                            {"type": "null"},
                        ],
                    },
                    "patches": {
                        "allOf": [
                            {"type": "array", "items": {"type": "object"}},
                        ],
                    },
                },
            },
        },
    )
    parser = make_parser(tools=[tool])
    request = make_request(tools=[tool])
    model_output = (
        f"{TC_START}\n"
        f'{INV_START}set_timer">\n'
        f'{PARAM_START}wait" string="false">'
        f'{{"type":"for","minutes":2880}}'
        f"{PARAM_END}\n"
        f'{PARAM_START}patches" string="false">'
        f'[{{"op":"replace","path":"/schedule","value":"quiet"}}]'
        f"{PARAM_END}\n"
        f"{INV_END}\n"
        f"{TC_END}"
    )

    result = parser.extract_tool_calls(model_output, request)

    assert result.tools_called
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args == {
        "wait": {"type": "for", "minutes": 2880},
        "patches": [{"op": "replace", "path": "/schedule", "value": "quiet"}],
    }


class _V4Parser(DelegatingParser):
    """Mirrors what ParserManager composes for the `deepseek_v4` model."""

    reasoning_parser_cls = DeepSeekV3ReasoningWithThinkingParser
    tool_parser_cls = DeepSeekV4ToolParser


def make_full_parser() -> _V4Parser:
    tokenizer = MagicMock()
    # The reasoning parser only needs these tokens to exist in the vocab;
    # non-streaming parse() is string-based so the ids are arbitrary.
    tokenizer.get_vocab.return_value = {"<think>": 1, "</think>": 2}
    return _V4Parser(tokenizer)


def make_auto_request() -> MagicMock:
    req = MagicMock()
    req.tools = None
    req.tool_choice = "auto"
    req.request_id = "test-dangling-think"
    return req


def stream_full(
    parser: _V4Parser,
    full_text: str,
    request: MagicMock,
    chunk_size: int = 7,
) -> list:
    """Drive the composed parser's streaming entry point chunk by chunk.

    Mirrors a prefilled `<think>` generation: the prompt carries the think
    start token (id 1) so the parser begins in the reasoning phase, and the
    generated deltas never contain the `</think>` end token (id 2) -- exactly
    the dangling-think shape. Placeholder id 3 stands in for ordinary
    (non-special) generated tokens.
    """
    deltas = []
    n = len(full_text)
    for start in range(0, n, chunk_size):
        delta_text = full_text[start : start + chunk_size]
        delta = parser.parse_delta(
            delta_text=delta_text,
            delta_token_ids=[3],
            request=request,
            prompt_token_ids=[1] if start == 0 else None,
            finished=start + chunk_size >= n,
        )
        if delta is not None:
            deltas.append(delta)
    return deltas


def reconstruct_tool_names(deltas) -> list[str]:
    return [
        tool_call.function.name
        for delta in deltas
        if delta.tool_calls
        for tool_call in delta.tool_calls
        if tool_call.function and tool_call.function.name
    ]


def test_tool_call_without_think_close_is_recovered():
    """Regression: a complete DSML tool block with NO preceding `</think>`.

    The block would otherwise stay trapped in `reasoning` with empty content and
    the tool call silently lost. The guarded non-streaming recovery promotes it
    to a real tool call and keeps the leading prose as reasoning.
    """
    parser = make_full_parser()
    # No </think> anywhere: the canonical lost-tool-call shape.
    model_output = "Let me check.\n\n" + build_tool_call(
        "terminal", {"command": "git --no-pager log -1"}
    )

    reasoning, content, tool_calls = parser.parse(
        model_output, make_auto_request(), enable_auto_tools=True
    )

    assert tool_calls and len(tool_calls) == 1
    assert tool_calls[0].name == "terminal"
    assert json.loads(tool_calls[0].arguments) == {
        "command": "git --no-pager log -1"
    }
    # Reasoning prose is kept, but the DSML block must not remain inside it.
    assert reasoning is not None and TC_START not in reasoning


def test_tool_call_without_think_close_is_recovered_streaming():
    """Streaming counterpart of the dangling-`</think>` recovery.

    Same canonical lost-tool-call shape, but fed through ``parse_delta`` chunk
    by chunk. Because no ``</think>`` token ever arrives, the reasoning channel
    never ends on its own; the recovery detects the ``<｜DSML｜tool_calls>``
    marker mid-stream (buffering partial markers across chunks), stops reasoning
    right before it, and routes the block to the tool-call phase so it surfaces
    as a real tool call instead of being streamed as reasoning.
    """
    parser = make_full_parser()
    request = make_auto_request()
    # No </think> anywhere: the canonical lost-tool-call shape.
    full_text = "Let me check.\n\n" + build_tool_call(
        "terminal", {"command": "git --no-pager log -1"}
    )

    deltas = stream_full(parser, full_text, request, chunk_size=5)

    assert reconstruct_tool_names(deltas) == ["terminal"]
    assert json.loads(reconstruct_args(deltas)) == {
        "command": "git --no-pager log -1"
    }
    # The DSML block must not leak into the streamed reasoning channel.
    streamed_reasoning = "".join(
        delta.reasoning for delta in deltas if delta.reasoning
    )
    assert TC_START not in streamed_reasoning
