# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ParsableContext's parsing behavior.

These tests verify that ParsableContext correctly delegates to the unified
Parser (via parse) and properly builds response output items.
"""

import json
from collections.abc import Sequence
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from openai.types.responses import ResponseFunctionToolCall
from openai.types.responses.response_function_tool_call_output_item import (
    ResponseFunctionToolCallOutputItem,
)
from openai.types.responses.response_output_item import McpCall

from vllm.entrypoints.generate.base.protocol import (
    DeltaMessage,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.mcp.tool import HarmonyPythonTool
from vllm.entrypoints.openai.responses.context import ParsableContext
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.parser.abstract_parser import DelegatingParser

pytestmark = pytest.mark.skip_global_cleanup


# ---------------------------------------------------------------------------
# Test parser stubs
# ---------------------------------------------------------------------------


class _NoOpParser(DelegatingParser):
    """Parser that extracts no reasoning and no tool calls."""

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        return False

    def extract_reasoning(self, model_output, request):
        return None, model_output

    def extract_reasoning_streaming(self, *args, **kwargs):
        return None

    def extract_tool_calls(self, model_output, request):
        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=model_output
        )

    def extract_tool_calls_streaming(self, *args, **kwargs):
        return None

    def parse_delta(self, *args, **kwargs) -> DeltaMessage | None:
        return None


class _ReasoningOnlyParser(DelegatingParser):
    """Parser that extracts reasoning but no tool calls."""

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        return False

    def extract_reasoning(self, model_output, request):
        if "<think>" in model_output and "</think>" in model_output:
            start = model_output.index("<think>") + len("<think>")
            end = model_output.index("</think>")
            reasoning = model_output[start:end]
            content = model_output[end + len("</think>") :]
            return reasoning, content.strip() or None
        return None, model_output

    def extract_reasoning_streaming(self, *args, **kwargs):
        return None

    def extract_tool_calls(self, model_output, request):
        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=model_output
        )

    def extract_tool_calls_streaming(self, *args, **kwargs):
        return None

    def parse_delta(self, *args, **kwargs) -> DeltaMessage | None:
        return None


class _StubToolParser:
    """Minimal tool parser stub that always returns a hardcoded tool call."""

    supports_required_and_named = False

    def __init__(self, tokenizer=None, tools=None):
        pass

    def extract_tool_calls(self, model_output, request):
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=[
                ToolCall(
                    id="call_123",
                    type="function",
                    function=FunctionCall(
                        name="get_weather",
                        arguments='{"location": "Paris"}',
                    ),
                )
            ],
            content=None,
        )

    def extract_tool_calls_streaming(self, *args, **kwargs):
        return None

    def adjust_request(self, request):
        return request


class _ToolCallingParser(DelegatingParser):
    """Parser that extracts a hardcoded tool call from any input."""

    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer)
        self._tool_parser = _StubToolParser()

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        return False

    def extract_reasoning(self, model_output, request):
        return None, model_output

    def extract_reasoning_streaming(self, *args, **kwargs):
        return None

    def extract_tool_calls_streaming(self, *args, **kwargs):
        return None

    def parse_delta(self, *args, **kwargs) -> DeltaMessage | None:
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(**overrides) -> ResponsesRequest:
    defaults = {"model": "test-model", "input": "test"}
    defaults.update(overrides)
    return ResponsesRequest.model_validate(defaults)


def _make_request_output(
    text: str = "Hello, world!",
    token_ids: Sequence[int] = (1, 2, 3),
    finish_reason: str = "stop",
) -> RequestOutput:
    return RequestOutput(
        request_id="test",
        prompt=None,
        prompt_token_ids=[],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text=text,
                token_ids=list(token_ids),
                cumulative_logprob=None,
                logprobs=None,
                finish_reason=finish_reason,
            )
        ],
        finished=True,
    )


def _make_context(parser_cls, **overrides):
    # ParsableContext no longer lazily builds a parser from ``parser_cls``;
    # the caller (here, the serving layer in production) must supply one.
    request = overrides.get("request", _make_request())
    response_parser = overrides.pop("response_parser", None)
    if response_parser is None and parser_cls is not None:
        response_parser = parser_cls(MagicMock(), request.tools)

    defaults = dict(
        tokenizer=MagicMock(),
        parser_cls=parser_cls,
        response_parser=response_parser,
        response_messages=[],
        request=request,
        available_tools=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    defaults.update(overrides)
    return ParsableContext(**defaults)


def _make_builtin_tool_call(
    name: str, call_id: str, arguments: dict | str
) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        id=f"fc_{call_id}",
        call_id=call_id,
        type="function_call",
        name=name,
        arguments=(
            json.dumps(arguments) if not isinstance(arguments, str) else arguments
        ),
    )


# ---------------------------------------------------------------------------
# Tests: basic text passthrough
# ---------------------------------------------------------------------------


def test_process_text_with_parser():
    """Parser with no reasoning/tools returns a single message item."""
    ctx = _make_context(_NoOpParser)
    ctx.append_output(_make_request_output(text="Hello!"))

    assert len(ctx.response_messages) == 1
    msg = ctx.response_messages[0]
    assert msg.type == "message"
    assert msg.content[0].text == "Hello!"


def test_process_text_without_parser():
    """parser_cls=None falls back to plain text wrapping."""
    ctx = _make_context(None)
    ctx.append_output(_make_request_output(text="Hello!"))

    assert len(ctx.response_messages) == 1
    msg = ctx.response_messages[0]
    assert msg.type == "message"
    assert msg.content[0].text == "Hello!"


# ---------------------------------------------------------------------------
# Tests: empty / whitespace output
# ---------------------------------------------------------------------------


def test_process_empty_text_without_parser():
    """Empty text with no parser produces no output items."""
    ctx = _make_context(None)
    ctx.append_output(_make_request_output(text=""))

    assert len(ctx.response_messages) == 0


def test_process_empty_text_with_parser():
    """Empty text with parser produces no output items."""
    ctx = _make_context(_NoOpParser)
    ctx.append_output(_make_request_output(text=""))

    assert len(ctx.response_messages) == 0


# ---------------------------------------------------------------------------
# Tests: reasoning extraction
# ---------------------------------------------------------------------------


def test_process_extracts_reasoning():
    """Parser that finds reasoning produces both reasoning and message items."""
    ctx = _make_context(_ReasoningOnlyParser)
    ctx.append_output(
        _make_request_output(text="<think>Let me check</think>The answer is 42")
    )

    types = [m.type for m in ctx.response_messages]
    assert "reasoning" in types
    assert "message" in types

    reasoning_item = next(m for m in ctx.response_messages if m.type == "reasoning")
    assert reasoning_item.content[0].text == "Let me check"

    message_item = next(m for m in ctx.response_messages if m.type == "message")
    assert message_item.content[0].text == "The answer is 42"


def test_process_reasoning_only_no_content():
    """When reasoning consumes all text, only a reasoning item is produced."""
    ctx = _make_context(_ReasoningOnlyParser)
    ctx.append_output(_make_request_output(text="<think>Just thinking</think>"))

    types = [m.type for m in ctx.response_messages]
    assert "reasoning" in types
    assert "message" not in types


# ---------------------------------------------------------------------------
# Tests: tool call extraction
# ---------------------------------------------------------------------------


def test_process_extracts_tool_calls():
    """Parser that finds tool calls produces function_call items."""
    request = _make_request(
        tool_choice="auto",
        tools=[
            {
                "type": "function",
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
    )
    ctx = _make_context(_ToolCallingParser, request=request, enable_auto_tools=True)
    ctx.append_output(_make_request_output(text="calling tool"))

    types = [m.type for m in ctx.response_messages]
    assert "function_call" in types

    tool_item = next(m for m in ctx.response_messages if m.type == "function_call")
    assert tool_item.name == "get_weather"
    assert tool_item.arguments == '{"location": "Paris"}'
    assert tool_item.status == "completed"


@pytest.mark.parametrize(
    ("tool_name", "session_name", "dispatched_name", "arguments"),
    [
        ("code_interpreter", "python", "python", {"code": "print(42)"}),
        ("web_search_preview", "browser", "search", {"query": "vLLM"}),
        ("container.exec", "container", "exec", {"cmd": ["pwd"]}),
    ],
)
@pytest.mark.asyncio
async def test_builtin_tool_output_preserves_function_call_id(
    tool_name, session_name, dispatched_name, arguments
):
    """Built-in tool outputs remain correlated with their originating call."""
    tool_session = MagicMock()
    tool_session.call_tool = AsyncMock(
        return_value=SimpleNamespace(content=[SimpleNamespace(text="result")])
    )
    context = _make_context(None, available_tools=[session_name])
    tool_call = ResponseFunctionToolCall(
        id=f"fc_{session_name}",
        call_id=f"call_{session_name}",
        type="function_call",
        name=tool_name,
        arguments=json.dumps(arguments),
    )
    context.response_messages.append(tool_call)
    context._tool_sessions[session_name] = tool_session

    output = await context.call_tool()

    tool_session.call_tool.assert_awaited_once_with(dispatched_name, arguments)
    assert output[0].call_id == tool_call.call_id


@pytest.mark.asyncio
async def test_local_python_tool_output_preserves_function_call_id():
    """Local code-interpreter output remains correlated with its call."""

    async def process(_):
        yield SimpleNamespace(content=[SimpleNamespace(text="result")])

    python_tool = object.__new__(HarmonyPythonTool)
    python_tool.python_tool = MagicMock(process=process)
    context = _make_context(None, available_tools=["python"])
    tool_call = ResponseFunctionToolCall(
        id="fc_python",
        call_id="call_python",
        type="function_call",
        name="code_interpreter",
        arguments='{"code": "print(42)"}',
    )
    context.response_messages.append(tool_call)
    context._tool_sessions["python"] = python_tool

    output = await context.call_tool()

    assert output[0].call_id == tool_call.call_id


@pytest.mark.parametrize(
    "tool_calls",
    [
        [
            _make_builtin_tool_call(
                "web_search_preview", "call_search", {"query": "vLLM"}
            ),
            _make_builtin_tool_call(
                "code_interpreter", "call_python", {"code": "print(42)"}
            ),
        ],
        [
            _make_builtin_tool_call(
                "code_interpreter", "call_python", {"code": "print(42)"}
            ),
            _make_builtin_tool_call(
                "web_search_preview", "call_search", {"query": "vLLM"}
            ),
        ],
    ],
)
@pytest.mark.asyncio
async def test_call_tool_executes_all_parallel_builtin_calls_in_order(tool_calls):
    browser = MagicMock()
    browser.call_tool = AsyncMock(
        return_value=SimpleNamespace(content=[SimpleNamespace(text="search result")])
    )
    python = MagicMock()
    python.call_tool = AsyncMock(
        return_value=SimpleNamespace(content=[SimpleNamespace(text="python result")])
    )
    context = _make_context(None, available_tools=["browser", "python"])
    context.response_messages.extend(tool_calls)
    context._tool_sessions = {"browser": browser, "python": python}

    assert context.need_builtin_tool_call()
    outputs = await context.call_tool()

    browser.call_tool.assert_awaited_once_with("search", {"query": "vLLM"})
    python.call_tool.assert_awaited_once_with("python", {"code": "print(42)"})
    assert [output.call_id for output in outputs] == [
        tool_call.call_id for tool_call in tool_calls
    ]


@pytest.mark.asyncio
async def test_call_tool_executes_multiple_calls_on_same_session():
    browser = MagicMock()
    browser.call_tool = AsyncMock(
        side_effect=[
            SimpleNamespace(content=[SimpleNamespace(text="first result")]),
            SimpleNamespace(content=[SimpleNamespace(text="second result")]),
        ]
    )
    tool_calls = [
        _make_builtin_tool_call("web_search_preview", "call_first", {"query": "first"}),
        _make_builtin_tool_call(
            "web_search_preview", "call_second", {"query": "second"}
        ),
    ]
    context = _make_context(None, available_tools=["browser"])
    context.response_messages.extend(tool_calls)
    context._tool_sessions = {"browser": browser}

    outputs = await context.call_tool()

    assert browser.call_tool.await_args_list == [
        call("search", {"query": "first"}),
        call("search", {"query": "second"}),
    ]
    assert [output.output for output in outputs] == ["first result", "second result"]


@pytest.mark.asyncio
async def test_call_tool_does_not_cross_generation_boundary():
    initial_call = _make_builtin_tool_call(
        "web_search_preview", "call_initial", {"query": "initial"}
    )
    current_call = _make_builtin_tool_call(
        "web_search_preview", "call_current", {"query": "current"}
    )
    browser = MagicMock()
    browser.call_tool = AsyncMock(
        return_value=SimpleNamespace(content=[SimpleNamespace(text="current result")])
    )
    context = _make_context(
        None,
        response_messages=[initial_call],
        available_tools=["browser"],
    )
    context.response_messages.append(current_call)
    context._tool_sessions = {"browser": browser}

    outputs = await context.call_tool()
    context.append_tool_output(outputs)

    browser.call_tool.assert_awaited_once_with("search", {"query": "current"})
    assert [output.call_id for output in outputs] == ["call_current"]
    assert not context.need_builtin_tool_call()


@pytest.mark.parametrize(
    "tool_calls",
    [
        [
            _make_builtin_tool_call(
                "web_search_preview", "call_search", {"query": "vLLM"}
            ),
            _make_builtin_tool_call(
                "get_weather", "call_weather", {"location": "Paris"}
            ),
        ],
        [
            _make_builtin_tool_call(
                "get_weather", "call_weather", {"location": "Paris"}
            ),
            _make_builtin_tool_call(
                "web_search_preview", "call_search", {"query": "vLLM"}
            ),
        ],
    ],
)
def test_mixed_builtin_and_client_tool_turn_is_rejected(tool_calls):
    context = _make_context(None, available_tools=["browser"])
    context.response_messages.extend(tool_calls)

    with pytest.raises(ValueError, match="cannot mix"):
        context.need_builtin_tool_call()


@pytest.mark.asyncio
async def test_runtime_failure_stops_later_tool_dispatch():
    browser = MagicMock()
    browser.call_tool = AsyncMock(
        side_effect=[
            SimpleNamespace(content=[SimpleNamespace(text="first result")]),
            RuntimeError("tool failed"),
            SimpleNamespace(content=[SimpleNamespace(text="third result")]),
        ]
    )
    tool_calls = [
        _make_builtin_tool_call(
            "web_search_preview", f"call_{index}", {"query": str(index)}
        )
        for index in range(3)
    ]
    context = _make_context(None, available_tools=["browser"])
    context.response_messages.extend(tool_calls)
    context._tool_sessions = {"browser": browser}

    with pytest.raises(RuntimeError, match="tool failed"):
        await context.call_tool()

    assert browser.call_tool.await_count == 2


@pytest.mark.asyncio
async def test_local_tool_receives_each_parallel_call_explicitly():
    processed_code = []

    async def process(message):
        processed_code.append(message.content[0].text)
        yield SimpleNamespace(content=[SimpleNamespace(text="result")])

    python_tool = object.__new__(HarmonyPythonTool)
    python_tool.python_tool = MagicMock(process=process)
    tool_calls = [
        _make_builtin_tool_call("code_interpreter", "call_first", {"code": "print(1)"}),
        _make_builtin_tool_call(
            "code_interpreter", "call_second", {"code": "print(2)"}
        ),
    ]
    context = _make_context(None, available_tools=["python"])
    context.response_messages.extend(tool_calls)
    context._tool_sessions = {"python": python_tool}

    outputs = await context.call_tool()

    assert processed_code == ["print(1)", "print(2)"]
    assert [output.call_id for output in outputs] == [
        "call_first",
        "call_second",
    ]


def test_make_response_output_items_pairs_parallel_results_by_call_id():
    tool_calls = [
        _make_builtin_tool_call("web_search_preview", "call_search", {"query": "vLLM"}),
        _make_builtin_tool_call(
            "code_interpreter", "call_python", {"code": "print(42)"}
        ),
    ]
    tool_outputs = [
        ResponseFunctionToolCallOutputItem(
            id="fco_search",
            call_id="call_search",
            type="function_call_output",
            output="search result",
            status="completed",
        ),
        ResponseFunctionToolCallOutputItem(
            id="fco_python",
            call_id="call_python",
            type="function_call_output",
            output="python result",
            status="completed",
        ),
    ]
    context = _make_context(None)
    context.response_messages.extend([*tool_calls, *tool_outputs])

    outputs = context.make_response_output_items()

    assert all(isinstance(output, McpCall) for output in outputs)
    assert [(output.name, output.output) for output in outputs] == [
        ("web_search_preview", "search result"),
        ("code_interpreter", "python result"),
    ]


# ---------------------------------------------------------------------------
# Tests: finish_reason tracking
# ---------------------------------------------------------------------------


def test_finish_reason_tracked():
    """finish_reason from CompletionOutput is stored on the context."""
    ctx = _make_context(_NoOpParser)
    assert ctx.finish_reason is None

    ctx.append_output(_make_request_output(finish_reason="stop"))
    assert ctx.finish_reason == "stop"

    ctx.append_output(_make_request_output(finish_reason="length"))
    assert ctx.finish_reason == "length"


# ---------------------------------------------------------------------------
# Tests: multi-turn accumulation
# ---------------------------------------------------------------------------


def test_multi_turn_accumulation():
    """Multiple append_output() calls accumulate response_messages."""
    ctx = _make_context(_NoOpParser)

    ctx.append_output(_make_request_output(text="First turn"))
    ctx.append_output(_make_request_output(text="Second turn"))

    assert len(ctx.response_messages) == 2
    texts = [m.content[0].text for m in ctx.response_messages]
    assert texts == ["First turn", "Second turn"]


def test_num_init_messages_offset():
    """Initial messages are preserved and offset works correctly."""
    init_messages = [MagicMock(type="message")]
    ctx = _make_context(_NoOpParser, response_messages=init_messages)

    assert ctx.num_init_messages == 1

    ctx.append_output(_make_request_output(text="New output"))

    assert len(ctx.response_messages) == 2
    items = ctx.make_response_output_items()
    assert len(items) == 1
    assert items[0].type == "message"
