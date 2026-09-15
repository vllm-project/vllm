# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ParsableContext's parsing behavior.

These tests verify that ParsableContext correctly delegates to the unified
Parser (via parse) and properly builds response output items.
"""

import json
from collections.abc import Sequence
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types.responses import ResponseFunctionToolCall

from vllm.entrypoints.generate.base.protocol import (
    DeltaMessage,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.mcp.tool import Tool
from vllm.entrypoints.openai.responses import context as responses_context
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

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        return input_ids

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

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        return input_ids

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

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        return input_ids

    def extract_reasoning(self, model_output, request):
        return None, model_output

    def extract_reasoning_streaming(self, *args, **kwargs):
        return None

    def extract_tool_calls_streaming(self, *args, **kwargs):
        return None

    def parse_delta(self, *args, **kwargs) -> DeltaMessage | None:
        return None


class _JsonParsingTool(Tool):
    def __init__(self):
        self.called = False

    async def get_result(self, context):
        raise NotImplementedError

    async def get_result_parsable_context(self, context):
        self.called = True
        last_msg = context.response_messages[-1]
        json.loads(last_msg.arguments)
        return []


_TOOL_CASES = (
    ("code_interpreter", "python", "python", {"code": "print(6 * 7)"}),
    ("web_search_preview", "browser", "search", {"query": "vLLM"}),
    ("container.exec", "container", "exec", {"cmd": ["pwd"]}),
)
_INVALID_JSON = '{"unterminated":'


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


def _make_tool_context(tool_case, tool_session, arguments):
    tool_name, session_name, _, _ = tool_case
    ctx = _make_context(None)
    tool_call = ResponseFunctionToolCall(
        id=f"fc_{tool_name}",
        call_id=f"call_{tool_name}",
        type="function_call",
        name=tool_name,
        arguments=arguments,
    )
    ctx.response_messages.append(tool_call)
    ctx._tool_sessions[session_name] = tool_session
    return ctx, tool_call


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


@pytest.mark.parametrize("tool_case", _TOOL_CASES)
@pytest.mark.asyncio
async def test_call_tool_valid_json_dispatches_client_session_once(
    monkeypatch, tool_case
):
    monkeypatch.setattr(
        responses_context.envs, "VLLM_TOOL_JSON_ERROR_AUTOMATIC_RETRY", True
    )
    _, session_name, dispatched_name, arguments = tool_case
    tool_session = MagicMock()
    tool_session.call_tool = AsyncMock(
        return_value=SimpleNamespace(content=[SimpleNamespace(text="result")])
    )
    ctx, _ = _make_tool_context(tool_case, tool_session, json.dumps(arguments))

    output = await ctx.call_tool()

    tool_session.call_tool.assert_awaited_once_with(dispatched_name, arguments)
    assert ctx.called_tools == {session_name}
    assert output[0].output == "result"


@pytest.mark.parametrize("tool_case", _TOOL_CASES)
@pytest.mark.parametrize("use_tool", [True, False], ids=["Tool", "ClientSession"])
@pytest.mark.asyncio
async def test_call_tool_invalid_json_returns_retry_before_dispatch(
    monkeypatch, tool_case, use_tool
):
    monkeypatch.setattr(
        responses_context.envs, "VLLM_TOOL_JSON_ERROR_AUTOMATIC_RETRY", True
    )
    _, session_name, _, _ = tool_case
    tool_session = _JsonParsingTool() if use_tool else MagicMock()
    ctx, tool_call = _make_tool_context(tool_case, tool_session, _INVALID_JSON)

    output = await ctx.call_tool()

    retry_item = output[0]
    assert retry_item.call_id == tool_call.call_id
    assert "Error parsing tool arguments as JSON" in retry_item.output
    assert session_name not in ctx.called_tools
    if isinstance(tool_session, _JsonParsingTool):
        assert not tool_session.called
    else:
        tool_session.call_tool.assert_not_called()


@pytest.mark.parametrize(
    ("tool_case", "use_tool"),
    [
        pytest.param(_TOOL_CASES[0], True, id="Tool"),
        pytest.param(_TOOL_CASES[1], False, id="ClientSession"),
    ],
)
@pytest.mark.asyncio
async def test_invalid_builtin_tool_json_raises_when_retry_disabled(
    monkeypatch, tool_case, use_tool
):
    monkeypatch.setattr(
        responses_context.envs,
        "VLLM_TOOL_JSON_ERROR_AUTOMATIC_RETRY",
        False,
    )
    _, session_name, _, _ = tool_case
    tool_session = _JsonParsingTool() if use_tool else MagicMock()
    ctx, _ = _make_tool_context(tool_case, tool_session, _INVALID_JSON)

    with pytest.raises(json.JSONDecodeError):
        await ctx.call_tool()

    assert session_name in ctx.called_tools
    if isinstance(tool_session, _JsonParsingTool):
        assert tool_session.called
    else:
        tool_session.call_tool.assert_not_called()


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
