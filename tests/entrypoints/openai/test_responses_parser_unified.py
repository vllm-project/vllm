# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ResponsesParser with the unified Parser interface.

These tests verify that ResponsesParser correctly delegates to the unified
Parser (via parse) instead of calling separate ReasoningParser / ToolParser
instances directly.
"""

from collections.abc import Sequence
from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.parser.responses_parser import (
    ResponsesParser,
    get_responses_parser_for_simple_context,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.outputs import CompletionOutput
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(**overrides) -> ResponsesRequest:
    defaults = {"model": "test-model", "input": "test"}
    defaults.update(overrides)
    return ResponsesRequest.model_validate(defaults)


def _make_output(
    text: str = "Hello, world!",
    token_ids: Sequence[int] = (1, 2, 3),
    finish_reason: str = "stop",
) -> CompletionOutput:
    return CompletionOutput(
        index=0,
        text=text,
        token_ids=list(token_ids),
        cumulative_logprob=None,
        logprobs=None,
        finish_reason=finish_reason,
    )


def _make_parser(parser_cls, **overrides):
    defaults = dict(
        tokenizer=MagicMock(),
        parser_cls=parser_cls,
        response_messages=[],
        request=_make_request(),
        chat_template=None,
        chat_template_content_format="auto",
    )
    defaults.update(overrides)
    return ResponsesParser(**defaults)


# ---------------------------------------------------------------------------
# Tests: basic text passthrough
# ---------------------------------------------------------------------------


def test_process_text_with_parser():
    """Parser with no reasoning/tools returns a single message item."""
    parser = _make_parser(_NoOpParser)
    parser.process(_make_output(text="Hello!"))

    assert len(parser.response_messages) == 1
    msg = parser.response_messages[0]
    assert msg.type == "message"
    assert msg.content[0].text == "Hello!"


def test_process_text_without_parser():
    """parser_cls=None falls back to plain text wrapping."""
    parser = _make_parser(None)
    parser.process(_make_output(text="Hello!"))

    assert len(parser.response_messages) == 1
    msg = parser.response_messages[0]
    assert msg.type == "message"
    assert msg.content[0].text == "Hello!"


# ---------------------------------------------------------------------------
# Tests: empty / whitespace output
# ---------------------------------------------------------------------------


def test_process_empty_text_without_parser():
    """Empty text with no parser produces no output items."""
    parser = _make_parser(None)
    parser.process(_make_output(text=""))

    assert len(parser.response_messages) == 0


def test_process_empty_text_with_parser():
    """Empty text with parser produces no output items."""
    parser = _make_parser(_NoOpParser)
    parser.process(_make_output(text=""))

    assert len(parser.response_messages) == 0


# ---------------------------------------------------------------------------
# Tests: reasoning extraction
# ---------------------------------------------------------------------------


def test_process_extracts_reasoning():
    """Parser that finds reasoning produces both reasoning and message items."""
    parser = _make_parser(_ReasoningOnlyParser)
    parser.process(_make_output(text="<think>Let me check</think>The answer is 42"))

    types = [m.type for m in parser.response_messages]
    assert "reasoning" in types
    assert "message" in types

    reasoning_item = next(m for m in parser.response_messages if m.type == "reasoning")
    assert reasoning_item.content[0].text == "Let me check"

    message_item = next(m for m in parser.response_messages if m.type == "message")
    assert message_item.content[0].text == "The answer is 42"


def test_process_reasoning_only_no_content():
    """When reasoning consumes all text, only a reasoning item is produced."""
    parser = _make_parser(_ReasoningOnlyParser)
    parser.process(_make_output(text="<think>Just thinking</think>"))

    types = [m.type for m in parser.response_messages]
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
    parser = _make_parser(_ToolCallingParser, request=request, enable_auto_tools=True)
    parser.process(_make_output(text="calling tool"))

    types = [m.type for m in parser.response_messages]
    assert "function_call" in types

    tool_item = next(m for m in parser.response_messages if m.type == "function_call")
    assert tool_item.name == "get_weather"
    assert tool_item.arguments == '{"location": "Paris"}'
    assert tool_item.status == "completed"


# ---------------------------------------------------------------------------
# Tests: finish_reason tracking
# ---------------------------------------------------------------------------


def test_finish_reason_tracked():
    """finish_reason from CompletionOutput is stored on the parser."""
    parser = _make_parser(_NoOpParser)
    assert parser.finish_reason is None

    parser.process(_make_output(finish_reason="stop"))
    assert parser.finish_reason == "stop"

    parser.process(_make_output(finish_reason="length"))
    assert parser.finish_reason == "length"


# ---------------------------------------------------------------------------
# Tests: multi-turn accumulation
# ---------------------------------------------------------------------------


def test_multi_turn_accumulation():
    """Multiple process() calls accumulate response_messages."""
    parser = _make_parser(_NoOpParser)

    parser.process(_make_output(text="First turn"))
    parser.process(_make_output(text="Second turn"))

    assert len(parser.response_messages) == 2
    texts = [m.content[0].text for m in parser.response_messages]
    assert texts == ["First turn", "Second turn"]


def test_num_init_messages_offset():
    """Initial messages are preserved and offset works correctly."""
    init_messages = [MagicMock(type="message")]
    parser = _make_parser(_NoOpParser, response_messages=init_messages)

    assert parser.num_init_messages == 1

    parser.process(_make_output(text="New output"))

    assert len(parser.response_messages) == 2
    items = parser.make_response_output_items_from_parsable_context()
    assert len(items) == 1
    assert items[0].type == "message"


# ---------------------------------------------------------------------------
# Tests: factory function
# ---------------------------------------------------------------------------


def test_factory_function_creates_parser():
    """get_responses_parser_for_simple_context returns a working parser."""
    rp = get_responses_parser_for_simple_context(
        tokenizer=MagicMock(),
        parser_cls=_NoOpParser,
        response_messages=[],
        request=_make_request(),
        chat_template=None,
        chat_template_content_format="auto",
    )
    assert isinstance(rp, ResponsesParser)

    rp.process(_make_output(text="Works!"))
    assert len(rp.response_messages) == 1


def test_factory_function_none_parser():
    """Factory function works with parser_cls=None."""
    rp = get_responses_parser_for_simple_context(
        tokenizer=MagicMock(),
        parser_cls=None,
        response_messages=[],
        request=_make_request(),
        chat_template=None,
        chat_template_content_format="auto",
    )
    assert isinstance(rp, ResponsesParser)
    assert rp.parser_instance is None
