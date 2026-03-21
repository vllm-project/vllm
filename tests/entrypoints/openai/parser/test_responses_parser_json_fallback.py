# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the JSON fallback parsing in ResponsesParser.process().

When tool_choice="required" and guided generation forces the model to output
a JSON array of {name, parameters} objects (instead of native tool-call
tokens), the native tool parser won't recognise them.  The fallback path
parses the JSON directly.
"""

import json
from unittest.mock import MagicMock

import pytest
from openai.types.responses import ResponseFunctionToolCall

from vllm.entrypoints.openai.parser.responses_parser import ResponsesParser
from vllm.outputs import CompletionOutput

pytestmark = pytest.mark.cpu_test


def _make_parser(*, tool_choice="required", has_tool_parser=False):
    """Create a ResponsesParser with mocked dependencies."""
    tokenizer = MagicMock()

    # Mock reasoning parser that returns (None, content) — no reasoning
    reasoning_parser = MagicMock()
    reasoning_parser.extract_reasoning = lambda text, request=None: (None, text)
    reasoning_parser_cls = MagicMock(return_value=reasoning_parser)

    # Mock tool parser that never finds tool calls (simulating native parser failure)
    tool_parser = None
    tool_parser_cls = None
    if has_tool_parser:
        tool_parser = MagicMock()
        tool_parser.extract_tool_calls = MagicMock(return_value=MagicMock(
            tools_called=False,
            tool_calls=[],
            content=None,
        ))
        tool_parser_cls = MagicMock(return_value=tool_parser)

    request = MagicMock()
    request.tool_choice = tool_choice

    parser = ResponsesParser(
        tokenizer=tokenizer,
        reasoning_parser_cls=reasoning_parser_cls,
        response_messages=[],
        request=request,
        tool_parser_cls=tool_parser_cls,
    )
    return parser


def _make_output(text: str) -> CompletionOutput:
    return CompletionOutput(
        index=0,
        text=text,
        token_ids=(),
        cumulative_logprob=None,
        logprobs=None,
        finish_reason="stop",
    )


class TestJsonFallbackParsing:

    def test_fallback_parses_json_array(self):
        """tool_choice=required + content is JSON array → parse as function_calls."""
        parser = _make_parser(tool_choice="required")
        content = json.dumps([
            {"name": "get_weather", "parameters": {"city": "NYC"}},
        ])
        output = _make_output(content)
        parser.process(output)

        # Should have 1 function call, no text message
        msgs = parser.response_messages
        func_calls = [m for m in msgs if isinstance(m, ResponseFunctionToolCall)]
        assert len(func_calls) == 1
        assert func_calls[0].name == "get_weather"
        assert json.loads(func_calls[0].arguments) == {"city": "NYC"}
        # No text message since content was consumed
        text_msgs = [m for m in msgs if not isinstance(m, ResponseFunctionToolCall)]
        assert len(text_msgs) == 0

    def test_fallback_parses_multiple_tools(self):
        """JSON array with multiple tool calls."""
        parser = _make_parser(tool_choice="required")
        content = json.dumps([
            {"name": "fn_a", "parameters": {"x": 1}},
            {"name": "fn_b", "parameters": {"y": 2}},
        ])
        output = _make_output(content)
        parser.process(output)

        func_calls = [m for m in parser.response_messages
                      if isinstance(m, ResponseFunctionToolCall)]
        assert len(func_calls) == 2
        assert func_calls[0].name == "fn_a"
        assert func_calls[1].name == "fn_b"

    def test_fallback_skipped_when_tool_choice_auto(self):
        """tool_choice=auto → no fallback even if content is JSON."""
        parser = _make_parser(tool_choice="auto")
        content = json.dumps([{"name": "fn", "parameters": {}}])
        output = _make_output(content)
        parser.process(output)

        # Should be a text message, not a function call
        func_calls = [m for m in parser.response_messages
                      if isinstance(m, ResponseFunctionToolCall)]
        assert len(func_calls) == 0

    def test_fallback_skipped_when_native_parser_succeeds(self):
        """Native parser found tool calls → fallback not triggered."""
        parser = _make_parser(tool_choice="required", has_tool_parser=True)
        # Override the mock to return a tool call
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "native_fn"
        mock_tool_call.function.arguments = '{}'
        parser.tool_parser_instance.extract_tool_calls.return_value = MagicMock(
            tools_called=True,
            tool_calls=[mock_tool_call],
            content=None,
        )

        content = json.dumps([{"name": "fallback_fn", "parameters": {}}])
        output = _make_output(content)
        parser.process(output)

        # Should have the native call, not the fallback
        func_calls = [m for m in parser.response_messages
                      if isinstance(m, ResponseFunctionToolCall)]
        assert len(func_calls) == 1
        assert func_calls[0].name == "native_fn"

    def test_fallback_handles_invalid_json(self):
        """Malformed JSON content → gracefully ignored, becomes text message."""
        parser = _make_parser(tool_choice="required")
        content = "not valid json {"
        output = _make_output(content)
        parser.process(output)

        func_calls = [m for m in parser.response_messages
                      if isinstance(m, ResponseFunctionToolCall)]
        assert len(func_calls) == 0
        # Content preserved as text message
        assert len(parser.response_messages) == 1

    def test_fallback_handles_json_object_not_array(self):
        """JSON object (not array) → ignored, becomes text message."""
        parser = _make_parser(tool_choice="required")
        content = json.dumps({"name": "fn", "parameters": {}})
        output = _make_output(content)
        parser.process(output)

        func_calls = [m for m in parser.response_messages
                      if isinstance(m, ResponseFunctionToolCall)]
        assert len(func_calls) == 0

    def test_fallback_skipped_when_content_is_none(self):
        """No content → fallback not triggered."""
        parser = _make_parser(tool_choice="required")
        # Mock reasoning parser to return no content
        parser.reasoning_parser_instance.extract_reasoning = (
            lambda text, request=None: (None, None)
        )
        output = _make_output("")
        parser.process(output)

        assert len(parser.response_messages) == 0
