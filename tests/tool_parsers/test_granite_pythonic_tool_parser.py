# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for GranitePythonicToolParser.

Covers:
- Single tool call
- Multiple sequential tool calls
- Tool call with no arguments
- Mixed content + tool calls
- Plain text passthrough (no tool call)
- Streaming incremental output
- Parser registration via ToolParserManager

These tests are intentionally tokenizer-free: the Granite pythonic format
does not rely on special tokens, so we pass a minimal mock tokenizer.
"""

from unittest.mock import MagicMock

import pytest

from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.granite_pythonic_tool_parser import (
    GranitePythonicToolParser,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_tokenizer():
    """Minimal tokenizer mock — the pythonic parser does not use it."""
    return MagicMock()


@pytest.fixture()
def parser(mock_tokenizer):
    return GranitePythonicToolParser(mock_tokenizer)


@pytest.fixture()
def mock_request():
    req = MagicMock()
    req.tools = []
    return req


# ---------------------------------------------------------------------------
# Batch (non-streaming) tests
# ---------------------------------------------------------------------------

class TestExtractToolCalls:
    def test_single_call(self, parser, mock_request):
        """A single Python-style call should yield one ToolCall."""
        output = 'get_weather(location="San Francisco", unit="celsius")'
        result = parser.extract_tool_calls(output, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.type == "function"
        assert tc.function.name == "get_weather"
        import json
        args = json.loads(tc.function.arguments)
        assert args == {"location": "San Francisco", "unit": "celsius"}
        assert result.content is None  # no leftover text

    def test_multiple_calls(self, parser, mock_request):
        """Multiple calls on consecutive lines should yield multiple ToolCalls."""
        output = (
            'get_weather(location="Paris")\n'
            'search_web(query="vLLM release notes")'
        )
        result = parser.extract_tool_calls(output, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[1].function.name == "search_web"

    def test_no_args_call(self, parser, mock_request):
        """A function call with no arguments should produce empty JSON object."""
        output = "list_files()"
        result = parser.extract_tool_calls(output, mock_request)

        assert result.tools_called is True
        tc = result.tool_calls[0]
        assert tc.function.name == "list_files"
        import json
        assert json.loads(tc.function.arguments) == {}

    def test_plain_text_passthrough(self, parser, mock_request):
        """Output without a function call should be returned as plain content."""
        output = "I don't need to call any tools to answer this."
        result = parser.extract_tool_calls(output, mock_request)

        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content == output

    def test_mixed_content_and_call(self, parser, mock_request):
        """Text before/after a tool call should appear in content."""
        output = "Sure, let me check the weather.\nget_weather(location=\"London\")"
        result = parser.extract_tool_calls(output, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert "Sure, let me check" in (result.content or "")


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------

class TestExtractToolCallsStreaming:
    def _stream(self, parser, mock_request, full_text: str):
        """Simulate streaming by feeding one character at a time."""
        messages = []
        accumulated = ""
        for char in full_text:
            prev = accumulated
            accumulated += char
            msg = parser.extract_tool_calls_streaming(
                previous_text=prev,
                current_text=accumulated,
                delta_text=char,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[],
                request=mock_request,
            )
            if msg is not None:
                messages.append(msg)
        return messages

    def test_streaming_single_call(self, parser, mock_request):
        full = 'get_weather(location="Berlin")\n'
        messages = self._stream(parser, mock_request, full)

        tool_call_msgs = [
            m for m in messages if m.tool_calls
        ]
        assert len(tool_call_msgs) >= 1, "Expected at least one DeltaToolCall"
        combined_name = "".join(
            tc.function.get("name", "")
            for m in tool_call_msgs
            for tc in (m.tool_calls or [])
        )
        assert "get_weather" in combined_name

    def test_streaming_plain_text(self, parser, mock_request):
        """Plain text should come back as content, no tool_calls."""
        full = "Hello, how can I help you today?\n"
        messages = self._stream(parser, mock_request, full)

        tool_call_msgs = [m for m in messages if m.tool_calls]
        assert len(tool_call_msgs) == 0


# ---------------------------------------------------------------------------
# Registration test
# ---------------------------------------------------------------------------

def test_registered():
    """GranitePythonicToolParser must be reachable via the manager."""
    assert (
        ToolParserManager.get_tool_parser("granite_pythonic")
        is GranitePythonicToolParser
    )
