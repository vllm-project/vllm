# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for vllm.tool_parsers.gemma4_utils — offline inference tool parser."""

import pytest

from vllm.tool_parsers.gemma4_utils import (
    _parse_tool_arguments,
    has_tool_response_tag,
    parse_tool_calls,
)


# ---------------------------------------------------------------------------
# _parse_tool_arguments
# ---------------------------------------------------------------------------


class TestParseToolArguments:
    """Unit tests for the offline tool argument parser."""

    def test_simple_string(self):
        result = _parse_tool_arguments('location:<|"|>Tokyo<|"|>')
        assert result["location"] == "Tokyo"

    def test_multiple_string_values(self):
        result = _parse_tool_arguments(
            'location:<|"|>San Francisco<|"|>,unit:<|"|>celsius<|"|>'
        )
        assert result == {"location": "San Francisco", "unit": "celsius"}

    def test_string_with_internal_quotes(self):
        """Regression: internal " must not truncate the value."""
        result = _parse_tool_arguments(
            'content:<|"|>She said "hello" loudly<|"|>'
        )
        assert result["content"] == 'She said "hello" loudly'

    def test_html_with_quoted_attributes(self):
        """Regression: HTML attributes like class="main" must be preserved."""
        result = _parse_tool_arguments(
            'path:<|"|>out.html<|"|>,'
            'content:<|"|><div class="main">hello</div><|"|>'
        )
        assert result["path"] == "out.html"
        assert result["content"] == '<div class="main">hello</div>'

    def test_string_with_braces(self):
        """String values containing { and } must be preserved."""
        result = _parse_tool_arguments(
            'content:<|"|><html><div>{test}</div></html><|"|>'
        )
        assert result["content"] == "<html><div>{test}</div></html>"

    def test_string_with_mixed_special_chars(self):
        """Values with quotes, braces, and angle brackets."""
        result = _parse_tool_arguments(
            'code:<|"|>function() { return "ok"; }<|"|>'
        )
        assert result["code"] == 'function() { return "ok"; }'

    def test_empty_string(self):
        assert _parse_tool_arguments("") == {}

    def test_whitespace_only(self):
        assert _parse_tool_arguments("   ") == {}

    def test_bare_numeric_value(self):
        """Bare (non-delimited) numeric values."""
        result = _parse_tool_arguments("count:42")
        # Should return the value (as string is acceptable for offline parser)
        assert "count" in result

    def test_multiline_content(self):
        """Multi-line string values (e.g., code blocks)."""
        result = _parse_tool_arguments(
            'content:<|"|>line1\nline2\nline3<|"|>'
        )
        assert "line1" in result["content"]
        assert "line3" in result["content"]


# ---------------------------------------------------------------------------
# parse_tool_calls (end-to-end)
# ---------------------------------------------------------------------------


class TestParseToolCalls:
    """End-to-end tests for parse_tool_calls."""

    def test_standard_format(self):
        text = '<|tool_call>call:get_weather{location:<|"|>London<|"|>}<tool_call|>'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        assert calls[0]["arguments"]["location"] == "London"

    def test_standard_format_with_internal_quotes(self):
        """Regression: tool call with internal quotes in string value."""
        text = (
            "<|tool_call>call:write_file{"
            'path:<|"|>index.html<|"|>,'
            'content:<|"|><div class="app">Hello</div><|"|>'
            "}<tool_call|>"
        )
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "write_file"
        assert calls[0]["arguments"]["path"] == "index.html"
        assert calls[0]["arguments"]["content"] == '<div class="app">Hello</div>'

    def test_multiple_tool_calls(self):
        text = (
            '<|tool_call>call:read_file{path:<|"|>a.txt<|"|>}<tool_call|>'
            '<|tool_call>call:read_file{path:<|"|>b.txt<|"|>}<tool_call|>'
        )
        calls = parse_tool_calls(text)
        assert len(calls) == 2

    def test_no_tool_calls(self):
        assert parse_tool_calls("Hello, how can I help?") == []

    def test_strict_mode_ignores_fallback(self):
        text = "call:get_weather{location:Tokyo}"
        assert parse_tool_calls(text, strict=True) == []
        assert len(parse_tool_calls(text, strict=False)) >= 1


# ---------------------------------------------------------------------------
# has_tool_response_tag
# ---------------------------------------------------------------------------


class TestHasToolResponseTag:
    def test_with_tag(self):
        assert has_tool_response_tag("some text <|tool_response>") is True

    def test_without_tag(self):
        assert has_tool_response_tag("some text <eos>") is False

    def test_with_trailing_whitespace(self):
        assert has_tool_response_tag("some text <|tool_response>  ") is True
