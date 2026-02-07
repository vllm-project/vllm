# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Anthropic API protocol compliance fixes."""

import json

import pytest

from vllm.entrypoints.anthropic.protocol import (
    AnthropicMessagesResponse,
    AnthropicUsage,
    generate_tool_call_id,
)
from vllm.entrypoints.anthropic.serving import _extract_tool_result_text


class TestMessageIdFormat:
    """Message IDs must use the ``msg_`` prefix per the Anthropic API spec."""

    def test_auto_generated_id_has_msg_prefix(self):
        resp = AnthropicMessagesResponse(
            id="",
            content=[],
            model="test",
            usage=AnthropicUsage(input_tokens=0, output_tokens=0),
        )
        assert resp.id.startswith("msg_")

    def test_openai_style_id_is_replaced(self):
        resp = AnthropicMessagesResponse(
            id="chatcmpl-abc123",
            content=[],
            model="test",
            usage=AnthropicUsage(input_tokens=0, output_tokens=0),
        )
        assert resp.id.startswith("msg_")
        assert "chatcmpl" not in resp.id

    def test_valid_msg_id_preserved(self):
        resp = AnthropicMessagesResponse(
            id="msg_existing123",
            content=[],
            model="test",
            usage=AnthropicUsage(input_tokens=0, output_tokens=0),
        )
        assert resp.id == "msg_existing123"

    def test_unique_ids(self):
        ids = set()
        for _ in range(100):
            resp = AnthropicMessagesResponse(
                id="",
                content=[],
                model="test",
                usage=AnthropicUsage(input_tokens=0, output_tokens=0),
            )
            ids.add(resp.id)
        assert len(ids) == 100


class TestToolCallIdFormat:
    """Tool call IDs must use the ``toolu_`` prefix."""

    def test_has_toolu_prefix(self):
        tid = generate_tool_call_id()
        assert tid.startswith("toolu_")

    def test_unique(self):
        ids = {generate_tool_call_id() for _ in range(100)}
        assert len(ids) == 100


class TestExtractToolResultText:
    """tool_result content can be a string or a list of content blocks."""

    def test_none(self):
        assert _extract_tool_result_text(None) == ""

    def test_plain_string(self):
        assert _extract_tool_result_text("hello") == "hello"

    def test_single_text_block(self):
        content = [{"type": "text", "text": "result"}]
        assert _extract_tool_result_text(content) == "result"

    def test_multiple_text_blocks(self):
        content = [
            {"type": "text", "text": "line1"},
            {"type": "text", "text": "line2"},
        ]
        assert _extract_tool_result_text(content) == "line1\nline2"

    def test_image_block(self):
        content = [{"type": "image", "source": {"data": "base64..."}}]
        assert _extract_tool_result_text(content) == "[image]"

    def test_unknown_block_preserved_as_json(self):
        content = [{"type": "custom", "data": 42}]
        result = _extract_tool_result_text(content)
        parsed = json.loads(result)
        assert parsed == {"type": "custom", "data": 42}

    def test_does_not_produce_python_repr(self):
        """Regression: str([{"type": "text", "text": "hi"}]) produces
        "[{'type': 'text', 'text': 'hi'}]" which is wrong."""
        content = [{"type": "text", "text": "hi"}]
        result = _extract_tool_result_text(content)
        assert result == "hi"
        assert "[{" not in result
