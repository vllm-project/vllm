# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Anthropic-to-OpenAI request conversion.

Tests the image source handling and tool_result content parsing in
AnthropicServingMessages._convert_anthropic_to_openai_request().
"""

import pytest

from vllm.entrypoints.anthropic.protocol import (
    AnthropicContentBlock,
    AnthropicMessage,
    AnthropicMessagesRequest,
)
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages


# ---------------------------------------------------------------------------
# Helper: build a minimal instance without the full __init__ dependency chain
# ---------------------------------------------------------------------------

def _build_handler() -> AnthropicServingMessages:
    """Create a bare AnthropicServingMessages for unit-testing conversion."""
    instance = object.__new__(AnthropicServingMessages)
    return instance


def _make_request(
    messages: list[dict],
    **kwargs,
) -> AnthropicMessagesRequest:
    return AnthropicMessagesRequest(
        model="test-model",
        max_tokens=128,
        messages=messages,
        **kwargs,
    )


# ======================================================================
# _convert_image_source_to_url
# ======================================================================


class TestConvertImageSourceToUrl:

    def test_base64_source(self):
        source = {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": "iVBORw0KGgo=",
        }
        url = AnthropicServingMessages._convert_image_source_to_url(source)
        assert url == "data:image/jpeg;base64,iVBORw0KGgo="

    def test_base64_png(self):
        source = {
            "type": "base64",
            "media_type": "image/png",
            "data": "AAAA",
        }
        url = AnthropicServingMessages._convert_image_source_to_url(source)
        assert url == "data:image/png;base64,AAAA"

    def test_url_source(self):
        source = {
            "type": "url",
            "url": "https://example.com/image.jpg",
        }
        url = AnthropicServingMessages._convert_image_source_to_url(source)
        assert url == "https://example.com/image.jpg"

    def test_missing_type_defaults_to_base64(self):
        """When 'type' is absent, treat as base64 (Gemini bot review fix)."""
        source = {
            "media_type": "image/webp",
            "data": "UklGR",
        }
        url = AnthropicServingMessages._convert_image_source_to_url(source)
        assert url == "data:image/webp;base64,UklGR"

    def test_missing_media_type_defaults_to_jpeg(self):
        source = {"type": "base64", "data": "abc123"}
        url = AnthropicServingMessages._convert_image_source_to_url(source)
        assert url == "data:image/jpeg;base64,abc123"

    def test_url_source_missing_url_returns_empty(self):
        source = {"type": "url"}
        url = AnthropicServingMessages._convert_image_source_to_url(source)
        assert url == ""

    def test_empty_source_returns_data_uri_shell(self):
        source: dict = {}
        url = AnthropicServingMessages._convert_image_source_to_url(source)
        assert url == "data:image/jpeg;base64,"


# ======================================================================
# Image blocks inside user messages
# ======================================================================


class TestImageContentBlocks:

    def test_base64_image_in_user_message(self):
        handler = _build_handler()
        request = _make_request([
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": "iVBORw0KGgo=",
                        },
                    },
                ],
            }
        ])

        result = handler._convert_anthropic_to_openai_request(request)
        user_msg = result.messages[0]
        assert user_msg["role"] == "user"

        parts = user_msg["content"]
        assert len(parts) == 2
        assert parts[0] == {"type": "text", "text": "Describe this image"}
        assert parts[1] == {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,iVBORw0KGgo="},
        }

    def test_url_image_in_user_message(self):
        handler = _build_handler()
        request = _make_request([
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": "https://example.com/cat.png",
                        },
                    },
                ],
            }
        ])

        result = handler._convert_anthropic_to_openai_request(request)
        parts = result.messages[0]["content"]
        assert parts[1] == {
            "type": "image_url",
            "image_url": {"url": "https://example.com/cat.png"},
        }


# ======================================================================
# tool_result content handling
# ======================================================================


class TestToolResultContent:

    @pytest.fixture()
    def handler(self):
        return _build_handler()

    def _make_tool_result_request(
        self, tool_result_content
    ) -> AnthropicMessagesRequest:
        """Build a request with assistant tool_use followed by user
        tool_result."""
        return _make_request([
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_001",
                        "name": "read_file",
                        "input": {"path": "/tmp/img.png"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_001",
                        "content": tool_result_content,
                    }
                ],
            },
        ])

    def test_tool_result_string_content(self, handler):
        request = self._make_tool_result_request("file contents here")
        result = handler._convert_anthropic_to_openai_request(request)

        tool_msg = [m for m in result.messages if m["role"] == "tool"]
        assert len(tool_msg) == 1
        assert tool_msg[0]["content"] == "file contents here"
        assert tool_msg[0]["tool_call_id"] == "call_001"

    def test_tool_result_text_blocks(self, handler):
        request = self._make_tool_result_request([
            {"type": "text", "text": "line 1"},
            {"type": "text", "text": "line 2"},
        ])
        result = handler._convert_anthropic_to_openai_request(request)

        tool_msg = [m for m in result.messages if m["role"] == "tool"]
        assert len(tool_msg) == 1
        assert tool_msg[0]["content"] == "line 1\nline 2"

    def test_tool_result_with_image(self, handler):
        """Image in tool_result should produce a follow-up user message."""
        request = self._make_tool_result_request([
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "AAAA",
                },
            }
        ])
        result = handler._convert_anthropic_to_openai_request(request)

        tool_msg = [m for m in result.messages if m["role"] == "tool"]
        assert len(tool_msg) == 1
        assert tool_msg[0]["content"] == ""

        # The image should be injected as a follow-up user message
        follow_up = [
            m
            for m in result.messages
            if m["role"] == "user"
            and isinstance(m.get("content"), list)
        ]
        assert len(follow_up) == 1
        img_parts = follow_up[0]["content"]
        assert len(img_parts) == 1
        assert img_parts[0] == {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,AAAA"},
        }

    def test_tool_result_with_text_and_image(self, handler):
        """Mixed text+image tool_result: text in tool msg, image in user
        msg."""
        request = self._make_tool_result_request([
            {"type": "text", "text": "Here is the screenshot"},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": "QUFB",
                },
            },
        ])
        result = handler._convert_anthropic_to_openai_request(request)

        tool_msg = [m for m in result.messages if m["role"] == "tool"]
        assert len(tool_msg) == 1
        assert tool_msg[0]["content"] == "Here is the screenshot"

        follow_up = [
            m
            for m in result.messages
            if m["role"] == "user"
            and isinstance(m.get("content"), list)
        ]
        assert len(follow_up) == 1
        assert follow_up[0]["content"][0]["image_url"]["url"] == (
            "data:image/jpeg;base64,QUFB"
        )

    def test_tool_result_with_multiple_images(self, handler):
        request = self._make_tool_result_request([
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "IMG1",
                },
            },
            {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": "https://example.com/img2.jpg",
                },
            },
        ])
        result = handler._convert_anthropic_to_openai_request(request)

        follow_up = [
            m
            for m in result.messages
            if m["role"] == "user"
            and isinstance(m.get("content"), list)
        ]
        assert len(follow_up) == 1
        urls = [p["image_url"]["url"] for p in follow_up[0]["content"]]
        assert urls == [
            "data:image/png;base64,IMG1",
            "https://example.com/img2.jpg",
        ]

    def test_tool_result_none_content(self, handler):
        request = self._make_tool_result_request(None)
        result = handler._convert_anthropic_to_openai_request(request)

        tool_msg = [m for m in result.messages if m["role"] == "tool"]
        assert len(tool_msg) == 1
        assert tool_msg[0]["content"] == ""

    def test_tool_result_no_follow_up_when_no_images(self, handler):
        """Ensure no extra user message is added when there are no images."""
        request = self._make_tool_result_request([
            {"type": "text", "text": "just text"},
        ])
        result = handler._convert_anthropic_to_openai_request(request)

        user_follow_ups = [
            m
            for m in result.messages
            if m["role"] == "user"
            and isinstance(m.get("content"), list)
        ]
        assert len(user_follow_ups) == 0
