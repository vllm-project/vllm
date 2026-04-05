# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Anthropic-to-OpenAI request conversion.

Tests the image source handling and tool_result content parsing in
AnthropicServingMessages._convert_anthropic_to_openai_request().

Also covers extended-thinking edge cases such as ``redacted_thinking``
blocks echoed back by Anthropic clients.
"""

from vllm.entrypoints.anthropic.protocol import (
    AnthropicMessagesRequest,
)
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages

_convert = AnthropicServingMessages._convert_anthropic_to_openai_request
_img_url = AnthropicServingMessages._convert_image_source_to_url


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
        assert _img_url(source) == "data:image/jpeg;base64,iVBORw0KGgo="

    def test_base64_png(self):
        source = {
            "type": "base64",
            "media_type": "image/png",
            "data": "AAAA",
        }
        assert _img_url(source) == "data:image/png;base64,AAAA"

    def test_url_source(self):
        source = {
            "type": "url",
            "url": "https://example.com/image.jpg",
        }
        assert _img_url(source) == "https://example.com/image.jpg"

    def test_missing_type_defaults_to_base64(self):
        """When 'type' is absent, treat as base64."""
        source = {
            "media_type": "image/webp",
            "data": "UklGR",
        }
        assert _img_url(source) == "data:image/webp;base64,UklGR"

    def test_missing_media_type_defaults_to_jpeg(self):
        source = {"type": "base64", "data": "abc123"}
        assert _img_url(source) == "data:image/jpeg;base64,abc123"

    def test_url_source_missing_url_returns_empty(self):
        source = {"type": "url"}
        assert _img_url(source) == ""

    def test_empty_source_returns_data_uri_shell(self):
        source: dict = {}
        assert _img_url(source) == "data:image/jpeg;base64,"


# ======================================================================
# Image blocks inside user messages
# ======================================================================


class TestImageContentBlocks:
    def test_base64_image_in_user_message(self):
        request = _make_request(
            [
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
            ]
        )

        result = _convert(request)
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
        request = _make_request(
            [
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
            ]
        )

        result = _convert(request)
        parts = result.messages[0]["content"]
        assert parts[1] == {
            "type": "image_url",
            "image_url": {"url": "https://example.com/cat.png"},
        }


# ======================================================================
# tool_result content handling
# ======================================================================


class TestToolResultContent:
    def _make_tool_result_request(
        self, tool_result_content
    ) -> AnthropicMessagesRequest:
        """Build a request with assistant tool_use followed by user
        tool_result."""
        return _make_request(
            [
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
            ]
        )

    def test_tool_result_string_content(self):
        request = self._make_tool_result_request("file contents here")
        result = _convert(request)

        tool_msg = [m for m in result.messages if m["role"] == "tool"]
        assert len(tool_msg) == 1
        assert tool_msg[0]["content"] == "file contents here"
        assert tool_msg[0]["tool_call_id"] == "call_001"

    def test_tool_result_text_blocks(self):
        request = self._make_tool_result_request(
            [
                {"type": "text", "text": "line 1"},
                {"type": "text", "text": "line 2"},
            ]
        )
        result = _convert(request)

        tool_msg = [m for m in result.messages if m["role"] == "tool"]
        assert len(tool_msg) == 1
        assert tool_msg[0]["content"] == "line 1\nline 2"

    def test_tool_result_with_image(self):
        """Image in tool_result should produce a follow-up user message."""
        request = self._make_tool_result_request(
            [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "AAAA",
                    },
                }
            ]
        )
        result = _convert(request)

        tool_msg = [m for m in result.messages if m["role"] == "tool"]
        assert len(tool_msg) == 1
        assert tool_msg[0]["content"] == ""

        # The image should be injected as a follow-up user message
        follow_up = [
            m
            for m in result.messages
            if m["role"] == "user" and isinstance(m.get("content"), list)
        ]
        assert len(follow_up) == 1
        img_parts = follow_up[0]["content"]
        assert len(img_parts) == 1
        assert img_parts[0] == {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,AAAA"},
        }

    def test_tool_result_with_text_and_image(self):
        """Mixed text+image tool_result: text in tool msg, image in user
        msg."""
        request = self._make_tool_result_request(
            [
                {"type": "text", "text": "Here is the screenshot"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "QUFB",
                    },
                },
            ]
        )
        result = _convert(request)

        tool_msg = [m for m in result.messages if m["role"] == "tool"]
        assert len(tool_msg) == 1
        assert tool_msg[0]["content"] == "Here is the screenshot"

        follow_up = [
            m
            for m in result.messages
            if m["role"] == "user" and isinstance(m.get("content"), list)
        ]
        assert len(follow_up) == 1
        assert follow_up[0]["content"][0]["image_url"]["url"] == (
            "data:image/jpeg;base64,QUFB"
        )

    def test_tool_result_with_multiple_images(self):
        request = self._make_tool_result_request(
            [
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
            ]
        )
        result = _convert(request)

        follow_up = [
            m
            for m in result.messages
            if m["role"] == "user" and isinstance(m.get("content"), list)
        ]
        assert len(follow_up) == 1
        urls = [p["image_url"]["url"] for p in follow_up[0]["content"]]
        assert urls == [
            "data:image/png;base64,IMG1",
            "https://example.com/img2.jpg",
        ]

    def test_tool_result_none_content(self):
        request = self._make_tool_result_request(None)
        result = _convert(request)

        tool_msg = [m for m in result.messages if m["role"] == "tool"]
        assert len(tool_msg) == 1
        assert tool_msg[0]["content"] == ""

    def test_tool_result_no_follow_up_when_no_images(self):
        """Ensure no extra user message is added when there are no images."""
        request = self._make_tool_result_request(
            [
                {"type": "text", "text": "just text"},
            ]
        )
        result = _convert(request)

        user_follow_ups = [
            m
            for m in result.messages
            if m["role"] == "user" and isinstance(m.get("content"), list)
        ]
        assert len(user_follow_ups) == 0


# ======================================================================
# Attribution header stripping
# ======================================================================


class TestAttributionHeaderStripping:
    def test_billing_header_stripped_from_system(self):
        """Claude Code's x-anthropic-billing-header block should be
        stripped to preserve prefix caching."""
        request = _make_request(
            [{"role": "user", "content": "Hello"}],
            system=[
                {"type": "text", "text": "You are a helpful assistant."},
                {
                    "type": "text",
                    "text": "x-anthropic-billing-header: "
                    "cc_version=2.1.37.abc; cc_entrypoint=cli;",
                },
            ],
        )
        result = _convert(request)
        system_msg = result.messages[0]
        assert system_msg["role"] == "system"
        assert system_msg["content"] == "You are a helpful assistant."

    def test_system_without_billing_header_unchanged(self):
        """Normal system blocks should pass through unchanged."""
        request = _make_request(
            [{"role": "user", "content": "Hello"}],
            system=[
                {"type": "text", "text": "You are a helpful assistant."},
                {"type": "text", "text": " Be concise."},
            ],
        )
        result = _convert(request)
        system_msg = result.messages[0]
        assert system_msg["content"] == "You are a helpful assistant. Be concise."

    def test_system_string_unchanged(self):
        """String system prompts should pass through unchanged."""
        request = _make_request(
            [{"role": "user", "content": "Hello"}],
            system="You are a helpful assistant.",
        )
        result = _convert(request)
        system_msg = result.messages[0]
        assert system_msg["content"] == "You are a helpful assistant."


# ======================================================================
# Thinking block conversion (Anthropic → OpenAI)
# ======================================================================


class TestThinkingBlockConversion:
    """Verify that thinking blocks in assistant messages are correctly
    moved to the ``reasoning`` field and stripped from ``content`` during
    the Anthropic→OpenAI conversion.

    This is the Anthropic-endpoint path: the client echoes back the full
    assistant message (including thinking blocks emitted by vllm) in
    subsequent requests.
    """

    def test_thinking_plus_text_in_assistant_message(self):
        """thinking + text → reasoning field + plain-string content."""
        request = _make_request(
            [
                {"role": "user", "content": "Write me some code."},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "I should write a simple example.",
                            "signature": "sig_abc123",
                        },
                        {"type": "text", "text": "Sure! Here is the code."},
                    ],
                },
                {"role": "user", "content": "Can you fix the bug?"},
            ]
        )
        result = _convert(request)

        # Find the assistant message in the converted output.
        asst_msgs = [m for m in result.messages if m.get("role") == "assistant"]
        assert len(asst_msgs) == 1
        asst = asst_msgs[0]

        # Thinking content must be in reasoning, NOT in content.
        assert asst.get("reasoning") == "I should write a simple example."
        assert asst.get("content") == "Sure! Here is the code."

    def test_thinking_only_in_assistant_message(self):
        """Assistant message with only a thinking block (no visible text).

        This can happen when the model emits reasoning but no final answer
        yet (e.g. a mid-turn reasoning step).  Content should be None.
        """
        request = _make_request(
            [
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "Just thinking...",
                            "signature": "sig_xyz",
                        }
                    ],
                },
                {"role": "user", "content": "Go on."},
            ]
        )
        result = _convert(request)

        asst_msgs = [m for m in result.messages if m.get("role") == "assistant"]
        assert len(asst_msgs) == 1
        asst = asst_msgs[0]

        assert asst.get("reasoning") == "Just thinking..."
        # No visible text → content should be empty string (not absent),
        # because some tokenizers (e.g. mistral-common) require the field.
        assert asst.get("content") == ""

    def test_thinking_plus_tool_use_in_assistant_message(self):
        """thinking + tool_use: reasoning field set, tool_calls populated."""
        request = _make_request(
            [
                {"role": "user", "content": "What is 2+2?"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "I need to call the calculator.",
                            "signature": "sig_tool",
                        },
                        {
                            "type": "tool_use",
                            "id": "call_001",
                            "name": "calculator",
                            "input": {"expression": "2+2"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_001",
                            "content": "4",
                        }
                    ],
                },
            ]
        )
        result = _convert(request)

        asst_msgs = [m for m in result.messages if m.get("role") == "assistant"]
        assert len(asst_msgs) == 1
        asst = asst_msgs[0]

        assert asst.get("reasoning") == "I need to call the calculator."
        tool_calls = list(asst.get("tool_calls", []))
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "calculator"
        # No text content alongside reasoning + tool_use → empty string
        # (not absent) for tokenizer compatibility.
        assert asst.get("content") == ""

    def test_tool_use_only_in_assistant_message(self):
        """tool_use only (no text, no thinking): content must be empty str."""
        request = _make_request(
            [
                {"role": "user", "content": "What is 2+2?"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_001",
                            "name": "calculator",
                            "input": {"expression": "2+2"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_001",
                            "content": "4",
                        }
                    ],
                },
            ]
        )
        result = _convert(request)

        asst_msgs = [m for m in result.messages if m.get("role") == "assistant"]
        assert len(asst_msgs) == 1
        asst = asst_msgs[0]

        assert asst.get("content") == ""
        tool_calls = list(asst.get("tool_calls", []))
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "calculator"

    def test_multi_turn_tool_calling_content_always_set(self):
        """Multi-turn tool calling must always set content on assistant msgs.

        Regression test for https://github.com/vllm-project/vllm/issues/38738.
        """
        request = _make_request(
            [
                {"role": "user", "content": "Search for vLLM"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_001",
                            "name": "search",
                            "input": {"query": "vLLM"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_001",
                            "content": "vLLM is a fast LLM inference engine",
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Based on the search...",
                        },
                        {
                            "type": "tool_use",
                            "id": "call_002",
                            "name": "search",
                            "input": {"query": "vLLM performance"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_002",
                            "content": "vLLM achieves high throughput",
                        }
                    ],
                },
            ]
        )
        result = _convert(request)

        asst_msgs = [m for m in result.messages if m.get("role") == "assistant"]
        assert len(asst_msgs) == 2
        # First assistant: tool_use only → content must be ""
        assert asst_msgs[0].get("content") == ""
        assert len(list(asst_msgs[0].get("tool_calls", []))) == 1
        # Second assistant: text + tool_use → content is the text
        assert asst_msgs[1].get("content") == "Based on the search..."
        assert len(list(asst_msgs[1].get("tool_calls", []))) == 1

    def test_multiple_thinking_blocks_concatenated(self):
        """Multiple thinking blocks should be joined in order."""
        request = _make_request(
            [
                {"role": "user", "content": "Think hard."},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "First thought. ",
                            "signature": "s1",
                        },
                        {
                            "type": "thinking",
                            "thinking": "Second thought.",
                            "signature": "s2",
                        },
                        {"type": "text", "text": "Done."},
                    ],
                },
            ]
        )
        result = _convert(request)

        asst_msgs = [m for m in result.messages if m.get("role") == "assistant"]
        assert len(asst_msgs) == 1
        asst = asst_msgs[0]

        assert asst.get("reasoning") == "First thought. Second thought."
        assert asst.get("content") == "Done."

    def test_no_thinking_blocks_unchanged(self):
        """Messages without thinking blocks must not be modified."""
        request = _make_request(
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ]
        )
        result = _convert(request)

        asst_msgs = [m for m in result.messages if m.get("role") == "assistant"]
        assert len(asst_msgs) == 1
        asst = asst_msgs[0]

        assert asst.get("content") == "Hello!"
        assert "reasoning" not in asst

    def test_multi_turn_with_thinking_blocks(self):
        """Full multi-turn conversation: previous assistant messages that
        include thinking blocks must all be converted without a 400 error.

        This is the primary regression scenario from the bug report:
        upgrading vllm from v0.15.1 → v0.17.0 introduced thinking-block
        support in responses, but echoing those responses back in subsequent
        requests caused a Pydantic validation failure.
        """
        request = _make_request(
            [
                {"role": "user", "content": "Turn 1 question"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "Reasoning for turn 1.",
                            "signature": "s_t1",
                        },
                        {"type": "text", "text": "Answer for turn 1."},
                    ],
                },
                {"role": "user", "content": "Turn 2 question"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "Reasoning for turn 2.",
                            "signature": "s_t2",
                        },
                        {"type": "text", "text": "Answer for turn 2."},
                    ],
                },
                {"role": "user", "content": "Turn 3 question"},
            ]
        )
        # Must not raise a ValidationError / 400.
        result = _convert(request)

        asst_msgs = [m for m in result.messages if m.get("role") == "assistant"]
        assert len(asst_msgs) == 2

        assert asst_msgs[0].get("reasoning") == "Reasoning for turn 1."
        assert asst_msgs[0].get("content") == "Answer for turn 1."
        assert asst_msgs[1].get("reasoning") == "Reasoning for turn 2."
        assert asst_msgs[1].get("content") == "Answer for turn 2."

    def test_redacted_thinking_block_is_accepted(self):
        """Anthropic clients may echo back redacted thinking blocks.

        vLLM should accept these blocks (to avoid 400 validation errors)
        and ignore them when constructing the OpenAI-format prompt.
        """
        request = _make_request(
            [
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "Thinking...",
                            "signature": "sig_think",
                        },
                        {
                            "type": "redacted_thinking",
                            "data": "BASE64_OR_OTHER_OPAQUE_DATA",
                        },
                        {"type": "text", "text": "Hi!"},
                    ],
                },
                {"role": "user", "content": "Continue"},
            ]
        )
        result = _convert(request)

        asst_msgs = [m for m in result.messages if m.get("role") == "assistant"]
        assert len(asst_msgs) == 1
        asst = asst_msgs[0]

        # Redacted thinking is ignored, normal thinking still becomes reasoning.
        assert asst.get("reasoning") == "Thinking..."
        assert asst.get("content") == "Hi!"
