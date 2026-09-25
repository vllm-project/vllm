# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Anthropic-to-OpenAI request conversion.

Tests the image source handling and tool_result content parsing in
AnthropicServingMessages.to_chat_completion_request().

Also covers extended-thinking edge cases such as ``redacted_thinking``
blocks echoed back by Anthropic clients, and streaming conversion in
``message_stream_converter``.

Also covers cache usage computation in ``_build_anthropic_usage``.
"""

import asyncio
import json
from argparse import Namespace
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import regex as re
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field, ValidationError

from vllm.entrypoints.anthropic.api_router import attach_router
from vllm.entrypoints.anthropic.inline_system import (
    InlineSystemModeResolver,
    _Prober,
)
from vllm.entrypoints.anthropic.protocol import (
    AnthropicMessagesRequest,
)
from vllm.entrypoints.anthropic.serving import (
    AnthropicServingMessages,
    _build_anthropic_usage,
)
from vllm.entrypoints.chat_utils import parse_chat_messages
from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
)
from vllm.entrypoints.serve.engine.protocol import PromptTokenUsageInfo, UsageInfo
from vllm.entrypoints.serve.exception_handling.handlers.validation import (
    validation_exception_handler,
)
from vllm.exceptions import VLLMValidationError
from vllm.tokenizers import (
    deepseek_v4_encoding,
    deepseek_v32_encoding,
    deepseek_v41_encoding,
)
from vllm.tokenizers.deepseek_v4 import get_deepseek_v4_tokenizer
from vllm.tokenizers.deepseek_v32 import get_deepseek_v32_tokenizer
from vllm.tokenizers.deepseek_v41 import get_deepseek_v41_tokenizer

from ...utils import VLLM_PATH

_convert = AnthropicServingMessages.to_chat_completion_request
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
# vllm_xargs pass-through
# ======================================================================


class TestVllmXargs:
    def test_vllm_xargs_passed_through(self):
        request = _make_request(
            [{"role": "user", "content": "Hello"}],
            vllm_xargs={
                "kv_cache_report_mode": "full",
                "existing_extension": 7,
            },
        )

        result = _convert(request)

        assert result.vllm_xargs == {
            "kv_cache_report_mode": "full",
            "existing_extension": 7,
        }

    def test_vllm_xargs_reaches_sampling_params_with_kv_transfer(self):
        kv_transfer_params = {
            "do_remote_decode": True,
            "do_remote_prefill": False,
        }
        request = _make_request(
            [{"role": "user", "content": "Hello"}],
            vllm_xargs={
                "kv_cache_report_mode": "full",
                "existing_extension": "kept",
            },
            kv_transfer_params=kv_transfer_params,
        )

        converted = _convert(request)
        sampling_params = converted.to_sampling_params(
            max_tokens=converted.max_completion_tokens or 0,
            default_sampling_params={},
        )

        assert converted.kv_transfer_params == kv_transfer_params
        assert sampling_params.extra_args == {
            "kv_cache_report_mode": "full",
            "existing_extension": "kept",
            "kv_transfer_params": kv_transfer_params,
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
        """Thinking + text → reasoning field + plain-string content."""
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
        # No visible text → content should be absent or None.
        assert asst.get("content") is None

    def test_thinking_plus_tool_use_in_assistant_message(self):
        """Thinking + tool_use: reasoning field set, tool_calls populated."""
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
        # No text content alongside reasoning + tool_use.
        assert asst.get("content") is None

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


# ======================================================================
# Cache usage computation
# ======================================================================


class TestBuildAnthropicUsage:
    """Tests for _build_anthropic_usage helper.

    Anthropic defines: total_input = input_tokens + cache_read + cache_creation
    vLLM's prompt_tokens is the total.
    """

    def test_cache_hit(self):
        """When cache is hit, input_tokens excludes cached tokens."""
        usage = UsageInfo(
            prompt_tokens=100,
            completion_tokens=10,
            prompt_tokens_details=PromptTokenUsageInfo(
                cached_tokens=80, created_cache_tokens=10
            ),
        )
        result = _build_anthropic_usage(usage)
        assert result.input_tokens == 10  # 100 - 80 - 10
        assert result.output_tokens == 10
        assert result.cache_read_input_tokens == 80
        assert result.cache_creation_input_tokens == 10

    def test_zero_cached_tokens(self):
        """Zero cached tokens should still set cache_creation to 0."""
        usage = UsageInfo(
            prompt_tokens=100,
            completion_tokens=10,
            prompt_tokens_details=PromptTokenUsageInfo(
                cached_tokens=0, created_cache_tokens=0
            ),
        )
        result = _build_anthropic_usage(usage)
        assert result.input_tokens == 100  # 100 - 0 - 0
        assert result.cache_read_input_tokens == 0
        assert result.cache_creation_input_tokens == 0

    def test_all_tokens_cached(self):
        """When all tokens are cached, input_tokens should be 0."""
        usage = UsageInfo(
            prompt_tokens=100,
            completion_tokens=10,
            prompt_tokens_details=PromptTokenUsageInfo(
                cached_tokens=100, created_cache_tokens=0
            ),
        )
        result = _build_anthropic_usage(usage)
        assert result.input_tokens == 0
        assert result.cache_read_input_tokens == 100
        assert result.cache_creation_input_tokens == 0

    def test_no_prompt_tokens_details(self):
        """UsageInfo without prompt_tokens_details returns no cache info."""
        usage = UsageInfo(prompt_tokens=100, completion_tokens=10)
        result = _build_anthropic_usage(usage)
        assert result.input_tokens == 100
        assert result.cache_read_input_tokens is None
        assert result.cache_creation_input_tokens is None


class TestInlineSystemMessageInMessagesArray:
    """Verify that, in ``preserve`` mode (the converter default), ``role:
    system`` messages inside the ``messages`` array keep their position when
    the request has a leading system prompt, so earlier turns render unchanged
    and stay prefix-cacheable.
    """

    def test_inline_system_merged_with_top_level_system(self):
        """Full integration: inline system + top-level system + user message."""
        request = _make_request(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "<system-reminder>\n.....\n</system-reminder>\n\n",
                        },
                        {
                            "type": "text",
                            "text": "help?",
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                },
                {
                    "role": "system",
                    "content": ".....",
                },
            ],
            system=[
                {
                    "type": "text",
                    "text": "x-anthropic-billing-header: "
                    "cc_version=2.1.160.bca; cc_entrypoint=cli; cch=d1d48;",
                },
                {
                    "type": "text",
                    "text": "You are Claude Code, Anthropic's official CLI for Claude.",
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "text",
                    "text": "....",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            tools=[],
        )

        result = _convert(request)

        # First message: top-level system prompt (billing header stripped).
        assert result.messages[0]["role"] == "system"
        assert (
            result.messages[0]["content"]
            == "You are Claude Code, Anthropic's official CLI for Claude."
            "...."
        )

        # Second message: user message, content preserved at original position.
        assert result.messages[1]["role"] == "user"
        user_content = result.messages[1]["content"]
        assert len(user_content) == 2
        assert user_content[0] == {
            "type": "text",
            "text": "<system-reminder>\n.....\n</system-reminder>\n\n",
        }
        assert user_content[1] == {
            "type": "text",
            "text": "help?",
        }

        # Third message: inline system stays in original position
        # (after user, not merged into leading system).
        assert result.messages[2]["role"] == "system"
        assert result.messages[2]["content"] == "....."

    def test_inline_system_string_only(self):
        """Without a leading system prompt, preserve falls back to folding."""
        request = _make_request(
            [
                {"role": "user", "content": "Hello"},
                {"role": "system", "content": "Be concise."},
            ]
        )
        result = _convert(request)

        assert result.messages == [{"role": "user", "content": "Hello\n\nBe concise."}]

    def test_inline_system_list_content(self):
        """Inline system with list content blocks."""
        request = _make_request(
            [
                {"role": "user", "content": "Hi"},
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "Part one. "},
                        {"type": "text", "text": "Part two."},
                    ],
                },
            ],
            system="Top-level prompt.",
        )
        result = _convert(request)

        # Inline system stays in its original position;
        # text blocks are concatenated (same as top-level system).
        assert result.messages[1] == {"role": "user", "content": "Hi"}
        assert result.messages[2] == {
            "role": "system",
            "content": "Part one. Part two.",
        }

    def test_multiple_inline_system_messages(self):
        """Multiple inline system messages each stay in their position."""
        request = _make_request(
            [
                {"role": "system", "content": "First system."},
                {"role": "user", "content": "Hello"},
                {"role": "system", "content": "Second system."},
            ]
        )
        result = _convert(request)

        # Each system message stays in its original position.
        assert result.messages[0]["role"] == "system"
        assert result.messages[0]["content"] == "First system."
        assert result.messages[1]["role"] == "user"
        assert result.messages[1]["content"] == "Hello"
        assert result.messages[2]["role"] == "system"
        assert result.messages[2]["content"] == "Second system."

    def test_inline_system_with_top_level_string(self):
        """Top-level system is a string, inline system is also present."""
        request = _make_request(
            [
                {"role": "user", "content": "Hello"},
                {"role": "system", "content": "Inline hint."},
            ],
            system="Top-level prompt.",
        )
        result = _convert(request)

        # Top-level system goes first; inline system stays in position.
        assert result.messages[0]["role"] == "system"
        assert result.messages[0]["content"] == "Top-level prompt."
        assert result.messages[1]["role"] == "user"
        assert result.messages[1]["content"] == "Hello"
        assert result.messages[2]["role"] == "system"
        assert result.messages[2]["content"] == "Inline hint."

    def test_inline_system_billing_header_stripped(self):
        """Inline system that is only a billing header is omitted."""
        request = _make_request(
            [
                {"role": "user", "content": "Hello"},
                {
                    "role": "system",
                    "content": "x-anthropic-billing-header: cc_version=2.1.160",
                },
                {"role": "assistant", "content": "Hi there"},
            ]
        )
        result = _convert(request)

        # Billing-header-only system message should be dropped entirely.
        assert len(result.messages) == 2
        assert result.messages[0]["role"] == "user"
        assert result.messages[1]["role"] == "assistant"

    def test_inline_system_billing_header_mixed_with_content(self):
        """Inline system with billing header block + real content."""
        request = _make_request(
            [
                {"role": "user", "content": "Hello"},
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "x-anthropic-billing-header: "
                            "cc_version=2.1.160.bca; cch=d1d48;",
                        },
                        {"type": "text", "text": "Real system content."},
                    ],
                },
            ],
            system="Top-level prompt.",
        )
        result = _convert(request)

        # Billing header stripped, real content preserved in position.
        assert result.messages[1:] == [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "Real system content."},
        ]


# ======================================================================
# Streaming conversion: message_stream_converter
# ======================================================================


def _make_stream_converter():
    obj = MagicMock(spec=AnthropicServingMessages)
    obj.stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
    }
    obj.message_stream_converter = (
        AnthropicServingMessages.message_stream_converter.__get__(obj)
    )
    return obj


def _parse_sse_events(raw_events: list[str]) -> list[tuple[str, dict]]:
    results = []
    for raw in raw_events:
        headers = dict(
            line.split(": ", 1) for line in raw.strip().split("\n") if ": " in line
        )
        if "event" in headers and "data" in headers:
            results.append((headers["event"], json.loads(headers["data"])))
    return results


def _make_stream_chunk(
    *,
    delta: DeltaMessage | None = None,
    finish_reason: str | None = None,
    stop_reason: int | str | None = None,
    choices: list[ChatCompletionResponseStreamChoice] | None = None,
    usage: UsageInfo | None = None,
) -> str:
    if choices is None:
        choices = [
            ChatCompletionResponseStreamChoice(
                index=0,
                delta=delta or DeltaMessage(),
                finish_reason=finish_reason,
                stop_reason=stop_reason,
            )
        ]
    chunk = ChatCompletionStreamResponse(
        id="chatcmpl-test",
        created=0,
        model="test-model",
        choices=choices,
        usage=usage,
    )
    return f"data: {chunk.model_dump_json()}"


def _tc(*, args, id=None, name=None):
    return DeltaToolCall(
        index=0,
        id=id,
        function=DeltaFunctionCall(name=name, arguments=args),
    )


class TestMessageStreamConverterToolUseContentBuffering:
    """Regression test for tool_use arguments being silently dropped.

    With speculative decoding or multi-token prediction, a single delta
    can carry both the final tool_call argument fragment and trailing
    content.
    """

    @pytest.mark.asyncio
    async def test_tool_use_args_not_dropped_when_content_in_same_chunk(
        self,
    ):
        async def sse_input():
            yield _make_stream_chunk(
                delta=DeltaMessage(role="assistant"),
                usage=UsageInfo(prompt_tokens=10, total_tokens=10),
            )
            yield _make_stream_chunk(
                delta=DeltaMessage(
                    tool_calls=[
                        _tc(id="call_abc123", name="read_file", args=""),
                    ]
                )
            )
            yield _make_stream_chunk(
                delta=DeltaMessage(
                    tool_calls=[
                        _tc(args='{"path":"/tmp/f"'),
                    ]
                )
            )
            # BUG TRIGGER: final tool_call args and trailing content in
            # one delta, as happens with spec decoding / multi-token
            # prediction where multiple tokens land in a single chunk.
            yield _make_stream_chunk(
                delta=DeltaMessage(
                    content="\nOkay",
                    tool_calls=[_tc(args="}")],
                )
            )
            yield _make_stream_chunk(finish_reason="tool_calls")
            yield _make_stream_chunk(
                choices=[],
                usage=UsageInfo(
                    prompt_tokens=10,
                    total_tokens=30,
                    completion_tokens=20,
                ),
            )
            yield "data: [DONE]"

        converter = _make_stream_converter()
        output = []
        async for event in converter.message_stream_converter(sse_input()):
            output.append(event)

        events = _parse_sse_events(output)

        assert events[0][0] == "message_start"

        arg_fragments = [
            data["delta"]["partial_json"]
            for _, data in events
            if data.get("delta", {}).get("type") == "input_json_delta"
        ]
        full_args = "".join(arg_fragments)
        assert full_args == '{"path":"/tmp/f"}'

        text_deltas = [
            data["delta"]["text"]
            for _, data in events
            if data.get("delta", {}).get("type") == "text_delta"
        ]
        assert text_deltas == ["\nOkay"]

        block_starts = [
            (data["content_block"]["type"], data.get("index"))
            for ev_type, data in events
            if ev_type == "content_block_start"
        ]
        assert block_starts[0] == ("tool_use", 0)
        assert block_starts[1] == ("text", 1)

        msg_deltas = [data for ev_type, data in events if ev_type == "message_delta"]
        assert msg_deltas[0]["delta"]["stop_reason"] == "tool_use"

        assert events[-1][0] == "message_stop"

    @pytest.mark.asyncio
    async def test_buffered_content_flushed_on_done_without_usage_chunk(self):
        """Content buffered during tool_use must be emitted even if the
        stream jumps straight from finish_reason to [DONE], skipping the
        empty-choices usage chunk."""

        async def sse_input():
            yield _make_stream_chunk(
                delta=DeltaMessage(role="assistant"),
                usage=UsageInfo(prompt_tokens=10, total_tokens=10),
            )
            yield _make_stream_chunk(
                delta=DeltaMessage(
                    tool_calls=[
                        _tc(id="call_xyz", name="get_weather", args=""),
                    ]
                )
            )
            yield _make_stream_chunk(
                delta=DeltaMessage(
                    tool_calls=[_tc(args='{"city":"NYC"}')],
                )
            )
            yield _make_stream_chunk(
                delta=DeltaMessage(content="\nDone"),
                finish_reason="tool_calls",
            )
            # No empty-choices usage chunk — go straight to [DONE].
            yield "data: [DONE]"

        converter = _make_stream_converter()
        output = []
        async for event in converter.message_stream_converter(sse_input()):
            output.append(event)

        events = _parse_sse_events(output)

        text_deltas = [
            data["delta"]["text"]
            for _, data in events
            if data.get("delta", {}).get("type") == "text_delta"
        ]
        assert text_deltas == ["\nDone"]

        block_starts = [
            data["content_block"]["type"]
            for ev_type, data in events
            if ev_type == "content_block_start"
        ]
        assert "tool_use" in block_starts
        assert "text" in block_starts

        assert events[-1][0] == "message_stop"


class TestMessageStartIncludesTypeAndRole:
    """Regression test for issue #45367: the streaming message_start event is
    serialized with exclude_unset=True, which silently dropped the
    default-valued ``type``/``role`` fields of the nested message object.
    Strict Anthropic SDK clients (e.g. Claude Code) validate
    ``message_start.message.type``/``role`` and reject the whole stream when
    they are missing.
    """

    @pytest.mark.asyncio
    async def test_message_start_contains_message_type_and_role(self):
        async def sse_input():
            yield _make_stream_chunk(
                delta=DeltaMessage(content="Hello"),
                usage=UsageInfo(
                    prompt_tokens=20,
                    total_tokens=20,
                    completion_tokens=0,
                ),
            )
            yield _make_stream_chunk(finish_reason="stop")
            yield "data: [DONE]"

        converter = _make_stream_converter()
        output = []
        async for event in converter.message_stream_converter(sse_input()):
            output.append(event)

        events = _parse_sse_events(output)

        assert events[0][0] == "message_start"
        message = events[0][1]["message"]
        assert message["type"] == "message"
        assert message["role"] == "assistant"


class TestStreamingCacheUsageSemantics:
    """Locks in the documented streaming behavior of cache usage fields.

    vLLM's OpenAI chat completion streaming only attaches
    ``prompt_tokens_details`` to the terminal usage chunk. The Anthropic layer
    mirrors that contract: cache fields are omitted on ``message_start`` (key
    absence signals "unknown") and populated on ``message_delta`` (the final
    cumulative count). This is intentionally consistent with vLLM's OpenAI
    behavior, even though Anthropic's upstream API populates cache fields on
    ``message_start``; closing that gap requires plumbing cache info into the
    first chunk at the OpenAI layer, which is out of scope here.
    """

    @pytest.mark.asyncio
    async def test_streaming_cache_fields_absent_then_populated(self):
        """First chunk lacks prompt_tokens_details (vLLM contract);
        message_start omits cache fields. The final chunk carries
        prompt_tokens_details, so message_delta carries resolved values."""

        async def sse_input():
            yield _make_stream_chunk(
                delta=DeltaMessage(role="assistant", content="hi"),
                usage=UsageInfo(prompt_tokens=100, total_tokens=100),
            )
            yield _make_stream_chunk(finish_reason="stop")
            yield _make_stream_chunk(
                choices=[],
                usage=UsageInfo(
                    prompt_tokens=100,
                    completion_tokens=5,
                    total_tokens=105,
                    prompt_tokens_details=PromptTokenUsageInfo(
                        cached_tokens=80, created_cache_tokens=10
                    ),
                ),
            )
            yield "data: [DONE]"

        converter = _make_stream_converter()
        output = []
        async for event in converter.message_stream_converter(sse_input()):
            output.append(event)
        events = _parse_sse_events(output)

        # message_start: cache fields unknown → omitted from JSON entirely.
        start_usage = events[0][1]["message"]["usage"]
        assert events[0][0] == "message_start"
        assert start_usage["input_tokens"] == 100
        assert "cache_read_input_tokens" not in start_usage
        assert "cache_creation_input_tokens" not in start_usage

        # message_delta: authoritative usage with cache fields populated.
        delta_usage = next(
            data["usage"] for ev, data in events if ev == "message_delta"
        )
        assert delta_usage["input_tokens"] == 10  # 100 - 80 - 10
        assert delta_usage["cache_read_input_tokens"] == 80
        assert delta_usage["cache_creation_input_tokens"] == 10

    @pytest.mark.asyncio
    async def test_streaming_no_cache_hit(self):
        """When the final chunk reports cached_tokens=0, message_delta carries
        cache fields = 0 (cache miss); message_start still omits them."""

        async def sse_input():
            yield _make_stream_chunk(
                delta=DeltaMessage(role="assistant"),
                usage=UsageInfo(prompt_tokens=50, total_tokens=50),
            )
            yield _make_stream_chunk(finish_reason="stop")
            yield _make_stream_chunk(
                choices=[],
                usage=UsageInfo(
                    prompt_tokens=50,
                    completion_tokens=5,
                    total_tokens=55,
                    prompt_tokens_details=PromptTokenUsageInfo(
                        cached_tokens=0, created_cache_tokens=0
                    ),
                ),
            )
            yield "data: [DONE]"

        converter = _make_stream_converter()
        output = []
        async for event in converter.message_stream_converter(sse_input()):
            output.append(event)
        events = _parse_sse_events(output)

        start_usage = events[0][1]["message"]["usage"]
        delta_usage = next(
            data["usage"] for ev, data in events if ev == "message_delta"
        )
        assert start_usage["input_tokens"] == 50
        assert "cache_read_input_tokens" not in start_usage
        assert "cache_creation_input_tokens" not in start_usage
        assert delta_usage["input_tokens"] == 50  # 50 - 0 - 0
        assert delta_usage["cache_read_input_tokens"] == 0
        assert delta_usage["cache_creation_input_tokens"] == 0

    @pytest.mark.asyncio
    async def test_streaming_no_prompt_tokens_details_at_all(self):
        """If --enable-prompt-tokens-details is off, no chunk carries cache
        info; both message_start and message_delta omit cache fields."""

        async def sse_input():
            yield _make_stream_chunk(
                delta=DeltaMessage(role="assistant"),
                usage=UsageInfo(prompt_tokens=30, total_tokens=30),
            )
            yield _make_stream_chunk(finish_reason="stop")
            yield _make_stream_chunk(
                choices=[],
                usage=UsageInfo(prompt_tokens=30, completion_tokens=2, total_tokens=32),
            )
            yield "data: [DONE]"

        converter = _make_stream_converter()
        output = []
        async for event in converter.message_stream_converter(sse_input()):
            output.append(event)
        events = _parse_sse_events(output)

        start_usage = events[0][1]["message"]["usage"]
        delta_usage = next(
            data["usage"] for ev, data in events if ev == "message_delta"
        )
        assert "cache_read_input_tokens" not in start_usage
        assert "cache_creation_input_tokens" not in start_usage
        assert "cache_read_input_tokens" not in delta_usage
        assert "cache_creation_input_tokens" not in delta_usage


# ======================================================================
# Inline system message placement
# ======================================================================


def _tool_use(tool_id: str, command: str = "ls") -> dict:
    return {
        "type": "tool_use",
        "id": tool_id,
        "name": "Bash",
        "input": {"command": command},
    }


def _tool_result(tool_id: str, content="out") -> dict:
    return {"type": "tool_result", "tool_use_id": tool_id, "content": content}


def _roles(messages: list[dict]) -> list[str]:
    return [m["role"] for m in messages]


class TestNormalizeInlineSystem:
    """Placement of inline system messages in ``preserve`` and ``fold``."""

    @pytest.mark.parametrize("mode", ["preserve", "fold"])
    def test_leading_inline_system_joins_system_prompt(self, mode):
        request = _make_request(
            [
                {"role": "system", "content": "Lead."},
                {"role": "user", "content": "Q"},
            ],
            system="Top.",
        )
        result = _convert(request, inline_system=mode)

        assert result.messages == [
            {"role": "system", "content": "Top.\n\nLead."},
            {"role": "user", "content": "Q"},
        ]

    @pytest.mark.parametrize(
        ("mode", "expected_tail"),
        [
            (
                "preserve",
                [
                    {"role": "tool", "tool_call_id": "t1", "content": "out"},
                    {"role": "system", "content": "S"},
                ],
            ),
            ("fold", [{"role": "tool", "tool_call_id": "t1", "content": "out\n\nS"}]),
        ],
    )
    def test_system_between_tool_call_and_result_moves_after_results(
        self, mode, expected_tail
    ):
        request = _make_request(
            [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": [_tool_use("t1")]},
                {"role": "system", "content": "S"},
                {"role": "user", "content": [_tool_result("t1")]},
            ],
            system="Top.",
        )
        result = _convert(request, inline_system=mode)

        assert _roles(result.messages[:3]) == ["system", "user", "assistant"]
        assert result.messages[3:] == expected_tail

    def test_fold_appends_to_last_string_tool_result(self):
        """Images and tool_reference parts in the run are skipped over."""
        image = {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "x"},
        }
        request = _make_request(
            [
                {"role": "user", "content": "Q"},
                {
                    "role": "assistant",
                    "content": [_tool_use("t1"), _tool_use("t2")],
                },
                {
                    "role": "user",
                    "content": [
                        _tool_result("t1", [{"type": "text", "text": "a"}, image]),
                        _tool_result(
                            "t2", [{"type": "tool_reference", "tool_name": "Read"}]
                        ),
                    ],
                },
                {"role": "system", "content": "S1"},
                {"role": "system", "content": "S2"},
            ],
            system="Top.",
        )
        result = _convert(request, inline_system="fold")

        assert _roles(result.messages) == [
            "system",
            "user",
            "assistant",
            "tool",
            "user",
            "tool",
            "tool",
        ]
        # The empty t2 result gains the text without a leading separator.
        assert result.messages[5]["content"] == "S1\n\nS2"
        assert result.messages[3]["content"] == "a"
        assert result.messages[6]["content"] == [
            {"type": "tool_reference", "name": "Read"}
        ]

    def test_fold_appends_to_user_text_after_tool_results(self):
        """Claude Code's own user text after tool results is reused."""
        request = _make_request(
            [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": [_tool_use("t1")]},
                {
                    "role": "user",
                    "content": [
                        _tool_result("t1"),
                        {
                            "type": "text",
                            "text": "<system-reminder>r</system-reminder>",
                        },
                    ],
                },
                {"role": "system", "content": "S"},
            ],
            system="Top.",
        )
        result = _convert(request, inline_system="fold")

        assert _roles(result.messages) == [
            "system",
            "user",
            "assistant",
            "tool",
            "user",
        ]
        assert (
            result.messages[4]["content"] == "<system-reminder>r</system-reminder>\n\nS"
        )

    def test_fold_extends_list_user_content(self):
        request = _make_request(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "A"},
                        {"type": "text", "text": "B"},
                    ],
                },
                {"role": "system", "content": "S"},
            ],
            system="Top.",
        )
        result = _convert(request, inline_system="fold")

        assert result.messages[1]["content"] == [
            {"type": "text", "text": "A"},
            {"type": "text", "text": "B\n\nS"},
        ]

    def test_fold_after_assistant_prepends_to_next_user(self):
        request = _make_request(
            [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"},
                {"role": "system", "content": "S"},
                {"role": "user", "content": "Q2"},
            ],
            system="Top.",
        )
        result = _convert(request, inline_system="fold")

        assert result.messages[1:] == [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
            {"role": "user", "content": "S\n\nQ2"},
        ]

    @pytest.mark.parametrize(
        "after", [[], [{"role": "assistant", "content": "A2"}]], ids=["end", "asst"]
    )
    def test_fold_after_assistant_without_next_user(self, after):
        request = _make_request(
            [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"},
                {"role": "system", "content": "S"},
                *after,
            ],
            system="Top.",
        )
        result = _convert(request, inline_system="fold")

        assert result.messages[3] == {"role": "user", "content": "S"}
        assert len(result.messages) == 4 + len(after)

    def test_fold_into_empty_user_content(self):
        request = _make_request(
            [
                {"role": "user", "content": []},
                {"role": "system", "content": "S"},
            ],
            system="Top.",
        )
        result = _convert(request, inline_system="fold")

        assert result.messages[1:] == [{"role": "user", "content": "S"}]

    @pytest.mark.parametrize(
        "system",
        [
            None,
            [{"type": "text", "text": "x-anthropic-billing-header: cc_version=1;"}],
        ],
        ids=["none", "billing_header_only"],
    )
    def test_preserve_without_system_prompt_folds(self, system):
        request = _make_request(
            [
                {"role": "user", "content": "Q"},
                {"role": "system", "content": "S"},
            ],
            system=system,
        )
        result = _convert(request, inline_system="preserve")

        assert result.messages == [{"role": "user", "content": "Q\n\nS"}]


def _claude_code_session(num_turns: int) -> list[dict]:
    """Messages of a Claude Code-shaped session after ``num_turns`` tool loops.

    Each turn has thinking, parallel tool calls, an image tool result, user
    reminder text on alternate turns, and an inline system message.
    """
    image = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "x"},
    }
    messages: list[dict] = [
        {"role": "user", "content": "Fix the bug."},
        {"role": "system", "content": "# Environment\n - Platform: linux"},
    ]
    for turn in range(num_turns):
        ids = (f"t{turn}a", f"t{turn}b")
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": f"think {turn}", "signature": ""},
                    *(_tool_use(i, f"cmd {i}") for i in ids),
                ],
            }
        )
        results = [
            _tool_result(ids[0], f"out {turn}"),
            _tool_result(ids[1], [{"type": "text", "text": "shot"}, image]),
        ]
        if turn % 2:
            results.append({"type": "text", "text": f"reminder {turn}"})
        messages.append({"role": "user", "content": results})
        messages.append(
            {"role": "system", "content": f"<total_tokens>{turn}</total_tokens>"}
        )
    return messages


class TestInlineSystemSessionInvariants:
    """Invariants that keep Claude Code sessions cacheable and well-formed."""

    @staticmethod
    def _session(num_turns: int, mode: str, system: str | None) -> list[dict]:
        request = _make_request(_claude_code_session(num_turns), system=system)
        return _convert(request, inline_system=mode).messages

    @pytest.mark.parametrize("mode", ["preserve", "fold"])
    @pytest.mark.parametrize("system", ["Top.", None])
    def test_each_turn_extends_the_previous(self, mode, system):
        for turn in range(1, 5):
            prev = self._session(turn, mode, system)
            cur = self._session(turn + 1, mode, system)
            assert cur[: len(prev)] == prev

    @pytest.mark.parametrize("system", ["Top.", None])
    def test_fold_adds_no_user_turns(self, system):
        """Reasoning kept only after the last user turn stays in place."""
        without_inline = _make_request(
            [m for m in _claude_code_session(4) if m["role"] != "system"],
            system=system,
        )
        folded = self._session(4, "fold", system)

        assert _roles(folded) == _roles(_convert(without_inline).messages)

    @pytest.mark.parametrize("mode", ["preserve", "fold"])
    def test_tool_results_follow_their_tool_calls(self, mode):
        """Only tool results (and their images) follow a tool call."""
        messages = self._session(4, mode, "Top.")
        for i, message in enumerate(messages):
            if not message.get("tool_calls"):
                continue
            run = []
            for follower in messages[i + 1 :]:
                content = follower.get("content")
                is_image = isinstance(content, list) and all(
                    part["type"] == "image_url" for part in content
                )
                if follower["role"] != "tool" and not is_image:
                    break
                run.append(follower)
            assert [m["tool_call_id"] for m in run if m["role"] == "tool"] == [
                tc["id"] for tc in message["tool_calls"]
            ]


# ======================================================================
# Renderer probe for inline system support
# ======================================================================

_SPECIAL_RE = re.compile(r"<\|[^|<>]*\|>|<｜[^｜]*｜>")
_ROLE_MARKERS = "<|system|><|user|><|assistant|><|tool|>"
_TEMPLATES = VLLM_PATH / "examples"
_LLAMA_TEMPLATE = _TEMPLATES / "tool_chat_template_llama3.2_json.jinja"


class _FakeTokenizer:
    """Word-level tokenizer whose ``<|...|>``/``<｜...｜>`` markers are special."""

    def __init__(self, special_source: str) -> None:
        self.vocab: dict[str, int] = {}
        self.all_special_ids = [
            self._id(t) for t in dict.fromkeys(_SPECIAL_RE.findall(special_source))
        ]

    def _id(self, piece: str) -> int:
        return self.vocab.setdefault(piece, len(self.vocab))

    def encode(self, text: str) -> list[int]:
        pieces: list[str] = []
        pos = 0
        for match in _SPECIAL_RE.finditer(text):
            pieces += re.findall(r"\w+|\W", text[pos : match.start()])
            pieces.append(match.group())
            pos = match.end()
        pieces += re.findall(r"\w+|\W", text[pos:])
        return [self._id(p) for p in pieces]

    def decode(self, token_ids: list[int]) -> str:
        pieces = {i: p for p, i in self.vocab.items()}
        return "".join(pieces[i] for i in token_ids)

    def get_added_vocab(self) -> dict[str, int]:
        return {}


class _FakeOnlineRenderer:
    """Renders like ``OnlineRenderer.render_chat`` with ``apply_chat_template``."""

    model_config = SimpleNamespace(
        is_encoder_decoder=False,
        multimodal_config=None,
        allowed_local_media_path="",
        allowed_media_domains=None,
        enable_prompt_embeds=False,
    )

    def __init__(self, apply_chat_template, special_source: str) -> None:
        self.apply_chat_template = apply_chat_template
        tokenizer = _FakeTokenizer(special_source)
        self.renderer = SimpleNamespace(get_tokenizer=lambda: tokenizer)

    async def render_chat(self, request):
        conversation, _, _ = parse_chat_messages(
            request.messages, self.model_config, content_format="string"
        )
        params = request.build_chat_params(None, "auto")
        text = self.apply_chat_template(
            conversation,
            tools=[t.model_dump() for t in request.tools],
            tokenize=False,
            **params.chat_template_kwargs,
        )
        tokenizer = self.renderer.get_tokenizer()
        return conversation, [{"prompt_token_ids": tokenizer.encode(text)}]


def _jinja_online_renderer(template: str) -> _FakeOnlineRenderer:
    from transformers.utils.chat_template_utils import _compile_jinja_template

    compiled = _compile_jinja_template(template)

    def apply_chat_template(conversation, **kwargs):
        return compiled.render(messages=conversation, bos_token="<|bos|>", **kwargs)

    return _FakeOnlineRenderer(apply_chat_template, template + _ROLE_MARKERS)


class _FakeHfTokenizer:
    def get_added_vocab(self) -> dict[str, int]:
        return {}


def _probe(online_renderer) -> tuple[str, str]:
    return asyncio.run(_Prober(online_renderer).probe())


class TestProbeInlineSystem:
    """``auto`` keeps inline system turns only where the renderer marks them."""

    def test_role_marked_template_preserves(self):
        mode, reason = _probe(_jinja_online_renderer(_LLAMA_TEMPLATE.read_text()))
        assert mode == "preserve", reason

    @pytest.mark.parametrize(
        ("template", "expected"),
        [
            (
                (VLLM_PATH / "rust/src/chat/tests/templates/qwen35.jinja").read_text(),
                "renderer rejects it",
            ),
            (
                "{% for m in messages %}{% if m.role != 'system' or loop.first %}"
                "<|{{ m.role }}|>{{ m.content }}{% endif %}{% endfor %}"
                "{% if add_generation_prompt %}<|assistant|>{% endif %}",
                "drops it",
            ),
            (
                # Glued text also merges tokens across the boundary.
                "{% for m in messages %}"
                "{% if m.role == 'system' %}{{ m.content }}"
                "{% else %}<|{{ m.role }}|>{{ m.content }}{% endif %}{% endfor %}"
                "{% if add_generation_prompt %}<|assistant|>{% endif %}",
                "mid: rendering it changes other parts",
            ),
            (
                "{% for m in messages %}"
                "<|{{ 'user' if m.role == 'system' and not loop.first else m.role }}|>"
                "{{ m.content }}{% endfor %}"
                "{% if add_generation_prompt %}<|assistant|>{% endif %}",
                "renders the same as a user message",
            ),
            (
                "{% set ns = namespace(last=-1) %}{% for m in messages %}"
                "{% if m.role in ('user', 'system') %}{% set ns.last = loop.index0 %}"
                "{% endif %}{% endfor %}{% for m in messages %}<|{{ m.role }}|>"
                "{% if m.reasoning_content and loop.index0 > ns.last %}"
                "{{ m.reasoning_content }}{% endif %}{{ m.content or '' }}"
                "{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}",
                "after_tool: rendering it changes other parts",
            ),
        ],
        ids=["rejects", "drops", "unmarked", "as_user", "reasoning"],
    )
    def test_unsafe_templates_fold(self, template, expected):
        mode, reason = _probe(_jinja_online_renderer(template))
        assert mode == "fold"
        assert expected in reason

    def test_renderer_rejecting_history_reasoning_is_still_probed(self):
        template = (
            "{% for m in messages %}{% if m.reasoning_content %}"
            "{{ raise_exception('no think tokens') }}{% endif %}"
            "<|{{ m.role }}|>{{ m.content or '' }}<|end|>{% endfor %}"
            "{% if add_generation_prompt %}<|assistant|>{% endif %}"
        )
        mode, reason = _probe(_jinja_online_renderer(template))
        assert mode == "preserve", reason

    @pytest.mark.parametrize(
        ("wrap", "encoding", "expected"),
        [
            (get_deepseek_v32_tokenizer, deepseek_v32_encoding, "fold"),
            (get_deepseek_v4_tokenizer, deepseek_v4_encoding, "fold"),
            (get_deepseek_v41_tokenizer, deepseek_v41_encoding, "preserve"),
        ],
        ids=["v32", "v4", "v41"],
    )
    def test_deepseek_encoders(self, wrap, encoding, expected):
        online_renderer = _FakeOnlineRenderer(
            wrap(_FakeHfTokenizer()).apply_chat_template,
            Path(encoding.__file__).read_text(),
        )
        mode, reason = _probe(online_renderer)
        assert mode == expected, reason

    def test_unrenderable_probe_folds(self):
        online_renderer = _jinja_online_renderer("")
        failing = AsyncMock(side_effect=ValueError("boom"))
        with patch.object(online_renderer, "render_chat", failing):
            assert _probe(online_renderer)[0] == "fold"

    def test_resolver_probes_once(self):
        online_renderer = _jinja_online_renderer(_LLAMA_TEMPLATE.read_text())
        render_chat = AsyncMock(wraps=online_renderer.render_chat)
        resolver = InlineSystemModeResolver(online_renderer, "auto")

        async def resolve_many():
            return await asyncio.gather(*(resolver.resolve() for _ in range(4)))

        with patch.object(online_renderer, "render_chat", render_chat):
            assert asyncio.run(resolve_many()) == ["preserve"] * 4
        calls = render_chat.await_count
        assert asyncio.run(InlineSystemModeResolver(None, "fold").resolve()) == "fold"
        assert render_chat.await_count == calls


# ======================================================================
# Full (non-streaming) response conversion: messages_full_converter
# ======================================================================


def _make_full_converter():
    obj = MagicMock(spec=AnthropicServingMessages)
    obj.messages_full_converter = (
        AnthropicServingMessages.messages_full_converter.__get__(obj)
    )
    return obj


class TestMessagesFullConverter:
    def test_empty_completion_emits_one_text_block(self):
        """An empty completion still yields exactly one (empty) text block."""
        generator = ChatCompletionResponse(
            id="chatcmpl-empty",
            model="test-model",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=None),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(prompt_tokens=10, completion_tokens=0, total_tokens=10),
        )

        result = _make_full_converter().messages_full_converter(generator)

        assert len(result.content) == 1
        assert result.content[0].type == "text"
        assert result.content[0].text == ""


# ======================================================================
# cache_salt pass-through (Issue #46688)
# ======================================================================


class TestCacheSalt:
    def test_cache_salt_passed_through(self):
        """cache_salt on the Anthropic request reaches the converted
        ChatCompletionRequest so prefix-cache isolation works via /v1/messages."""
        request = _make_request(
            [{"role": "user", "content": "Hello"}],
            cache_salt="tenant-abc-secret-salt",
        )
        result = _convert(request)
        assert result.cache_salt == "tenant-abc-secret-salt"

    def test_cache_salt_defaults_to_none(self):
        """Omitting cache_salt leaves it unset (unchanged default behavior)."""
        request = _make_request([{"role": "user", "content": "Hello"}])
        result = _convert(request)
        assert result.cache_salt is None

    @staticmethod
    def _make_api_app():
        app = FastAPI()
        attach_router(app)
        app.state.args = Namespace(log_error_stack=False)
        app.exception_handler(RequestValidationError)(validation_exception_handler)

        handler = MagicMock(spec=AnthropicServingMessages)
        handler.create_messages.side_effect = AssertionError(
            "invalid requests must not reach the serving handler"
        )
        app.state.anthropic_serving_messages = handler
        return app, handler

    def test_cache_salt_openapi_requires_non_empty_string(self):
        app, _ = self._make_api_app()
        field_schema = app.openapi()["components"]["schemas"][
            "AnthropicMessagesRequest"
        ]["properties"]["cache_salt"]
        string_schema = next(
            option for option in field_schema["anyOf"] if option.get("type") == "string"
        )

        assert string_schema["minLength"] == 1

    def test_empty_cache_salt_returns_bad_request(self):
        app, handler = self._make_api_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/v1/messages",
                json={
                    "model": "test-model",
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "cache_salt": "",
                },
            )

        assert response.status_code == HTTPStatus.BAD_REQUEST
        handler.create_messages.assert_not_awaited()


class TestStopSequenceReason:
    """When generation stops because a configured stop string matched, the
    Anthropic Messages API must report ``stop_reason="stop_sequence"`` and echo
    the matched string in ``stop_sequence``. vLLM surfaces the matched string in
    the OpenAI choice's ``stop_reason`` field (a str) while ``finish_reason``
    stays ``"stop"``. A natural EOS (stop_reason None) or a stop token id (int)
    must still map to ``end_turn``.
    """

    def test_non_streaming_stop_string_maps_to_stop_sequence(self):
        converter = _make_full_converter()
        response = ChatCompletionResponse(
            id="chatcmpl-test",
            model="test-model",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="hello"),
                    finish_reason="stop",
                    stop_reason="</tool>",
                )
            ],
            usage=UsageInfo(prompt_tokens=5, total_tokens=8, completion_tokens=3),
        )

        result = converter.messages_full_converter(response)

        assert result.stop_reason == "stop_sequence"
        assert result.stop_sequence == "</tool>"

    def test_non_streaming_natural_eos_maps_to_end_turn(self):
        converter = _make_full_converter()
        response = ChatCompletionResponse(
            id="chatcmpl-test",
            model="test-model",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="hello"),
                    finish_reason="stop",
                    stop_reason=None,
                )
            ],
            usage=UsageInfo(prompt_tokens=5, total_tokens=8, completion_tokens=3),
        )

        result = converter.messages_full_converter(response)

        assert result.stop_reason == "end_turn"
        assert result.stop_sequence is None

    def test_non_streaming_stop_token_id_maps_to_end_turn(self):
        converter = _make_full_converter()
        response = ChatCompletionResponse(
            id="chatcmpl-test",
            model="test-model",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="hello"),
                    finish_reason="stop",
                    stop_reason=128009,
                )
            ],
            usage=UsageInfo(prompt_tokens=5, total_tokens=8, completion_tokens=3),
        )

        result = converter.messages_full_converter(response)

        assert result.stop_reason == "end_turn"
        assert result.stop_sequence is None

    @pytest.mark.asyncio
    async def test_streaming_stop_string_maps_to_stop_sequence(self):
        async def sse_input():
            yield _make_stream_chunk(
                delta=DeltaMessage(content="hi"),
                usage=UsageInfo(prompt_tokens=5, total_tokens=5, completion_tokens=0),
            )
            yield _make_stream_chunk(finish_reason="stop", stop_reason="</tool>")
            yield _make_stream_chunk(
                choices=[],
                usage=UsageInfo(prompt_tokens=5, total_tokens=8, completion_tokens=3),
            )
            yield "data: [DONE]"

        converter = _make_stream_converter()
        output = []
        async for event in converter.message_stream_converter(sse_input()):
            output.append(event)

        events = _parse_sse_events(output)
        msg_deltas = [data for ev_type, data in events if ev_type == "message_delta"]
        assert msg_deltas[0]["delta"]["stop_reason"] == "stop_sequence"
        assert msg_deltas[0]["delta"]["stop_sequence"] == "</tool>"

    @pytest.mark.asyncio
    async def test_streaming_no_stop_string_emits_explicit_null_stop_sequence(self):
        """exclude_unset=True drops stop_sequence unless it is set explicitly."""

        async def sse_input():
            yield _make_stream_chunk(delta=DeltaMessage(role="assistant"))
            yield _make_stream_chunk(delta=DeltaMessage(content="hi"))
            yield _make_stream_chunk(finish_reason="stop")
            yield _make_stream_chunk(
                choices=[],
                usage=UsageInfo(prompt_tokens=5, total_tokens=8, completion_tokens=3),
            )
            yield "data: [DONE]"

        converter = _make_stream_converter()
        output = []
        async for event in converter.message_stream_converter(sse_input()):
            output.append(event)

        events = _parse_sse_events(output)
        msg_deltas = [data for ev_type, data in events if ev_type == "message_delta"]
        assert len(msg_deltas) == 1
        assert msg_deltas[0]["delta"]["stop_reason"] == "end_turn"
        assert "stop_sequence" in msg_deltas[0]["delta"]
        assert msg_deltas[0]["delta"]["stop_sequence"] is None


# ======================================================================
# Client-caused errors are 4xx, not 500 (Issue #52088)
# ======================================================================


class TestClientErrorResponses:
    @staticmethod
    def _make_api_app(handler: MagicMock):
        app = FastAPI()
        attach_router(app)
        app.state.args = Namespace(log_error_stack=False)
        app.exception_handler(RequestValidationError)(validation_exception_handler)
        app.state.anthropic_serving_messages = handler
        return app

    @staticmethod
    def _request_body() -> dict:
        return {
            "model": "test-model",
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "Hello"}],
        }

    @staticmethod
    def _conversion_error() -> ValidationError:
        """A real pydantic ValidationError like the one ChatCompletionRequest
        construction raises when Anthropic input violates the OpenAI schema."""

        class _StubRequest(BaseModel):
            stop: Annotated[list[str], Field(max_length=4)] | None = None

        with pytest.raises(ValidationError) as exc_info:
            _StubRequest(stop=["a"] * 6)
        return exc_info.value

    def test_validation_error_returns_bad_request(self):
        """A pydantic ValidationError during Anthropic->OpenAI conversion is
        surfaced as a 400 BadRequestError, not a 500."""
        handler = MagicMock(spec=AnthropicServingMessages)
        handler.create_messages.side_effect = self._conversion_error()

        app = self._make_api_app(handler)
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/v1/messages", json=self._request_body())

        assert response.status_code == HTTPStatus.BAD_REQUEST
        body = response.json()
        assert body["type"] == "error"
        assert body["error"]["type"] == "BadRequestError"
        assert "at most 4 items" in body["error"]["message"]

    def test_vllm_client_error_returns_bad_request(self):
        """VLLMClientError raised by the serving layer maps to 400."""
        handler = MagicMock(spec=AnthropicServingMessages)
        handler.create_messages.side_effect = VLLMValidationError(
            "Invalid value for stop", parameter="stop"
        )

        app = self._make_api_app(handler)
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/v1/messages", json=self._request_body())

        assert response.status_code == HTTPStatus.BAD_REQUEST
        assert response.json()["error"]["type"] == "BadRequestError"

    def test_generic_error_still_returns_internal_server_error(self):
        """Non-client errors keep the existing 500 behaviour."""
        handler = MagicMock(spec=AnthropicServingMessages)
        handler.create_messages.side_effect = RuntimeError("boom")

        app = self._make_api_app(handler)
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/v1/messages", json=self._request_body())

        assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
        assert response.json()["error"]["type"] == "InternalServerError"

    def test_count_tokens_validation_error_returns_bad_request(self):
        """The count_tokens route maps conversion errors to 400 as well."""
        handler = MagicMock(spec=AnthropicServingMessages)
        handler.count_tokens.side_effect = self._conversion_error()

        app = self._make_api_app(handler)
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/v1/messages/count_tokens", json=self._request_body()
            )

        assert response.status_code == HTTPStatus.BAD_REQUEST
        assert response.json()["error"]["type"] == "BadRequestError"


# ======================================================================
# thinking configuration pass-through
# ======================================================================


class TestThinkingConfig:
    def test_absent_thinking_leaves_reasoning_untouched(self):
        """Requests without `thinking` must convert exactly as before."""
        request = _make_request([{"role": "user", "content": "Hello"}])

        result = _convert(request)
        assert result.reasoning_effort is None
        assert result.thinking_token_budget is None
        assert result.include_reasoning is True

    def test_disabled_clears_reasoning_effort(self):
        """`disabled` maps to reasoning_effort="none", which is what clears
        enable_thinking for templates that honor it."""
        request = _make_request(
            [{"role": "user", "content": "Hello"}],
            thinking={"type": "disabled"},
        )

        result = _convert(request)
        assert result.reasoning_effort == "none"

    def test_disabled_overrides_output_config_effort(self):
        """`thinking` is applied after output_config so an explicit opt-out wins
        over an inherited effort ceiling."""
        request = _make_request(
            [{"role": "user", "content": "Hello"}],
            output_config={"effort": "high"},
            thinking={"type": "disabled"},
        )

        result = _convert(request)
        assert result.reasoning_effort == "none"

    def test_enabled_sets_thinking_token_budget(self):
        request = AnthropicMessagesRequest(
            model="test-model",
            max_tokens=4096,
            messages=[{"role": "user", "content": "Hello"}],
            thinking={"type": "enabled", "budget_tokens": 2048},
        )

        result = _convert(request)
        assert result.thinking_token_budget == 2048
        assert result.reasoning_effort is None

    @pytest.mark.parametrize(
        "thinking",
        [
            pytest.param({"budget_tokens": 2048}, id="missing-type"),
            pytest.param({"type": "enabled"}, id="enabled-missing-budget"),
            pytest.param(
                {"type": "enabled", "budget_tokens": 1023}, id="budget-below-1024"
            ),
            pytest.param(
                {"type": "enabled", "budget_tokens": 4096}, id="budget-not-below-max"
            ),
            pytest.param({"type": "adaptive", "display": "full"}, id="bad-display"),
        ],
    )
    def test_rejects_invalid_thinking(self, thinking):
        """Mirror the Anthropic API's BetaThinkingConfigParam constraints."""
        with pytest.raises(ValidationError):
            AnthropicMessagesRequest(
                model="test-model",
                max_tokens=4096,
                messages=[{"role": "user", "content": "Hello"}],
                thinking=thinking,
            )

    def test_adaptive_pins_nothing_and_keeps_effort_ceiling(self):
        """`adaptive` lets the model choose depth, so only the ceiling from
        output_config.effort should survive."""
        request = _make_request(
            [{"role": "user", "content": "Hello"}],
            output_config={"effort": "low"},
            thinking={"type": "adaptive"},
        )

        result = _convert(request)
        assert result.reasoning_effort == "low"
        assert result.thinking_token_budget is None

    def test_disabled_uses_configured_effort(self):
        """Models that always think (GLM-5.3) or reject "none" (Harmony) are
        served with a low effort instead."""
        request = _make_request(
            [{"role": "user", "content": "Hello"}],
            thinking={"type": "disabled"},
        )

        result = _convert(request, disabled_thinking_effort="low")
        assert result.reasoning_effort == "low"

    @pytest.mark.parametrize("display", ["omitted", "summarized", "updates"])
    def test_display_keeps_reasoning_included(self, display):
        """Suppressing reasoning would mark it ended for structured outputs and
        drop it from multi-turn history, so `display` is ignored."""
        request = _make_request(
            [{"role": "user", "content": "Hello"}],
            thinking={"type": "adaptive", "display": display},
        )

        result = _convert(request)
        assert result.include_reasoning is True

    def test_claude_code_payload(self):
        """The combination Claude Code sends on every request: an effort ceiling
        plus adaptive thinking with reasoning display omitted."""
        request = _make_request(
            [{"role": "user", "content": "Hello"}],
            output_config={"effort": "high"},
            thinking={"type": "adaptive", "display": "omitted"},
        )

        result = _convert(request)
        assert result.reasoning_effort == "high"
        assert result.include_reasoning is True
        assert result.thinking_token_budget is None


class TestProbeDisabledThinkingEffort:
    """``auto`` falls back to ``low`` when ``none`` cannot turn thinking off."""

    @staticmethod
    async def _probe(render):
        obj = MagicMock(spec=AnthropicServingMessages)
        obj.online_renderer = MagicMock()
        obj.online_renderer.render_chat = AsyncMock(
            side_effect=lambda req: ([], [render(req.reasoning_effort)])
        )
        obj._extract_prompt_components = lambda engine_input: SimpleNamespace(
            token_ids=engine_input, text=None
        )
        obj._render_probe_prompt = (
            AnthropicServingMessages._render_probe_prompt.__get__(obj)
        )
        return await AnthropicServingMessages._probe_disabled_thinking_effort(obj)

    @staticmethod
    def _reject_none(effort):
        """Harmony (gpt-oss) raises on ``none``."""
        if effort == "none":
            raise ValueError(f"unsupported {effort=}")
        return [1]

    @staticmethod
    def _none_as_max(effort):
        """GLM-5.3 treats efforts other than low/high as max."""
        return [1, {"low": 0, "high": 1}.get(effort, 2)]

    @pytest.mark.asyncio
    async def test_template_honors_none(self):
        assert await self._probe(lambda effort: [1, int(effort == "none")]) == "none"

    @pytest.mark.asyncio
    async def test_template_ignores_effort(self):
        assert await self._probe(lambda effort: [1]) == "low"

    @pytest.mark.asyncio
    async def test_template_renders_none_as_thinking_effort(self):
        assert await self._probe(self._none_as_max) == "low"

    @pytest.mark.asyncio
    async def test_renderer_rejects_none(self):
        assert await self._probe(self._reject_none) == "low"
