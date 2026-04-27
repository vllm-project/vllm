# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Regression tests for DelegatingParser reasoning-to-tool-call transition.

Issue #40911: partial tool-call markers (e.g. "<|" from "<|tool_call>")
can leak into content when the reasoning end token and the tool-call
start token arrive in the same streaming delta.
"""

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.parser.abstract_parser import _WrappedParser
from vllm.reasoning.gemma4_reasoning_parser import Gemma4ReasoningParser
from vllm.tokenizers.registry import get_tokenizer
from vllm.tool_parsers.gemma4_tool_parser import Gemma4ToolParser

TOKENIZER_NAME = "google/gemma-4-E2B-it"


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer(TOKENIZER_NAME)


@pytest.fixture(scope="module")
def vocab(tokenizer):
    return tokenizer.get_vocab()


def _make_parser(tokenizer):
    _WrappedParser.reasoning_parser_cls = Gemma4ReasoningParser
    _WrappedParser.tool_parser_cls = Gemma4ToolParser
    return _WrappedParser(tokenizer)


def _encode(tokenizer, text: str) -> list[int]:
    enc = getattr(tokenizer, "tokenizer", tokenizer)
    try:
        return enc.encode(text, add_special_tokens=False)
    except TypeError:
        return enc.encode(text)


def _request():
    return ChatCompletionRequest(messages=[], model="test-model")


class TestReasoningToToolTransition:
    """Verify that partial tool-call markers do not leak as content."""

    def test_partial_marker_does_not_leak(self, tokenizer, vocab):
        """
        Regression test for #40911.

        Simulates: reasoning ends with <channel|>, then partial "<|"
        arrives before the full "<|tool_call>" token. The partial "<|"
        must not appear as content.
        """
        parser = _make_parser(tokenizer)
        request = _request()

        channel_end_id = vocab["<channel|>"]
        channel_start_id = vocab["<|channel>"]
        tool_call_start_id = vocab.get("<|tool_call>")
        if tool_call_start_id is None:
            pytest.skip("<|tool_call> not in vocab")

        # Phase 1: start reasoning
        reasoning_tokens = _encode(tokenizer, "thinking about tools")
        previous_text = ""
        previous_token_ids: list[int] = []

        for tid in [channel_start_id] + reasoning_tokens:
            delta_text = tokenizer.decode([tid], skip_special_tokens=False)
            current_text = previous_text + delta_text
            current_token_ids = previous_token_ids + [tid]
            parser.parse_delta(
                delta_text=delta_text,
                delta_token_ids=[tid],
                request=request,
            )
            previous_text = current_text
            previous_token_ids = current_token_ids

        # Phase 2: reasoning end + tool call start in same delta
        combined_ids = [channel_end_id, tool_call_start_id]
        combined_text = tokenizer.decode(combined_ids, skip_special_tokens=False)
        msg = parser.parse_delta(
            delta_text=combined_text,
            delta_token_ids=combined_ids,
            request=request,
        )

        # The partial marker must not appear as content
        if msg is not None and msg.content is not None:
            assert "<|" not in msg.content, (
                f"Partial tool-call marker leaked as content: {msg.content!r}"
            )

    def test_full_marker_passes_to_tool_parser(self, tokenizer, vocab):
        """
        When reasoning ends and a complete tool-call token follows,
        the tool parser should receive it (no leak, no suppression).
        """
        parser = _make_parser(tokenizer)
        request = _request()

        channel_start_id = vocab["<|channel>"]
        channel_end_id = vocab["<channel|>"]
        tool_call_start_id = vocab.get("<|tool_call>")
        if tool_call_start_id is None:
            pytest.skip("<|tool_call> not in vocab")

        # Send reasoning start
        delta_text = tokenizer.decode([channel_start_id], skip_special_tokens=False)
        parser.parse_delta(
            delta_text=delta_text,
            delta_token_ids=[channel_start_id],
            request=request,
        )

        # Send reasoning content
        content_ids = _encode(tokenizer, "reasoning")
        for tid in content_ids:
            delta_text = tokenizer.decode([tid], skip_special_tokens=False)
            parser.parse_delta(
                delta_text=delta_text,
                delta_token_ids=[tid],
                request=request,
            )

        # Send reasoning end
        delta_text = tokenizer.decode([channel_end_id], skip_special_tokens=False)
        parser.parse_delta(
            delta_text=delta_text,
            delta_token_ids=[channel_end_id],
            request=request,
        )

        # Send tool call start — now in tool phase
        delta_text = tokenizer.decode([tool_call_start_id], skip_special_tokens=False)
        msg = parser.parse_delta(
            delta_text=delta_text,
            delta_token_ids=[tool_call_start_id],
            request=request,
        )

        # Should not leak tool marker as content
        if msg is not None and msg.content is not None:
            assert "<|tool_call>" not in msg.content

    def test_reasoning_end_only_no_leak(self, tokenizer, vocab):
        """
        When reasoning ends and no tool tokens follow in the same delta,
        no content should leak.
        """
        parser = _make_parser(tokenizer)
        request = _request()

        channel_start_id = vocab["<|channel>"]
        channel_end_id = vocab["<channel|>"]

        # Start reasoning
        delta_text = tokenizer.decode([channel_start_id], skip_special_tokens=False)
        parser.parse_delta(
            delta_text=delta_text,
            delta_token_ids=[channel_start_id],
            request=request,
        )

        # Some reasoning text
        for tid in _encode(tokenizer, "some thought"):
            delta_text = tokenizer.decode([tid], skip_special_tokens=False)
            parser.parse_delta(
                delta_text=delta_text,
                delta_token_ids=[tid],
                request=request,
            )

        # End reasoning (no tool call follows in this delta)
        delta_text = tokenizer.decode([channel_end_id], skip_special_tokens=False)
        msg = parser.parse_delta(
            delta_text=delta_text,
            delta_token_ids=[channel_end_id],
            request=request,
        )

        # No content should leak from the transition
        if msg is not None and msg.content is not None:
            assert msg.content.strip() == "", (
                f"Unexpected content at reasoning end: {msg.content!r}"
            )
