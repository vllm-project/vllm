# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Regression tests for Gemma4 reasoning-to-tool-call transition.

Issue #40911: partial tool-call markers (e.g. "<|" from "<|tool_call>")
can leak into content when the reasoning end token and the tool-call
start token arrive in the same streaming delta.  The fix lives in
Gemma4ReasoningParser.extract_reasoning_streaming(), which reconstructs
content from token IDs instead of relying on text-based splitting.
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
    """Create a DelegatingParser with Gemma4 parsers using a local subclass
    to avoid polluting _WrappedParser class attributes across tests."""

    class _Gemma4TestParser(_WrappedParser):
        reasoning_parser_cls = Gemma4ReasoningParser
        tool_parser_cls = Gemma4ToolParser

    return _Gemma4TestParser(tokenizer)


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

        Simulates the actual failure mode: the reasoning end token and
        tool-call start token arrive in the same delta, but delta_text
        contains only a partial prefix of the tool marker (e.g. "<|")
        due to incremental detokenization.  The partial prefix must not
        appear as content.
        """
        parser = _make_parser(tokenizer)
        request = _request()

        channel_end_id = vocab["<channel|>"]
        channel_start_id = vocab["<|channel>"]
        tool_call_start_id = vocab.get("<|tool_call>")
        if tool_call_start_id is None:
            pytest.skip("<|tool_call> not in vocab")

        # Phase 1: feed reasoning tokens one by one
        reasoning_tokens = _encode(tokenizer, "thinking about tools")
        for tid in [channel_start_id] + reasoning_tokens:
            delta_text = tokenizer.decode([tid], skip_special_tokens=False)
            parser.parse_delta(
                delta_text=delta_text,
                delta_token_ids=[tid],
                request=request,
            )

        # Phase 2: reasoning end + tool call start in same delta.
        # Construct delta_text with a partial prefix of the tool token
        # to simulate the actual failure mode (incremental detokenization
        # can produce a prefix-diff that splits the special token text).
        channel_end_text = tokenizer.decode([channel_end_id], skip_special_tokens=False)
        tool_token_text = tokenizer.decode(
            [tool_call_start_id], skip_special_tokens=False
        )
        partial_prefix = tool_token_text[:2]  # e.g. "<|"
        combined_ids = [channel_end_id, tool_call_start_id]
        combined_text = channel_end_text + partial_prefix

        msg = parser.parse_delta(
            delta_text=combined_text,
            delta_token_ids=combined_ids,
            request=request,
        )

        if msg is not None and msg.content is not None:
            assert partial_prefix not in msg.content, (
                f"Partial tool-call marker leaked as content: {msg.content!r}"
            )

    def test_separate_deltas_no_leak(self, tokenizer, vocab):
        """
        When reasoning end and tool-call start arrive in separate deltas,
        the tool marker must not leak as content.
        """
        parser = _make_parser(tokenizer)
        request = _request()

        channel_start_id = vocab["<|channel>"]
        channel_end_id = vocab["<channel|>"]
        tool_call_start_id = vocab.get("<|tool_call>")
        if tool_call_start_id is None:
            pytest.skip("<|tool_call> not in vocab")

        # Reasoning start + content
        delta_text = tokenizer.decode([channel_start_id], skip_special_tokens=False)
        parser.parse_delta(
            delta_text=delta_text,
            delta_token_ids=[channel_start_id],
            request=request,
        )
        for tid in _encode(tokenizer, "reasoning"):
            delta_text = tokenizer.decode([tid], skip_special_tokens=False)
            parser.parse_delta(
                delta_text=delta_text,
                delta_token_ids=[tid],
                request=request,
            )

        # Reasoning end (separate delta)
        delta_text = tokenizer.decode([channel_end_id], skip_special_tokens=False)
        parser.parse_delta(
            delta_text=delta_text,
            delta_token_ids=[channel_end_id],
            request=request,
        )

        # Tool call start (separate delta) — should not leak as content
        delta_text = tokenizer.decode([tool_call_start_id], skip_special_tokens=False)
        msg = parser.parse_delta(
            delta_text=delta_text,
            delta_token_ids=[tool_call_start_id],
            request=request,
        )

        if msg is not None and msg.content is not None:
            assert "<|tool_call>" not in msg.content, (
                f"Tool marker leaked as content: {msg.content!r}"
            )

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

        if msg is not None and msg.content is not None:
            assert msg.content.strip() == "", (
                f"Unexpected content at reasoning end: {msg.content!r}"
            )
