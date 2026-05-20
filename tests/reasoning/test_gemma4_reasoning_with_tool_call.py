# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Combined reasoning + tool-call parsing tests for Gemma4.

Exercises DelegatingParser.parse_delta() with both Gemma4ReasoningParser
and Gemma4ToolParser active — the scenario where <|channel>thought...<channel|>
precedes a tool call, covering both token-by-token and single-delta (large
stream-interval) delivery.
"""

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.abstract_parser import _WrappedParser
from vllm.reasoning.gemma4_reasoning_parser import Gemma4ReasoningParser
from vllm.tool_parsers.gemma4_tool_parser import Gemma4ToolParser
from vllm.tokenizers.registry import get_tokenizer

TOKENIZER_NAME = "google/gemma-4-E2B-it"


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer(TOKENIZER_NAME)


@pytest.fixture
def parser(tokenizer):
    """Fresh parser per test — avoids _reasoning_text/_prefix_stripped state leak."""
    _WrappedParser.reasoning_parser_cls = Gemma4ReasoningParser
    _WrappedParser.tool_parser_cls = Gemma4ToolParser
    return _WrappedParser(tokenizer)


def _encode(tokenizer, text: str) -> list[int]:
    """Encode text including Gemma4 special tokens into token IDs."""
    vocab = tokenizer.get_vocab()
    enc = getattr(tokenizer, "tokenizer", tokenizer)
    for special, tok_id in [
        ("<|channel>", vocab.get("<|channel>")),
        ("<channel|>", vocab.get("<channel|>")),
        ("<|tool_call>", vocab.get("<|tool_call>")),
        ("<tool_call|>", vocab.get("<tool_call|>")),
        ('<|"|>', vocab.get('<|"|>')),
    ]:
        if special in text and tok_id is not None:
            parts = text.split(special, 1)
            return _encode(tokenizer, parts[0]) + [tok_id] + _encode(tokenizer, parts[1])
    try:
        return enc.encode(text, add_special_tokens=False)
    except TypeError:
        return enc.encode(text)


def _make_request():
    req = ChatCompletionRequest(messages=[], model="gemma4-test")
    req.skip_special_tokens = False
    return req


def _run_streaming(parser_instance, token_strings: list[str], tokenizer):
    """Feed token strings one at a time through parse_delta."""
    vocab = tokenizer.get_vocab()
    enc = getattr(tokenizer, "tokenizer", tokenizer)
    request = _make_request()
    reasoning_parts, content_parts, tool_calls = [], [], []

    for tok_str in token_strings:
        tok_id = vocab.get(tok_str)
        if tok_id is not None:
            ids = [tok_id]
        else:
            try:
                ids = enc.encode(tok_str, add_special_tokens=False)
            except TypeError:
                ids = enc.encode(tok_str)

        delta = parser_instance.parse_delta(tok_str, ids, request)
        if delta is None:
            continue
        if delta.reasoning:
            reasoning_parts.append(delta.reasoning)
        if delta.content:
            content_parts.append(delta.content)
        if delta.tool_calls:
            tool_calls.extend(delta.tool_calls)

    return (
        "".join(reasoning_parts) or None,
        "".join(content_parts) or None,
        tool_calls,
    )


def _run_single_delta(parser_instance, full_text: str, tokenizer):
    """Feed entire output as one delta (simulates large stream-interval)."""
    request = _make_request()
    full_ids = _encode(tokenizer, full_text)
    delta = parser_instance.parse_delta(full_text, full_ids, request)
    if delta is None:
        return None, None, []
    return (
        delta.reasoning or None,
        delta.content or None,
        delta.tool_calls or [],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_reasoning_then_tool_call_token_by_token(parser, tokenizer):
    """Token-by-token delivery: reasoning extracted, tool call parsed."""
    token_strings = (
        ["<|channel>", "thought", "\n", "I", " need", " to", " find", " files",
         "<channel|>"]
        + ["<|tool_call>", "call", ":", "find", "{", "path", ":", '<|"|>',
           "research", '<|"|>', "}", "<tool_call|>"]
    )
    reasoning, content, tool_calls = _run_streaming(parser, token_strings, tokenizer)

    assert reasoning is not None
    assert not reasoning.startswith("thought"), (
        f"'thought\\n' prefix must be stripped; got {reasoning!r}"
    )
    assert "<|channel>" not in reasoning
    assert "<channel|>" not in reasoning

    assert len(tool_calls) >= 1
    assert tool_calls[0].function.name == "find"


def test_reasoning_then_tool_call_single_delta(parser, tokenizer):
    """Single-delta delivery (large stream-interval): reasoning must not be lost."""
    full_text = (
        '<|channel>thought\nI need to find files<channel|>'
        '<|tool_call>call:find{path:<|"|>research<|"|>}<tool_call|>'
    )
    reasoning, content, tool_calls = _run_single_delta(parser, full_text, tokenizer)

    assert reasoning is not None, (
        "reasoning was silently dropped when tool call arrived in the same delta"
    )
    assert not reasoning.startswith("thought"), (
        f"'thought\\n' prefix must be stripped; got {reasoning!r}"
    )
    assert "<|channel>" not in reasoning
    assert "<channel|>" not in reasoning

    assert len(tool_calls) >= 1
    assert tool_calls[0].function.name == "find"


def test_reasoning_only_no_tool_call(parser, tokenizer):
    """Reasoning only (no tool call): content passes through cleanly."""
    token_strings = (
        ["<|channel>", "thought", "\n", "Let", " me", " think", "<channel|>"]
        + ["The", " answer", " is", " 42"]
    )
    reasoning, content, tool_calls = _run_streaming(parser, token_strings, tokenizer)

    assert reasoning is not None
    assert not reasoning.startswith("thought"), (
        f"'thought\\n' prefix must be stripped; got {reasoning!r}"
    )
    assert content is not None
    assert "42" in content
    assert len(tool_calls) == 0
