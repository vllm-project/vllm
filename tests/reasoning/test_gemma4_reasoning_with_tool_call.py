# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Combined reasoning + tool-call parsing tests for Gemma4.

Exercises DelegatingParser.parse_delta() with both Gemma4ReasoningParser
and Gemma4ToolParser active — the scenario where <|channel>thought...<channel|>
precedes a tool call, covering both token-by-token and single-delta (large
stream-interval) delivery.
"""

import pytest
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.abstract_parser import DelegatingParser
from vllm.reasoning.gemma4_reasoning_parser import Gemma4ReasoningParser
from vllm.tokenizers.registry import get_tokenizer
from vllm.tool_parsers.gemma4_tool_parser import Gemma4ToolParser

TOKENIZER_NAME = "google/gemma-4-E2B-it"


class _Gemma4Parser(DelegatingParser):
    reasoning_parser_cls = Gemma4ReasoningParser
    tool_parser_cls = Gemma4ToolParser


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer(TOKENIZER_NAME)


@pytest.fixture
def parser(tokenizer):
    """Fresh parser per test — avoids _reasoning_text/_prefix_stripped state leak."""
    return _Gemma4Parser(tokenizer)


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
            return (
                _encode(tokenizer, parts[0]) + [tok_id] + _encode(tokenizer, parts[1])
            )
    try:
        return enc.encode(text, add_special_tokens=False)
    except TypeError:
        return enc.encode(text)


_DUMMY_TOOL = ChatCompletionToolParam(
    type="function",
    function={"name": "find", "description": "Find files", "parameters": {}},
)


def _make_request():
    req = ChatCompletionRequest(messages=[], model="gemma4-test", tools=[_DUMMY_TOOL])
    req.skip_special_tokens = False
    return req


def _run_streaming(parser_instance, token_strings: list[str], tokenizer):
    """Feed token strings one at a time through parse_delta."""
    vocab = tokenizer.get_vocab()
    enc = getattr(tokenizer, "tokenizer", tokenizer)
    request = _make_request()
    reasoning_parts, content_parts, tool_calls = [], [], []

    for i, tok_str in enumerate(token_strings):
        tok_id = vocab.get(tok_str)
        if tok_id is not None:
            ids = [tok_id]
        else:
            try:
                ids = enc.encode(tok_str, add_special_tokens=False)
            except TypeError:
                ids = enc.encode(tok_str)

        delta = parser_instance.parse_delta(
            tok_str, ids, request, finished=(i == len(token_strings) - 1)
        )
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
    delta = parser_instance.parse_delta(full_text, full_ids, request, finished=True)
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
    token_strings = [
        "<|channel>",
        "thought",
        "\n",
        "I",
        " need",
        " to",
        " find",
        " files",
        "<channel|>",
    ] + [
        "<|tool_call>",
        "call",
        ":",
        "find",
        "{",
        "path",
        ":",
        '<|"|>',
        "research",
        '<|"|>',
        "}",
        "<tool_call|>",
    ]
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
        "<|channel>thought\nI need to find files<channel|>"
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


def test_reasoning_after_tool_response(parser, tokenizer):
    """Second-turn generation: reasoning must not leak when prompt has a prior
    completed tool call + tool response (the multi-turn reasoning-leak bug).

    Simulates: prompt_token_ids ends with <|tool_call>...<|tool_response>...
    which used to make is_reasoning_end() return True (finding the prior
    <|tool_call> while searching backward past <|tool_response>), causing
    reasoning_ended=True at the very start and leaking <|channel>thought...
    tokens as content.
    """
    vocab = tokenizer.get_vocab()

    tool_call_tok = vocab.get("<|tool_call>")
    tool_call_end_tok = vocab.get("<tool_call|>")
    tool_resp_tok = vocab.get("<|tool_response>")
    tool_resp_end_tok = vocab.get("<tool_response|>")

    # Synthetic prompt_token_ids: simulate a completed first-turn tool exchange.
    # The structure mirrors the Gemma4 template output:
    #   <|tool_call>body<tool_call|><|tool_response>body<tool_response|>
    # The <tool_response|> end marker is required for is_reasoning_end to
    # distinguish this (completed exchange) from a bare stop token.
    prompt_ids: list[int] = []
    if tool_call_tok is not None:
        prompt_ids.append(tool_call_tok)
    prompt_ids += [1000, 1001, 1002]  # tool call body tokens
    if tool_call_end_tok is not None:
        prompt_ids.append(tool_call_end_tok)
    if tool_resp_tok is not None:
        prompt_ids.append(tool_resp_tok)
    prompt_ids += [2000, 2001]  # tool response body tokens
    if tool_resp_end_tok is not None:
        prompt_ids.append(tool_resp_end_tok)

    request = _make_request()
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    tool_calls_found: list = []

    # Feed second-turn generation as individual token strings, passing
    # prompt_token_ids only on the very first call (mimics parse_delta usage).
    enc = getattr(tokenizer, "tokenizer", tokenizer)
    first = True
    for tok_str in [
        "<|channel>",
        "thought",
        "\n",
        "I",
        " need",
        " to",
        " answer",
        "<channel|>",
        "The",
        " answer",
        " is",
        " 42",
    ]:
        tok_id = vocab.get(tok_str)
        if tok_id is not None:
            ids = [tok_id]
        else:
            try:
                ids = enc.encode(tok_str, add_special_tokens=False)
            except TypeError:
                ids = enc.encode(tok_str)

        delta = parser.parse_delta(
            tok_str,
            ids,
            request,
            prompt_token_ids=prompt_ids if first else None,
            finished=False,
        )
        first = False
        if delta is None:
            continue
        if delta.reasoning:
            reasoning_parts.append(delta.reasoning)
        if delta.content:
            content_parts.append(delta.content)
        if delta.tool_calls:
            tool_calls_found.extend(delta.tool_calls)

    reasoning = "".join(reasoning_parts) or None
    content = "".join(content_parts) or None

    assert reasoning is not None, (
        "reasoning was lost in second-turn generation after tool response in prompt"
    )
    assert not reasoning.startswith("thought"), (
        f"'thought\\n' prefix must be stripped; got {reasoning!r}"
    )
    assert "<|channel>" not in reasoning
    assert "<channel|>" not in reasoning

    assert content is not None, "content after reasoning must not be dropped"
    assert "42" in content, f"expected '42' in content, got {content!r}"
    assert len(tool_calls_found) == 0

    # No raw thinking tokens should have leaked into content
    assert "<|channel>" not in (content or ""), (
        "thinking start token leaked into content"
    )
    assert "<channel|>" not in (content or ""), "thinking end token leaked into content"


def test_reasoning_only_no_tool_call(parser, tokenizer):
    """Reasoning only (no tool call): content passes through cleanly."""
    token_strings = [
        "<|channel>",
        "thought",
        "\n",
        "Let",
        " me",
        " think",
        "<channel|>",
    ] + ["The", " answer", " is", " 42"]
    reasoning, content, tool_calls = _run_streaming(parser, token_strings, tokenizer)

    assert reasoning is not None
    assert not reasoning.startswith("thought"), (
        f"'thought\\n' prefix must be stripped; got {reasoning!r}"
    )
    assert content is not None
    assert "42" in content
    assert len(tool_calls) == 0


def test_empty_thinking_block_tool_call_no_reasoning_leak(parser, tokenizer):
    """Empty thinking block (<|channel>thought\\n<channel|>) followed by a
    tool call must NOT emit an empty-string reasoning_content delta.

    When the model produces only the 'thought\\n' role label (nothing after
    it inside the channel) the prefix-stripping logic previously returned
    DeltaMessage(reasoning='') — an empty string, not None.  The harness
    received {"reasoning_content": ""} and mis-rendered it.  The fix makes
    the parser return None (or forward the post-channel content only) so
    no empty reasoning delta is ever emitted.

    Exercises both token-by-token and single-delta delivery.
    """
    # Token-by-token: each token arrives individually.
    token_strings = ["<|channel>", "thought", "\n", "<channel|>"] + [
        "<|tool_call>",
        "call",
        ":",
        "find",
        "{",
        "path",
        ":",
        '<|"|>',
        "research",
        '<|"|>',
        "}",
        "<tool_call|>",
    ]
    reasoning, content, tool_calls = _run_streaming(parser, token_strings, tokenizer)

    assert reasoning is None, (
        f"empty thinking block must not emit reasoning_content; got {reasoning!r}"
    )
    assert len(tool_calls) >= 1, "tool call must still be parsed"
    assert tool_calls[0].function.name == "find"

    # Single-delta: the whole output arrives in one chunk (stream-interval 20).
    parser2 = _Gemma4Parser(tokenizer)

    full_text = (
        "<|channel>thought\n<channel|>"
        '<|tool_call>call:find{path:<|"|>research<|"|>}<tool_call|>'
    )
    reasoning2, content2, tool_calls2 = _run_single_delta(parser2, full_text, tokenizer)

    assert reasoning2 is None, (
        f"single-delta empty thinking must not emit reasoning_content; "
        f"got {reasoning2!r}"
    )
    assert len(tool_calls2) >= 1, "tool call must still be parsed in single-delta mode"
    assert tool_calls2[0].function.name == "find"
