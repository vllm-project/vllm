# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TypedDict

import pytest
import regex as re

from tests.reasoning.utils import run_reasoning_extraction
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.parser.abstract_parser import DelegatingParser
from vllm.parser.engine.registered_adapters import (
    GraniteThinkingParserReasoningAdapter,
)
from vllm.reasoning import ReasoningParser, ReasoningParserManager

parser_name = "granite_thinking_parser"


class ReasoningCase(TypedDict):
    output: str
    reasoning: str | None
    content: str | None


class GraniteThinkingTokenizer:
    def __init__(self):
        self._vocab = {
            "<think>": 1,
            "</think>": 2,
        }
        self._inv_vocab = {v: k for k, v in self._vocab.items()}
        self._pattern = re.compile(r"(<think>|</think>)")

    def get_vocab(self) -> dict[str, int]:
        return self._vocab

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        for part in self._pattern.split(text):
            if part:
                tokens.append(part)
        return tokens

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self._inv_vocab.get(tid, f"<unk:{tid}>") for tid in token_ids)


@pytest.fixture
def tokenizer():
    return GraniteThinkingTokenizer()


# ── Basic reasoning extraction (non-streaming + streaming) ───────────


@pytest.mark.parametrize(
    "streaming,param_dict",
    [
        pytest.param(
            False,
            {
                "output": "<think>reasoning</think>\nHello",
                "reasoning": "reasoning",
                "content": "Hello",
            },
            id="leading_newline_stripped",
        ),
        pytest.param(
            True,
            {
                "output": "<think>reasoning</think>\nHello",
                "reasoning": "reasoning",
                "content": "Hello",
            },
            id="leading_newline_stripped_streaming",
        ),
        pytest.param(
            False,
            {
                "output": "<think>r</think>c",
                "reasoning": "r",
                "content": "c",
            },
            id="simple_reasoning",
        ),
        pytest.param(
            True,
            {
                "output": "<think>r</think>c",
                "reasoning": "r",
                "content": "c",
            },
            id="simple_reasoning_streaming",
        ),
        pytest.param(
            False,
            {
                "output": "This is a reasoning section</think>This is the rest",
                "reasoning": "This is a reasoning section",
                "content": "This is the rest",
            },
            id="without_start_token",
        ),
        pytest.param(
            True,
            {
                "output": "This is a reasoning section</think>This is the rest",
                "reasoning": "This is a reasoning section",
                "content": "This is the rest",
            },
            id="without_start_token_streaming",
        ),
        pytest.param(
            False,
            {
                "output": "<think>This is a reasoning section</think>This is the rest",  # noqa: E501
                "reasoning": "This is a reasoning section",
                "content": "This is the rest",
            },
            id="with_start_token",
        ),
        pytest.param(
            True,
            {
                "output": "<think>This is a reasoning section</think>This is the rest",  # noqa: E501
                "reasoning": "This is a reasoning section",
                "content": "This is the rest",
            },
            id="with_start_token_streaming",
        ),
        pytest.param(
            False,
            {
                "output": "<think>reasoning</think>\n\n\nHello",
                "reasoning": "reasoning",
                "content": "Hello",
            },
            id="multiple_leading_newlines_stripped",
        ),
        pytest.param(
            True,
            {
                "output": "<think>reasoning</think>\n\n\nHello",
                "reasoning": "reasoning",
                "content": "Hello",
            },
            id="multiple_leading_newlines_stripped_streaming",
        ),
        pytest.param(
            False,
            {
                "output": "<think>line1\nline2</think>\nresult1\nresult2",
                "reasoning": "line1\nline2",
                "content": "result1\nresult2",
            },
            id="multiline_reasoning_and_content",
        ),
        pytest.param(
            True,
            {
                "output": "<think>line1\nline2</think>\nresult1\nresult2",
                "reasoning": "line1\nline2",
                "content": "result1\nresult2",
            },
            id="multiline_reasoning_and_content_streaming",
        ),
    ],
)
def test_granite_thinking_reasoning(
    tokenizer: GraniteThinkingTokenizer,
    streaming: bool,
    param_dict: ReasoningCase,
):
    output = tokenizer.tokenize(param_dict["output"])
    model_output = [tokenizer.convert_tokens_to_string([token]) for token in output]
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        tokenizer
    )

    reasoning, content = run_reasoning_extraction(
        parser, model_output, streaming=streaming
    )

    assert reasoning == param_dict["reasoning"]
    assert content == param_dict["content"]


# ── No content after end token ───────────────────────────────────────


def test_granite_thinking_no_content_after_end_token(
    tokenizer: GraniteThinkingTokenizer,
):
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    parser = parser_cls(tokenizer)

    reasoning, content = run_reasoning_extraction(
        parser,
        ["<think>", "reasoning", "</think>"],
        streaming=False,
    )

    assert reasoning == "reasoning"
    assert content is None


# ── Whitespace-only content after end token ──────────────────────────


@pytest.mark.parametrize("streaming", [False, True])
def test_granite_thinking_whitespace_only_content(
    tokenizer: GraniteThinkingTokenizer,
    streaming: bool,
):
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    parser = parser_cls(tokenizer)

    reasoning, content = run_reasoning_extraction(
        parser,
        ["<think>", "reasoning", "</think>", "\n\n"],
        streaming=streaming,
    )

    assert reasoning == "reasoning"
    assert content is None


# ── Unterminated think block ─────────────────────────────────────────


def test_granite_thinking_unterminated_think_block(
    tokenizer: GraniteThinkingTokenizer,
):
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    parser = parser_cls(tokenizer)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        chat_template_kwargs={"enable_thinking": True},
    )

    reasoning, content = run_reasoning_extraction(
        parser,
        ["<think>", "reasoning only"],
        request=request,
        streaming=False,
    )

    assert reasoning == "reasoning only"
    assert content is None


# ── enable_thinking=False ────────────────────────────────────────────


def test_granite_thinking_disabled_moves_into_content(
    tokenizer: GraniteThinkingTokenizer,
):
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    parser = parser_cls(tokenizer)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        chat_template_kwargs={"enable_thinking": False},
    )

    reasoning, content = run_reasoning_extraction(
        parser,
        ["This is plain content"],
        request=request,
        streaming=False,
    )

    assert reasoning is None
    assert content == "This is plain content"


# ── enable_thinking=False + leading newline (§6.2 ordering) ─────────


def test_granite_thinking_disabled_with_leading_newline(
    tokenizer: GraniteThinkingTokenizer,
):
    # With enable_thinking=False, model output goes through the swap
    # path: content is initially None (all text classified as
    # reasoning), so lstrip doesn't fire pre-swap. After the swap,
    # any leading \n in the original output is preserved. In practice,
    # enable_thinking=False output has no template-injected \n, so
    # this is a correctness check, not a realistic scenario.
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    parser = parser_cls(tokenizer)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        chat_template_kwargs={"enable_thinking": False},
    )

    reasoning, content = run_reasoning_extraction(
        parser,
        ["\nThis is plain content"],
        request=request,
        streaming=False,
    )

    assert reasoning is None
    assert content == "\nThis is plain content"


# ── force_nonempty_content=True ──────────────────────────────────────


def test_granite_thinking_force_nonempty_content_moves_into_content(
    tokenizer: GraniteThinkingTokenizer,
):
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    parser = parser_cls(tokenizer)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        chat_template_kwargs={"force_nonempty_content": True},
    )

    reasoning, content = run_reasoning_extraction(
        parser,
        ["<think>", "This is plain content"],
        request=request,
        streaming=False,
    )

    assert reasoning is None
    assert content == "This is plain content"


def test_granite_thinking_force_nonempty_no_swap_when_newlines_only(
    tokenizer: GraniteThinkingTokenizer,
):
    # When </think> IS present and content is newlines-only, lstrip
    # removes them but the swap should NOT fire — content was present,
    # just whitespace. Matches the HF plugin behavior.
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    parser = parser_cls(tokenizer)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        chat_template_kwargs={"force_nonempty_content": True},
    )

    reasoning, content = run_reasoning_extraction(
        parser,
        ["<think>", "reasoning", "</think>", "\n\n"],
        request=request,
        streaming=False,
    )

    assert reasoning == "reasoning"
    assert content is None


def test_granite_thinking_force_nonempty_swaps_when_content_absent(
    tokenizer: GraniteThinkingTokenizer,
):
    # When </think> IS present but content is truly absent (zero
    # characters after </think>, e.g. max_tokens cut), swap fires.
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    parser = parser_cls(tokenizer)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        chat_template_kwargs={"force_nonempty_content": True},
    )

    reasoning, content = run_reasoning_extraction(
        parser,
        ["<think>", "reasoning", "</think>"],
        request=request,
        streaming=False,
    )

    assert reasoning is None
    assert content == "reasoning"


def test_granite_thinking_force_nonempty_keeps_real_content(
    tokenizer: GraniteThinkingTokenizer,
):
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    parser = parser_cls(tokenizer)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        chat_template_kwargs={"force_nonempty_content": True},
    )

    reasoning, content = run_reasoning_extraction(
        parser,
        ["<think>", "reasoning here", "</think>", "real answer"],
        request=request,
        streaming=False,
    )

    assert reasoning == "reasoning here"
    assert content == "real answer"


# ── Truncated reasoning with thinking on ─────────────────────────────


def test_granite_thinking_keeps_truncated_reasoning(
    tokenizer: GraniteThinkingTokenizer,
):
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    parser = parser_cls(tokenizer)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        chat_template_kwargs={"enable_thinking": True},
    )

    reasoning, content = run_reasoning_extraction(
        parser,
        ["This is truncated reasoning"],
        request=request,
        streaming=False,
    )

    assert reasoning == "This is truncated reasoning"
    assert content is None


# ── DelegatingParser / parse_delta streaming tests ───────────────────

_SPECIAL_TOKEN_IDS = {"<think>": 1, "</think>": 2}


def _token_id(token: str) -> int:
    return _SPECIAL_TOKEN_IDS.get(token, 0)


def _make_reasoning_parser(tokenizer):
    class _GraniteThinkingDelegating(DelegatingParser):
        reasoning_parser_cls = GraniteThinkingParserReasoningAdapter
        tool_parser_cls = None

    return _GraniteThinkingDelegating(tokenizer)


def _run_parse_delta(parser, tokenizer, text, request):
    tokens = tokenizer.tokenize(text)
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    for i, token in enumerate(tokens):
        delta = parser.parse_delta(
            delta_text=token,
            delta_token_ids=[_token_id(token)],
            request=request,
            prompt_token_ids=[] if i == 0 else None,
            finished=(i == len(tokens) - 1),
        )
        if delta is None:
            continue
        if delta.reasoning:
            reasoning_parts.append(delta.reasoning)
        if delta.content:
            content_parts.append(delta.content)
    return "".join(reasoning_parts), "".join(content_parts)


def test_granite_thinking_streaming_enable_thinking_false(
    tokenizer: GraniteThinkingTokenizer,
):
    # With enable_thinking=False, the parser (constructed without
    # kwargs) starts in REASONING state. All text streams as reasoning
    # AND is duplicated into content via the streaming fallback —
    # matching NemotronV3 behavior.
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        chat_template_kwargs={"enable_thinking": False},
    )
    parser = _make_reasoning_parser(tokenizer)

    reasoning, content = _run_parse_delta(
        parser, tokenizer, "This is plain content", request
    )

    assert reasoning == "This is plain content"
    assert content == "This is plain content"


def test_granite_thinking_streaming_strips_leading_newline(
    tokenizer: GraniteThinkingTokenizer,
):
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
    )
    parser = _make_reasoning_parser(tokenizer)

    reasoning, content = _run_parse_delta(
        parser, tokenizer, "<think>reason</think>\nHello", request
    )

    assert reasoning == "reason"
    assert content == "Hello"


def test_granite_thinking_streaming_promotes_reasoning_to_content(
    tokenizer: GraniteThinkingTokenizer,
):
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        chat_template_kwargs={"force_nonempty_content": True},
    )
    parser = _make_reasoning_parser(tokenizer)

    reasoning, content = _run_parse_delta(parser, tokenizer, "<think>4", request)

    assert reasoning == "4"
    assert content == "4"


def test_granite_thinking_streaming_no_promotion_with_real_content(
    tokenizer: GraniteThinkingTokenizer,
):
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        chat_template_kwargs={"force_nonempty_content": True},
    )
    parser = _make_reasoning_parser(tokenizer)

    reasoning, content = _run_parse_delta(
        parser,
        tokenizer,
        "<think>reason</think>real answer",
        request,
    )

    assert reasoning == "reason"
    assert content == "real answer"


def test_granite_thinking_streaming_no_promotion_without_opt_in(
    tokenizer: GraniteThinkingTokenizer,
):
    request = ChatCompletionRequest(model="test-model", messages=[])
    parser = _make_reasoning_parser(tokenizer)

    reasoning, content = _run_parse_delta(parser, tokenizer, "<think>4", request)

    assert reasoning == "4"
    assert content == ""


# ── Empty think block ────────────────────────────────────────────────


@pytest.mark.parametrize("streaming", [False, True])
def test_granite_thinking_empty_think_block(
    tokenizer: GraniteThinkingTokenizer,
    streaming: bool,
):
    # <think></think>\nHello — empty reasoning, content after newline.
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    parser = parser_cls(tokenizer)

    reasoning, content = run_reasoning_extraction(
        parser,
        ["<think>", "</think>", "\n", "Hello"],
        streaming=streaming,
    )

    assert reasoning is None or reasoning == ""
    assert content == "Hello"


# ── Chunking-independence streaming test ─────────────────────────────


@pytest.mark.parametrize(
    "text,expected_reasoning,expected_content",
    [
        ("<think>r</think>\nHello", "r", "Hello"),
        ("<think></think>\nHello", "", "Hello"),
    ],
)
def test_granite_thinking_streaming_chunking_independent(
    tokenizer: GraniteThinkingTokenizer,
    text: str,
    expected_reasoning: str,
    expected_content: str,
):
    # Verify the same input produces identical results regardless
    # of how it's chunked into streaming deltas.  Uses the
    # DelegatingParser path (_run_parse_delta) which matches the
    # real serving flow.
    tokens = tokenizer.tokenize(text)
    request = ChatCompletionRequest(model="test-model", messages=[])

    chunk_patterns = [
        tokens,
        # Split at the </think> boundary
        [
            tokenizer.convert_tokens_to_string(tokens[: tokens.index("</think>") + 1]),
            tokenizer.convert_tokens_to_string(tokens[tokens.index("</think>") + 1 :]),
        ],
    ]

    for chunks in chunk_patterns:
        parser = _make_reasoning_parser(tokenizer)
        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        all_tokens = []
        for chunk in chunks:
            chunk_tokens = tokenizer.tokenize(chunk)
            all_tokens.extend(chunk_tokens)
        for i, token in enumerate(all_tokens):
            delta = parser.parse_delta(
                delta_text=token,
                delta_token_ids=[_token_id(token)],
                request=request,
                prompt_token_ids=[] if i == 0 else None,
                finished=(i == len(all_tokens) - 1),
            )
            if delta is None:
                continue
            if delta.reasoning:
                reasoning_parts.append(delta.reasoning)
            if delta.content:
                content_parts.append(delta.content)
        reasoning = "".join(reasoning_parts)
        content = "".join(content_parts)
        assert reasoning == expected_reasoning or (
            not expected_reasoning and not reasoning
        ), f"reasoning mismatch with chunks={chunks}"
        assert content == expected_content, f"content mismatch with chunks={chunks}"
