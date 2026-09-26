# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dedicated unit tests for IdentityReasoningParser.

The IdentityReasoningParser is a pass-through parser that treats the entire
model output as content with no reasoning extraction.  These tests verify the
full abstract-base contract: property accessors, is_reasoning_end variants,
extract_content_ids, extract_reasoning, and extract_reasoning_streaming.
A complementary smoke test for IdentityReasoningParser lives in
test_deepseekv3_reasoning_parser.py (test_identity_reasoning_parser_basic).
"""

import pytest
from transformers import AutoTokenizer

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.identity_reasoning_parser import IdentityReasoningParser

# Global tokenizer — small model, matches the rest of the reasoning test suite.
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

# Minimal request object to satisfy the abstract base contract — the identity
# parser does not inspect the request, but the type signature requires it.
_REQUEST = ChatCompletionRequest(model="test-model", messages=[], temperature=1.0)


@pytest.fixture(scope="module")
def parser() -> IdentityReasoningParser:
    return IdentityReasoningParser(tokenizer)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


def test_reasoning_start_str_is_none(parser: IdentityReasoningParser):
    assert parser.reasoning_start_str is None


def test_reasoning_end_str_is_none(parser: IdentityReasoningParser):
    assert parser.reasoning_end_str is None


# ---------------------------------------------------------------------------
# is_reasoning_end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_ids",
    [
        pytest.param([], id="empty"),
        pytest.param([1], id="single_token"),
        pytest.param([1, 2, 3, 4, 5], id="multiple_tokens"),
    ],
)
def test_is_reasoning_end_always_true(
    parser: IdentityReasoningParser, input_ids: list[int]
):
    assert parser.is_reasoning_end(input_ids) is True


# ---------------------------------------------------------------------------
# is_reasoning_end_streaming
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prev_ids, delta_ids",
    [
        pytest.param([], [], id="both_empty"),
        pytest.param([1, 2], [3], id="prev_and_single_delta"),
        pytest.param([10, 20, 30], [40, 50], id="multiple_delta"),
        pytest.param([], [100], id="no_prev_with_delta"),
    ],
)
def test_is_reasoning_end_streaming_always_true(
    parser: IdentityReasoningParser,
    prev_ids: list[int],
    delta_ids: list[int],
):
    assert parser.is_reasoning_end_streaming(prev_ids, delta_ids) is True


# ---------------------------------------------------------------------------
# extract_content_ids
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_ids",
    [
        pytest.param([], id="empty"),
        pytest.param([42], id="single"),
        pytest.param([1, 2, 3], id="three_tokens"),
        pytest.param(list(range(20)), id="twenty_tokens"),
    ],
)
def test_extract_content_ids_is_identity(
    parser: IdentityReasoningParser, input_ids: list[int]
):
    result = parser.extract_content_ids(input_ids)
    # Must return the full token list unchanged — no stripping.
    assert result == input_ids


# ---------------------------------------------------------------------------
# extract_reasoning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_output",
    [
        pytest.param("", id="empty_string"),
        pytest.param("Hello, world!", id="simple_sentence"),
        pytest.param("Line one\nLine two\nLine three", id="multiline"),
        pytest.param(
            "Some reasoning here\nAnd more reasoning\nFinal answer: 42",
            id="reasoning_like_output",
        ),
        pytest.param("   ", id="whitespace_only"),
    ],
)
def test_extract_reasoning_returns_full_output_as_content(
    parser: IdentityReasoningParser, model_output: str
):
    # request is unused by the identity parser but kept to honor the
    # abstract base contract — pass a real ChatCompletionRequest.
    reasoning, content = parser.extract_reasoning(model_output, request=_REQUEST)
    assert reasoning is None
    assert content == model_output


# ---------------------------------------------------------------------------
# extract_reasoning_streaming
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "delta_text",
    [
        pytest.param("hello", id="single_word"),
        pytest.param(" world", id="word_with_space"),
        pytest.param("Line one\nLine two", id="multiline_delta"),
        pytest.param("x", id="single_char"),
    ],
)
def test_extract_reasoning_streaming_returns_delta_content(
    parser: IdentityReasoningParser, delta_text: str
):
    result = parser.extract_reasoning_streaming(
        previous_text="",
        current_text=delta_text,
        delta_text=delta_text,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
    )
    assert isinstance(result, DeltaMessage)
    assert result.content == delta_text
    # The identity parser never produces reasoning content.
    assert result.reasoning is None


def test_extract_reasoning_streaming_empty_delta_returns_none(
    parser: IdentityReasoningParser,
):
    result = parser.extract_reasoning_streaming(
        previous_text="some prior text",
        current_text="some prior text",
        delta_text="",
        previous_token_ids=[1, 2, 3],
        current_token_ids=[1, 2, 3],
        delta_token_ids=[],
    )
    assert result is None
