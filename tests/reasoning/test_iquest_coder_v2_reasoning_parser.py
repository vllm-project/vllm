# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoTokenizer

from tests.reasoning.utils import run_reasoning_extraction
from vllm.reasoning.iquest_coder_v2_reasoning_parser import (
    IquestCoderV2ReasoningParser,
)

# The v2 reasoning parser only requires that its special tokens exist in the
# vocab. We reuse a small tokenizer and inject the iQuest Coder V2 tokens.
MODEL = "Qwen/Qwen3-0.6B"

START_TOKEN = "<think>"
END_TOKEN = "</think>"
ASSISTANT_TOKEN = "<|iquestcoder_assistant|>"


@pytest.fixture(scope="module")
def iquest_v2_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    required = [START_TOKEN, END_TOKEN, ASSISTANT_TOKEN]
    existing = set(tokenizer.get_vocab().keys())
    missing = [token for token in required if token not in existing]
    if missing:
        tokenizer.add_tokens(missing)
    return tokenizer


def _parser(tokenizer, **chat_template_kwargs):
    kwargs = {}
    if chat_template_kwargs:
        kwargs["chat_template_kwargs"] = chat_template_kwargs
    return IquestCoderV2ReasoningParser(tokenizer, **kwargs)


def test_missing_tokens_raise():
    plain = AutoTokenizer.from_pretrained(MODEL)
    if ASSISTANT_TOKEN not in plain.get_vocab():
        with pytest.raises(ValueError, match="missing required"):
            IquestCoderV2ReasoningParser(plain)


@pytest.mark.parametrize(
    "chat_template_kwargs, expected",
    [
        ({}, True),
        ({"enable_thinking": True}, True),
        ({"enable_thinking": False}, False),
        ({"thinking": True}, True),
        ({"thinking": False}, False),
    ],
)
def test_thinking_enabled_flag(iquest_v2_tokenizer, chat_template_kwargs, expected):
    parser = _parser(iquest_v2_tokenizer, **chat_template_kwargs)
    assert parser._thinking_enabled is expected


# <think> in the prompt, only </think> emitted (the typical case for this
# template, which places <think>\n at the end of the prompt).
WITHOUT_START_TOKEN = {
    "output": "reasoning here</think>the answer",
    "reasoning": "reasoning here",
    "content": "the answer",
}
# Full <think>...</think> present in the output (edge case / old template).
WITH_THINK = {
    "output": "<think>reasoning here</think>the answer",
    "reasoning": "reasoning here",
    "content": "the answer",
}
COMPLETE_REASONING = {
    "output": "<think>reasoning here</think>",
    "reasoning": "reasoning here",
    "content": None,
}
MULTILINE_REASONING = {
    "output": "<think>line one\nline two</think>body\nmore",
    "reasoning": "line one\nline two",
    "content": "body\nmore",
}
# No closing tag yet: everything so far is reasoning, no content.
ONLY_OPEN_TAG = {
    "output": "<think>still reasoning",
    "reasoning": "still reasoning",
    "content": None,
}

TEXT_CASES = [
    pytest.param(False, WITHOUT_START_TOKEN, id="without_start_token"),
    pytest.param(True, WITHOUT_START_TOKEN, id="without_start_token_stream"),
    pytest.param(False, WITH_THINK, id="with_think"),
    pytest.param(True, WITH_THINK, id="with_think_stream"),
    pytest.param(False, COMPLETE_REASONING, id="complete_reasoning"),
    pytest.param(True, COMPLETE_REASONING, id="complete_reasoning_stream"),
    pytest.param(False, MULTILINE_REASONING, id="multiline_reasoning"),
    pytest.param(True, MULTILINE_REASONING, id="multiline_reasoning_stream"),
    pytest.param(False, ONLY_OPEN_TAG, id="only_open_tag"),
    pytest.param(True, ONLY_OPEN_TAG, id="only_open_tag_stream"),
]


@pytest.mark.parametrize("streaming, param_dict", TEXT_CASES)
def test_reasoning_extraction(streaming, param_dict, iquest_v2_tokenizer):
    output = iquest_v2_tokenizer.tokenize(param_dict["output"])
    output_tokens = [
        iquest_v2_tokenizer.convert_tokens_to_string([token]) for token in output
    ]
    parser = _parser(iquest_v2_tokenizer)

    reasoning, content = run_reasoning_extraction(
        parser, output_tokens, streaming=streaming
    )

    assert reasoning == param_dict["reasoning"]
    assert content == param_dict["content"]


def test_thinking_disabled_returns_all_content(iquest_v2_tokenizer):
    """With thinking disabled, output is treated as content verbatim."""
    parser = _parser(iquest_v2_tokenizer, enable_thinking=False)
    reasoning, content = parser.extract_reasoning(
        "<think>should be ignored</think>hello",
        request=None,
    )
    assert reasoning is None
    assert content == "<think>should be ignored</think>hello"


def test_is_reasoning_end_within_current_turn(iquest_v2_tokenizer):
    parser = _parser(iquest_v2_tokenizer)
    start = parser._start_token_id
    end = parser._end_token_id
    assistant = parser._assistant_token_id

    # </think> after <think> in this turn -> reasoning ended.
    assert parser.is_reasoning_end([start, 10, end, 11]) is True
    # Still inside <think> -> not ended.
    assert parser.is_reasoning_end([start, 10, 11]) is False
    # New assistant turn opened after a prior </think>: the closest boundary
    # scanning backwards is the assistant token, so reasoning has NOT ended.
    assert parser.is_reasoning_end([end, 10, assistant, 11]) is False
    # No think tokens at all -> not ended.
    assert parser.is_reasoning_end([10, 11, 12]) is False


def test_is_reasoning_end_when_thinking_disabled(iquest_v2_tokenizer):
    parser = _parser(iquest_v2_tokenizer, enable_thinking=False)
    # Always "ended" so generated tokens are content from the start.
    assert parser.is_reasoning_end([parser._start_token_id, 10]) is True


def test_extract_content_ids(iquest_v2_tokenizer):
    parser = _parser(iquest_v2_tokenizer)
    end = parser._end_token_id

    # Content is everything after the last </think>.
    assert parser.extract_content_ids([10, end, 20, 21]) == [20, 21]
    # </think> at the very end -> no content.
    assert parser.extract_content_ids([10, end]) == []
    # Open <think> with no close -> no content extracted.
    assert parser.extract_content_ids([parser._start_token_id, 10]) == []


def test_count_reasoning_tokens(iquest_v2_tokenizer):
    parser = _parser(iquest_v2_tokenizer)
    start = parser._start_token_id
    end = parser._end_token_id

    # 3 reasoning tokens between <think> and </think>; 2 content tokens after.
    tokens = [start, 1, 2, 3, end, 4, 5]
    assert parser.count_reasoning_tokens(tokens) == 3

    # Reasoning starts implicitly (template puts <think> in prompt): tokens
    # before the first </think> are reasoning.
    assert parser.count_reasoning_tokens([1, 2, end, 3]) == 2


def test_count_reasoning_tokens_thinking_disabled(iquest_v2_tokenizer):
    parser = _parser(iquest_v2_tokenizer, enable_thinking=False)
    assert parser.count_reasoning_tokens([1, 2, 3]) == 0
