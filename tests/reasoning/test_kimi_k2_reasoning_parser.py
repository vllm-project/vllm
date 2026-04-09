# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.identity_reasoning_parser import IdentityReasoningParser
from vllm.reasoning.kimi_k2_reasoning_parser import KimiK2ReasoningParser
from vllm.tokenizers import get_tokenizer

REASONING_MODEL_NAME = "moonshotai/Kimi-K2.5"


@pytest.fixture(scope="module")
def kimi_k2_tokenizer():
    return get_tokenizer(tokenizer_name=REASONING_MODEL_NAME, trust_remote_code=True)


def test_parser_selection_thinking_enabled(kimi_k2_tokenizer):
    parser = KimiK2ReasoningParser(
        kimi_k2_tokenizer, chat_template_kwargs={"thinking": True}
    )
    assert parser._identity_parser is None


def test_parser_selection_thinking_disabled(kimi_k2_tokenizer):
    parser = KimiK2ReasoningParser(
        kimi_k2_tokenizer, chat_template_kwargs={"thinking": False}
    )
    assert isinstance(parser._identity_parser, IdentityReasoningParser)


def test_extract_reasoning_with_think_tags(kimi_k2_tokenizer):
    parser = KimiK2ReasoningParser(kimi_k2_tokenizer)
    request = ChatCompletionRequest(model="test-model", messages=[], temperature=1.0)

    reasoning, content = parser.extract_reasoning(
        "<think>step by step reasoning</think>final answer", request
    )
    assert reasoning == "step by step reasoning"
    assert content == "final answer"


def test_extract_reasoning_empty_thinking(kimi_k2_tokenizer):
    parser = KimiK2ReasoningParser(kimi_k2_tokenizer)
    request = ChatCompletionRequest(model="test-model", messages=[], temperature=1.0)

    reasoning, content = parser.extract_reasoning(
        "<think></think>final answer", request
    )
    assert reasoning == ""
    assert content == "final answer"


def test_extract_reasoning_implicit_start(kimi_k2_tokenizer):
    """When there's no <think> tag, everything is treated as reasoning."""
    parser = KimiK2ReasoningParser(kimi_k2_tokenizer)
    request = ChatCompletionRequest(model="test-model", messages=[], temperature=1.0)

    reasoning, content = parser.extract_reasoning(
        "implicit reasoning with no tags", request
    )
    assert reasoning == "implicit reasoning with no tags"
    assert content is None


def test_extract_reasoning_tool_section_ends_reasoning(kimi_k2_tokenizer):
    """<|tool_calls_section_begin|> implicitly ends reasoning."""
    parser = KimiK2ReasoningParser(kimi_k2_tokenizer)
    request = ChatCompletionRequest(model="test-model", messages=[], temperature=1.0)

    text = "some reasoning<|tool_calls_section_begin|>tool call data"
    reasoning, content = parser.extract_reasoning(text, request)
    assert reasoning == "some reasoning"
    assert content == "<|tool_calls_section_begin|>tool call data"


def test_streaming_reasoning_then_content(kimi_k2_tokenizer):
    """Token-by-token streaming: reasoning tokens then content after </think>."""
    parser = KimiK2ReasoningParser(kimi_k2_tokenizer)

    think_id = parser._start_token_id
    end_think_id = parser._end_token_id
    # Use a real token ID from the tokenizer for regular content
    regular_id = kimi_k2_tokenizer.encode("hello", add_special_tokens=False)[0]

    # First token: <think> — single special token should be skipped
    result = parser.extract_reasoning_streaming(
        previous_text="",
        current_text="<think>",
        delta_text="<think>",
        previous_token_ids=[],
        current_token_ids=[think_id],
        delta_token_ids=[think_id],
    )
    assert result is None

    # Reasoning token
    result = parser.extract_reasoning_streaming(
        previous_text="<think>",
        current_text="<think>step one",
        delta_text="step one",
        previous_token_ids=[think_id],
        current_token_ids=[think_id, regular_id],
        delta_token_ids=[regular_id],
    )
    assert isinstance(result, DeltaMessage)
    assert result.reasoning == "step one"
    assert result.content is None

    # End token </think> as single token — should be skipped
    result = parser.extract_reasoning_streaming(
        previous_text="<think>step one",
        current_text="<think>step one</think>",
        delta_text="</think>",
        previous_token_ids=[think_id, regular_id],
        current_token_ids=[think_id, regular_id, end_think_id],
        delta_token_ids=[end_think_id],
    )
    assert result is None

    # Content after </think>
    content_id = kimi_k2_tokenizer.encode("world", add_special_tokens=False)[0]
    result = parser.extract_reasoning_streaming(
        previous_text="<think>step one</think>",
        current_text="<think>step one</think>answer",
        delta_text="answer",
        previous_token_ids=[think_id, regular_id, end_think_id],
        current_token_ids=[think_id, regular_id, end_think_id, content_id],
        delta_token_ids=[content_id],
    )
    assert isinstance(result, DeltaMessage)
    assert result.content == "answer"


def test_streaming_tool_section_ends_reasoning(kimi_k2_tokenizer):
    """<|tool_calls_section_begin|> in delta ends reasoning during streaming."""
    parser = KimiK2ReasoningParser(kimi_k2_tokenizer)

    think_id = parser._start_token_id
    tool_begin_id = parser._tool_section_start_token_id
    regular_id = kimi_k2_tokenizer.encode("hello", add_special_tokens=False)[0]

    # Tool section token arrives — should transition from reasoning to content
    result = parser.extract_reasoning_streaming(
        previous_text="<think>thinking",
        current_text="<think>thinking<|tool_calls_section_begin|>",
        delta_text="<|tool_calls_section_begin|>",
        previous_token_ids=[think_id, regular_id],
        current_token_ids=[think_id, regular_id, tool_begin_id],
        delta_token_ids=[tool_begin_id],
    )
    assert isinstance(result, DeltaMessage)
    assert result.content == "<|tool_calls_section_begin|>"
