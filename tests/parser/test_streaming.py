# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.parser.abstract_parser import _WrappedParser
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser


class ThinkReasoningParser(BaseThinkingReasoningParser):
    @property
    def start_token(self) -> str:
        return "<think>"

    @property
    def end_token(self) -> str:
        return "</think>"


MODEL_OUTPUT = (
    "<think>let me think about this</think>"
    '<tool_call>\n{"name": "get_weather", '
    '"arguments": {"city": "Dallas"}}\n</tool_call>'
)


@pytest.fixture(scope="module")
def tokenizer():
    from vllm.tokenizers import get_tokenizer

    return get_tokenizer("Qwen/Qwen3-32B")


@pytest.fixture
def request_obj():
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
    )


def make_parser(tokenizer, reasoning=False, tool=False):
    _WrappedParser.reasoning_parser_cls = ThinkReasoningParser if reasoning else None
    _WrappedParser.tool_parser_cls = Hermes2ProToolParser if tool else None
    return _WrappedParser(tokenizer)


def stream_text(parser, tokenizer, text, request, prompt_token_ids=None):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    results: list[DeltaMessage | None] = []
    for tid in token_ids:
        delta_text = tokenizer.decode([tid])
        result = parser.parse_delta(
            delta_text, [tid], request, prompt_token_ids=prompt_token_ids
        )
        prompt_token_ids = None
        results.append(result)
    return results


def stream_text_chunked(
    parser, tokenizer, text, request, chunk_size, prompt_token_ids=None
):
    """Stream text in multi-token chunks (simulates speculative decoding)."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    results: list[DeltaMessage | None] = []
    for i in range(0, len(token_ids), chunk_size):
        chunk_ids = token_ids[i : i + chunk_size]
        delta_text = tokenizer.decode(chunk_ids)
        result = parser.parse_delta(
            delta_text, chunk_ids, request, prompt_token_ids=prompt_token_ids
        )
        prompt_token_ids = None
        results.append(result)
    return results


def collect_fields(results):
    all_reasoning = "".join(r.reasoning for r in results if r and r.reasoning)
    all_content = "".join(r.content for r in results if r and r.content)
    all_tool_calls = [tc for r in results if r and r.tool_calls for tc in r.tool_calls]
    return all_reasoning, all_content, all_tool_calls


def test_parse_delta_neither_parser(tokenizer, request_obj):
    parser = make_parser(tokenizer, reasoning=False, tool=False)
    results = stream_text(
        parser, tokenizer, MODEL_OUTPUT, request_obj, prompt_token_ids=[]
    )
    reasoning, content, tool_calls = collect_fields(results)

    assert reasoning == ""
    assert len(tool_calls) == 0
    assert "<think>" in content
    assert "let me think about this" in content
    assert "<tool_call>" in content
    assert "get_weather" in content


def test_parse_delta_tool_parser_only(tokenizer, request_obj):
    parser = make_parser(tokenizer, reasoning=False, tool=True)
    results = stream_text(
        parser, tokenizer, MODEL_OUTPUT, request_obj, prompt_token_ids=[]
    )
    reasoning, content, tool_calls = collect_fields(results)

    assert reasoning == ""
    assert "<think>" in content
    assert "let me think about this" in content
    assert "</think>" in content

    assert len(tool_calls) > 0
    assert tool_calls[0].function.name == "get_weather"
    tool_args = "".join(
        tc.function.arguments for tc in tool_calls if tc.function.arguments
    )
    assert json.loads(tool_args) == {"city": "Dallas"}


def test_parse_delta_reasoning_parser_only(tokenizer, request_obj):
    parser = make_parser(tokenizer, reasoning=True, tool=False)
    results = stream_text(
        parser, tokenizer, MODEL_OUTPUT, request_obj, prompt_token_ids=[]
    )
    reasoning, content, tool_calls = collect_fields(results)

    assert "let me think about this" in reasoning
    assert len(tool_calls) == 0
    assert "<tool_call>" in content
    assert "get_weather" in content
    assert "</tool_call>" in content


def test_parse_delta_both_parsers(tokenizer, request_obj):
    parser = make_parser(tokenizer, reasoning=True, tool=True)
    results = stream_text(
        parser, tokenizer, MODEL_OUTPUT, request_obj, prompt_token_ids=[]
    )
    reasoning, content, tool_calls = collect_fields(results)

    assert "let me think about this" in reasoning
    assert content == ""

    assert len(tool_calls) > 0
    assert tool_calls[0].function.name == "get_weather"
    tool_args = "".join(
        tc.function.arguments for tc in tool_calls if tc.function.arguments
    )
    assert json.loads(tool_args) == {"city": "Dallas"}


def test_parse_delta_reasoning_only_thinking_disabled(tokenizer, request_obj):
    """Regression test for vllm-project/vllm#40466.

    When enable_thinking=False, the chat template places <think>\\n\\n</think>
    in the prompt. The model then generates pure content (no think tokens).
    All streaming output must go to delta.content, not delta.reasoning.
    """
    parser = make_parser(tokenizer, reasoning=True, tool=False)

    end_token_id = parser._reasoning_parser.end_token_id
    prompt_token_ids = [1, 2, end_token_id, 3]

    content_text = "Hello! How can I assist you today?"
    results = stream_text(
        parser,
        tokenizer,
        content_text,
        request_obj,
        prompt_token_ids=prompt_token_ids,
    )
    reasoning, content, tool_calls = collect_fields(results)

    assert reasoning == "", f"Expected no reasoning, got: {reasoning!r}"
    assert "Hello" in content
    assert "assist" in content
    assert len(tool_calls) == 0


@pytest.mark.parametrize("chunk_size", [3, 4, 5, 6])
def test_parse_delta_spec_decode_boundary_preserves_reasoning(
    tokenizer, request_obj, chunk_size
):
    """Regression test for vllm-project/vllm#42781.

    When speculative decoding (MTP) accepts a multi-token batch that
    spans the reasoning/content boundary, reasoning content in the
    boundary delta must not be lost when the tool parser processes the
    content portion.
    """
    parser = make_parser(tokenizer, reasoning=True, tool=True)

    # Single-token streaming (baseline): full reasoning preserved
    results_single = stream_text(
        parser, tokenizer, MODEL_OUTPUT, request_obj, prompt_token_ids=[]
    )
    reasoning_single, _, tools_single = collect_fields(results_single)

    # Multi-token chunked streaming (simulates spec decode batches)
    parser_chunked = make_parser(tokenizer, reasoning=True, tool=True)
    results_chunked = stream_text_chunked(
        parser_chunked,
        tokenizer,
        MODEL_OUTPUT,
        request_obj,
        chunk_size=chunk_size,
        prompt_token_ids=[],
    )
    reasoning_chunked, _, tools_chunked = collect_fields(results_chunked)

    assert "let me think about this" in reasoning_single
    assert "let me think about this" in reasoning_chunked, (
        f"Reasoning lost with chunk_size={chunk_size}: got {reasoning_chunked!r}"
    )
    assert len(tools_chunked) > 0
    assert tools_chunked[0].function.name == "get_weather"
