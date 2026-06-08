# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.parser.abstract_parser import DelegatingParser
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


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


@pytest.fixture
def request_obj():
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        tools=TOOLS,
        tool_choice="auto",
    )


def make_parser(tokenizer, reasoning=False, tool=False):
    class TestParser(DelegatingParser):
        reasoning_parser_cls = ThinkReasoningParser if reasoning else None
        tool_parser_cls = Hermes2ProToolParser if tool else None

    return TestParser(tokenizer)


def stream_text(parser, tokenizer, text, request, prompt_token_ids=None):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    results: list[DeltaMessage | None] = []
    for tid in token_ids:
        delta_text = tokenizer.decode([tid])
        result = parser.parse_delta(
            delta_text,
            [tid],
            request,
            prompt_token_ids=prompt_token_ids,
            finished=False,
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


def stream_chunks(parser, tokenizer, chunks, request_obj):
    """Stream pre-split token-ID chunks through the parser."""
    results: list[DeltaMessage | None] = []
    prompt_token_ids: list[int] | None = []
    for chunk in chunks:
        delta_text = tokenizer.decode(chunk)
        result = parser.parse_delta(
            delta_text,
            chunk,
            request_obj,
            prompt_token_ids=prompt_token_ids,
            finished=False,
        )
        prompt_token_ids = None
        results.append(result)
    return results


def _boundary_chunks(tokenizer, parser):
    """Split MODEL_OUTPUT into 3 chunks that straddle the </think> boundary."""
    token_ids = tokenizer.encode(MODEL_OUTPUT, add_special_tokens=False)
    end_token_id = parser._reasoning_parser.end_token_id
    end_idx = token_ids.index(end_token_id)
    return [
        token_ids[: end_idx - 1],
        token_ids[end_idx - 1 : end_idx + 2],
        token_ids[end_idx + 2 :],
    ]


def test_parse_delta_reasoning_not_dropped_on_boundary(tokenizer, request_obj):
    """Regression: reasoning must not be lost when a multi-token delta
    spans the reasoning/tool-call boundary."""
    parser = make_parser(tokenizer, reasoning=True, tool=True)
    chunks = _boundary_chunks(tokenizer, parser)
    results = stream_chunks(parser, tokenizer, chunks, request_obj)
    reasoning, content, tool_calls = collect_fields(results)

    assert "think about this" in reasoning
    assert content == ""
    assert len(tool_calls) > 0
    assert tool_calls[0].function.name == "get_weather"
    tool_args = "".join(
        tc.function.arguments for tc in tool_calls if tc.function.arguments
    )
    assert json.loads(tool_args) == {"city": "Dallas"}


def test_parse_delta_reasoning_boundary_no_tool_parser(tokenizer, request_obj):
    """When no tool parser is active, boundary-spanning chunks must still
    preserve reasoning and pass post-</think> text as content."""
    parser = make_parser(tokenizer, reasoning=True, tool=False)
    chunks = _boundary_chunks(tokenizer, parser)
    results = stream_chunks(parser, tokenizer, chunks, request_obj)
    reasoning, content, tool_calls = collect_fields(results)

    assert "think about this" in reasoning
    assert len(tool_calls) == 0
    assert "<tool_call>" in content
    assert "get_weather" in content


def test_parse_delta_reasoning_only_no_think_leak(tokenizer, request_obj):
    """Regression: </think> must not leak into content when streaming
    token-by-token with reasoning=True, tool=False."""
    parser = make_parser(tokenizer, reasoning=True, tool=False)
    results = stream_text(
        parser, tokenizer, MODEL_OUTPUT, request_obj, prompt_token_ids=[]
    )
    reasoning, content, tool_calls = collect_fields(results)

    assert "let me think about this" in reasoning
    assert "</think>" not in content
    assert "<think>" not in content


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


def test_parse_delta_finished_no_flush_without_tool_call_delta(tokenizer, request_obj):
    """When finished=True but the final parse_delta produces no
    tool-call delta, unstreamed args are not flushed."""
    parser = make_parser(tokenizer, reasoning=False, tool=True)

    results = stream_text(
        parser, tokenizer, MODEL_OUTPUT, request_obj, prompt_token_ids=[]
    )
    _, _, tool_calls = collect_fields(results)
    assert len(tool_calls) > 0

    streamed = parser._tool_parser.streamed_args_for_tool[0]
    assert len(streamed) > 5
    parser._tool_parser.streamed_args_for_tool[0] = streamed[:-5]

    # Prevent normal extraction from catching the gap — without a
    # tool-call delta to merge into, the flush is skipped.
    parser._tool_parser.extract_tool_calls_streaming = lambda *a, **kw: None

    flush_result = parser.parse_delta("", [], request_obj, finished=True)
    assert flush_result is None or flush_result.tool_calls is None


def test_parse_delta_finished_no_extra_args_when_fully_streamed(tokenizer, request_obj):
    """When all args have been streamed, finished=True must not
    produce extra or duplicate arguments."""
    parser = make_parser(tokenizer, reasoning=False, tool=True)
    results = stream_text(
        parser, tokenizer, MODEL_OUTPUT, request_obj, prompt_token_ids=[]
    )
    _, _, tool_calls = collect_fields(results)

    assert len(tool_calls) > 0
    assert tool_calls[0].function.name == "get_weather"
    tool_args = "".join(
        tc.function.arguments for tc in tool_calls if tc.function.arguments
    )
    assert json.loads(tool_args) == {"city": "Dallas"}

    flush_result = parser.parse_delta("", [], request_obj, finished=True)
    assert flush_result is None or flush_result.tool_calls is None


def test_parse_delta_finished_appends_remaining_args(tokenizer, request_obj):
    """When finished=True and the tool parser has unstreamed args,
    parse_delta appends the remaining arguments to the tool-call delta."""
    parser = make_parser(tokenizer, reasoning=False, tool=True)
    token_ids = tokenizer.encode(MODEL_OUTPUT, add_special_tokens=False)

    remainder = ',"unit":"celsius"}'
    prompt_ids: list[int] | None = []
    results: list[DeltaMessage | None] = []
    for i, tid in enumerate(token_ids):
        prev = results[-1] if results else None
        prev_had_args = (
            prev
            and prev.tool_calls
            and any(tc.function and tc.function.arguments for tc in prev.tool_calls)
        )

        if prev_had_args:
            parser._tool_parser.get_remaining_unstreamed_args = lambda: remainder

        result = parser.parse_delta(
            tokenizer.decode([tid]),
            [tid],
            request_obj,
            prompt_token_ids=prompt_ids,
            finished=prev_had_args,
        )
        prompt_ids = None
        results.append(result)

        if prev_had_args:
            break

    _, _, tool_calls = collect_fields(results)
    tool_args = "".join(
        tc.function.arguments for tc in tool_calls if tc.function.arguments
    )
    assert tool_args.endswith(remainder)


def test_parse_delta_tool_choice_none(tokenizer, request_obj):
    parser = make_parser(tokenizer, reasoning=False, tool=True)
    request = request_obj.model_copy(update={"tool_choice": "none"})
    results = stream_text(parser, tokenizer, MODEL_OUTPUT, request, prompt_token_ids=[])
    reasoning, content, tool_calls = collect_fields(results)

    assert reasoning == ""
    assert len(tool_calls) == 0
    assert "<tool_call>" in content
    assert "get_weather" in content


def test_parse_delta_tool_choice_none_with_reasoning(tokenizer, request_obj):
    parser = make_parser(tokenizer, reasoning=True, tool=True)
    request = request_obj.model_copy(update={"tool_choice": "none"})
    results = stream_text(parser, tokenizer, MODEL_OUTPUT, request, prompt_token_ids=[])
    reasoning, content, tool_calls = collect_fields(results)

    assert "let me think about this" in reasoning
    assert len(tool_calls) == 0
    assert "<tool_call>" in content
    assert "get_weather" in content
