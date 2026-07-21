# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.parser.abstract_parser import DelegatingParser
from vllm.parser.engine.registered_adapters import Qwen3ParserReasoningAdapter
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


KIMI_K2_MODEL_CONFIG = SimpleNamespace(
    hf_text_config=SimpleNamespace(model_type="kimi_k2"),
    hf_overrides=None,
)

HISTORY_MESSAGES = [
    {"role": "user", "content": "first"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "functions.get_current_weather:0",
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "arguments": "{}",
                },
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "functions.get_current_weather:0",
        "content": "{}",
    },
    {"role": "user", "content": "again"},
]


@pytest.fixture
def request_obj():
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        tools=TOOLS,
        tool_choice="auto",
    )


def make_parser(tokenizer, reasoning=False, tool=False, **kwargs):
    class TestParser(DelegatingParser):
        reasoning_parser_cls = ThinkReasoningParser if reasoning else None
        tool_parser_cls = Hermes2ProToolParser if tool else None

    return TestParser(tokenizer, **kwargs)


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


def _boundary_chunks(tokenizer, parser, end_token_id=None):
    """Split MODEL_OUTPUT into 3 chunks that straddle the </think> boundary."""
    token_ids = tokenizer.encode(MODEL_OUTPUT, add_special_tokens=False)
    if end_token_id is None:
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


def test_parse_delta_required_tool_choice_kimi_k2_ids(tokenizer, request_obj):
    parser = make_parser(
        tokenizer, reasoning=False, tool=True, model_config=KIMI_K2_MODEL_CONFIG
    )
    request = request_obj.model_copy(update={"tool_choice": "required"})
    output = json.dumps(
        [
            {
                "name": "get_current_weather",
                "parameters": {"city": "Dallas"},
            }
        ]
    )

    results: list[DeltaMessage | None] = []
    prompt_token_ids: list[int] | None = []
    for i in range(0, len(output), 3):
        chunk = output[i : i + 3]
        results.append(
            parser.parse_delta(
                chunk,
                [],
                request,
                prompt_token_ids=prompt_token_ids,
                finished=False,
            )
        )
        prompt_token_ids = None

    _, content, tool_calls = collect_fields(results)
    assert content == ""
    assert any(tc.id == "functions.get_current_weather:0" for tc in tool_calls)
    assert all(tc.id in (None, "functions.get_current_weather:0") for tc in tool_calls)


def test_parse_delta_required_tool_choice_kimi_k2_ids_after_history(
    tokenizer, request_obj
):
    parser = make_parser(
        tokenizer, reasoning=False, tool=True, model_config=KIMI_K2_MODEL_CONFIG
    )
    request = request_obj.model_copy(
        update={"messages": HISTORY_MESSAGES, "tool_choice": "required"}
    )
    output = json.dumps(
        [
            {
                "name": "get_current_weather",
                "parameters": {"city": "Dallas"},
            }
        ]
    )

    results: list[DeltaMessage | None] = []
    prompt_token_ids: list[int] | None = []
    for i in range(0, len(output), 3):
        chunk = output[i : i + 3]
        results.append(
            parser.parse_delta(
                chunk,
                [],
                request,
                prompt_token_ids=prompt_token_ids,
                finished=False,
            )
        )
        prompt_token_ids = None

    _, _, tool_calls = collect_fields(results)
    assert any(tc.id == "functions.get_current_weather:1" for tc in tool_calls)
    assert all(tc.id in (None, "functions.get_current_weather:1") for tc in tool_calls)


# ── Engine-based reasoning + non-engine tool parser (Qwen3 + Hermes) ──


class Qwen3ReasoningHermesToolParser(DelegatingParser):
    reasoning_parser_cls = Qwen3ParserReasoningAdapter
    tool_parser_cls = Hermes2ProToolParser


def test_engine_reasoning_hermes_tool_token_by_token(tokenizer, request_obj):
    """Qwen3 engine reasoning + Hermes tool parser, token-by-token.

    Sanity check that the mixed engine/non-engine configuration works
    when tokens arrive one at a time (no deferred content)."""
    parser = Qwen3ReasoningHermesToolParser(tokenizer)

    assert parser._reasoning_parser.engine_based_streaming is True
    assert parser._tool_parser.engine_based_streaming is False
    assert parser._engine_based is False

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


def test_engine_reasoning_hermes_tool_boundary(tokenizer, request_obj):
    """Qwen3 engine reasoning + Hermes tool parser, boundary chunks.

    When </think> and <tool_call> are in the same chunk with aligned
    text and token IDs, the engine processes both terminals and returns
    the <tool_call> text as content."""
    parser = Qwen3ReasoningHermesToolParser(tokenizer)
    end_token_id = parser._reasoning_parser._parser_engine._reasoning_end_token_id
    chunks = _boundary_chunks(tokenizer, parser, end_token_id=end_token_id)
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
    assert "tool_call" not in content


def test_engine_reasoning_hermes_tool_text_holdback(tokenizer, request_obj):
    """Qwen3 engine reasoning + Hermes tool parser with engine holdback.

    Simulates stream_interval > 1 where a batched delta contains
    '</think><'.  The '<' is a regular character token — not the
    <tool_call> special token — so the engine's incremental lexer
    buffers it (it could be the start of a text terminal like
    <tool_call>).  The buffered '<' is only recoverable via
    finish_streaming().

    Without the fix, finish_streaming() is never called at the
    reasoning->tool transition when _engine_based is False, so the '<'
    is lost and the Hermes parser sees 'tool_call>...' instead of
    '<tool_call>...'."""
    parser = Qwen3ReasoningHermesToolParser(tokenizer)
    vocab = tokenizer.get_vocab()
    think_end_id = vocab["</think>"]
    lt_id = vocab["<"]

    token_ids = tokenizer.encode(MODEL_OUTPUT, add_special_tokens=False)
    end_idx = token_ids.index(think_end_id)

    # Reasoning tokens (aligned text + IDs)
    pre_ids = token_ids[:end_idx]
    pre_text = tokenizer.decode(pre_ids)

    # Batched delta: '</think><' — the engine recognises </think> as
    # THINK_END but the trailing '<' is consumed by the engine's lexer
    # and held back (it could be the start of <tool_call>).  The '<'
    # is NOT in delta_message.content; it is only in the engine's
    # internal buffer, recoverable via finish_streaming().
    holdback_ids = [think_end_id, lt_id]
    holdback_text = "</think><"

    # Remaining text: 'tool_call>\n{...}\n</tool_call>' — the model
    # generated <tool_call> as character tokens (not the special token),
    # and the '<' was consumed above.  Encode separately to get the
    # correct token IDs for this substring.
    rest_text = (
        'tool_call>\n{"name": "get_weather", '
        '"arguments": {"city": "Dallas"}}\n</tool_call>'
    )
    rest_ids = tokenizer.encode(rest_text, add_special_tokens=False)

    results: list[DeltaMessage | None] = []
    results.append(
        parser.parse_delta(
            pre_text,
            pre_ids,
            request_obj,
            prompt_token_ids=[],
            finished=False,
        )
    )
    results.append(
        parser.parse_delta(
            holdback_text,
            holdback_ids,
            request_obj,
            finished=False,
        )
    )
    results.append(
        parser.parse_delta(
            rest_text,
            rest_ids,
            request_obj,
            finished=False,
        )
    )

    reasoning, content, tool_calls = collect_fields(results)

    assert "let me think about this" in reasoning
    assert len(tool_calls) > 0, (
        "Tool calls lost at engine-reasoning -> tool transition. "
        "finish_streaming() not called when _engine_based is False."
    )
    assert tool_calls[0].function.name == "get_weather"
    tool_args = "".join(
        tc.function.arguments for tc in tool_calls if tc.function.arguments
    )
    assert json.loads(tool_args) == {"city": "Dallas"}
    assert "tool_call" not in content


# ── Engine-based reasoning WITHOUT a tool parser (Qwen3 only) ──


class Qwen3ReasoningNoToolParser(DelegatingParser):
    reasoning_parser_cls = Qwen3ParserReasoningAdapter
    tool_parser_cls = None


def test_engine_reasoning_no_tool_batched_content_passthrough(tokenizer, request_obj):
    """Qwen3 engine reasoning with NO tool parser, batched boundary.

    The three mixed-parser tests above all pair the engine reasoning
    parser with Hermes; none exercise the engine-reasoning-only path
    through the hoisted finish_streaming() transition (where
    ``_engine_based`` is True and there is no tool parser).  A single
    batched delta carries ``</think>`` plus the following content
    (as happens with stream_interval > 1).  The post-``</think>`` text
    must be emitted as content -- not dropped, not reclassified as
    reasoning -- and the ``</think>`` marker must not leak either way."""
    parser = Qwen3ReasoningNoToolParser(tokenizer)
    assert parser._reasoning_parser.engine_based_streaming is True
    assert parser._tool_parser is None

    model_output = "<think>let me think about this</think>The answer is 42."
    end_token_id = parser._reasoning_parser._parser_engine._reasoning_end_token_id
    token_ids = tokenizer.encode(model_output, add_special_tokens=False)
    end_idx = token_ids.index(end_token_id)
    chunks = [token_ids[:end_idx], token_ids[end_idx:]]

    results = stream_chunks(parser, tokenizer, chunks, request_obj)
    reasoning, content, tool_calls = collect_fields(results)

    assert "let me think about this" in reasoning
    assert content == "The answer is 42."
    assert "</think>" not in content
    assert "</think>" not in reasoning
    assert len(tool_calls) == 0


def _decode_stream_deltas(tokenizer, groups):
    """Decode token-ID groups into ``(delta_text, group)`` pairs via the real
    incremental ``DecodeStream``.

    This mirrors how vLLM's detokenizer feeds ``parse_delta`` in
    production: byte-level UTF-8 hold-back means a character whose bytes
    span multiple tokens is only surfaced once complete (a naive
    per-token ``decode`` would instead emit U+FFFD replacement chars)."""
    from tokenizers.decoders import DecodeStream

    stream = DecodeStream(skip_special_tokens=False)
    inner = tokenizer._tokenizer
    pairs = []
    for group in groups:
        text = ""
        for token_id in group:
            piece = stream.step(inner, token_id)
            if piece:
                text += piece
        pairs.append((text, group))
    return pairs


def test_engine_reasoning_hermes_tool_multibyte_holdback(tokenizer, request_obj):
    """Multi-token character across the reasoning->tool boundary.

    Extends the ASCII '<' hold-back guard with bbrowning's multi-token
    *character* concern.  Two things must both hold:

    1. The batched ``</think><`` delta relies on the hoisted
       finish_streaming() to recover the engine-buffered '<'.  Without
       the fix the Hermes parser never sees ``<tool_call>`` and emits no
       tool call at all.
    2. The tool-call arguments carry ``東京🧑\u200d🚀``; the astronaut
       ZWJ sequence's bytes span multiple Qwen3 tokens, so the rest of
       the stream is fed one token at a time through the real
       ``DecodeStream``.  Its UTF-8 hold-back yields the correct
       character round-trip (a naive per-token decode would corrupt it),
       verifying the boundary stays byte-safe for multi-token
       characters."""
    parser = Qwen3ReasoningHermesToolParser(tokenizer)
    vocab = tokenizer.get_vocab()
    think_end_id = vocab["</think>"]
    lt_id = vocab["<"]

    city = "東京🧑\u200d🚀"
    # Faithfulness precondition: the emoji really is a multi-token char.
    assert len(tokenizer.encode("🧑\u200d🚀", add_special_tokens=False)) > 1

    model_output = (
        "<think>let me think about this</think>"
        '<tool_call>\n{"name": "get_weather", '
        f'"arguments": {{"city": "{city}"}}}}\n</tool_call>'
    )
    token_ids = tokenizer.encode(model_output, add_special_tokens=False)
    end_idx = token_ids.index(think_end_id)
    pre_ids = token_ids[:end_idx]

    rest_text = (
        'tool_call>\n{"name": "get_weather", '
        f'"arguments": {{"city": "{city}"}}}}\n</tool_call>'
    )
    rest_ids = tokenizer.encode(rest_text, add_special_tokens=False)

    # Deltas: reasoning, then a batched '</think><' (the engine buffers
    # the '<'), then the remaining tokens one at a time so the
    # multi-token character is genuinely split across deltas by the
    # detokenizer.
    groups = [pre_ids, [think_end_id, lt_id]] + [[tid] for tid in rest_ids]
    pairs = _decode_stream_deltas(tokenizer, groups)

    results: list[DeltaMessage | None] = []
    prompt_token_ids: list[int] | None = []
    for delta_text, group in pairs:
        results.append(
            parser.parse_delta(
                delta_text,
                group,
                request_obj,
                prompt_token_ids=prompt_token_ids,
                finished=False,
            )
        )
        prompt_token_ids = None

    reasoning, content, tool_calls = collect_fields(results)

    assert "let me think about this" in reasoning
    assert len(tool_calls) > 0, (
        "Tool call lost at engine-reasoning -> tool transition; the "
        "buffered '<' was not recovered by finish_streaming()."
    )
    assert tool_calls[0].function.name == "get_weather"
    tool_args = "".join(
        tc.function.arguments for tc in tool_calls if tc.function.arguments
    )
    assert json.loads(tool_args) == {"city": city}
    assert content == ""
