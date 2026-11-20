# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import TokenizersBackend

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.hy_v4 import HYV4Parser
from vllm.parser.parser_manager import ParserManager
from vllm.reasoning.hy_v4_reasoning_parser import HYV4ReasoningParser
from vllm.sampling_params import StructuredOutputsParams
from vllm.tool_parsers.hy_v4_tool_parser import HYV4ToolParser
from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag


def _tokenizer(suffix):
    tokenizer = Tokenizer(
        models.BPE(
            {s: i for i, s in enumerate(pre_tokenizers.ByteLevel.alphabet())}, []
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(
        [
            f"<{closing}{name}{suffix}>"
            for name in ("think", "tool_calls", "tool_call", "arg_key", "arg_value")
            for closing in ("", "/")
        ]
    )
    return TokenizersBackend(tokenizer_object=tokenizer)


@pytest.fixture(params=["", ":6124c78e"], ids=["unsuffixed", "suffixed"])
def suffix(request):
    return request.param


def _tool_call(arguments, suffix):
    pairs = "".join(
        f"<arg_key{suffix}>{key}</arg_key{suffix}>"
        f"<arg_value{suffix}>{value}</arg_value{suffix}>"
        for key, value in arguments
    )
    return f"<tool_call{suffix}>record{pairs}</tool_call{suffix}>"


def _assert_streaming_matches(
    properties, calls, expected, suffix, chunking, *, definitions=None
):
    tokenizer = _tokenizer(suffix)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "record",
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "$defs": definitions or {},
                    },
                },
            }
        ],
    )
    parser = HYV4ToolParser(tokenizer, tools=request.tools)
    output = (
        f"<tool_calls{suffix}>"
        + "".join(_tool_call(call, suffix) for call in calls)
        + f"</tool_calls{suffix}>"
    )
    non_streaming = parser.extract_tool_calls(output, request)
    assert non_streaming.tools_called
    assert [json.loads(tc.function.arguments) for tc in non_streaming.tool_calls] == (
        expected
    )
    final_json = [tc.function.arguments for tc in non_streaming.tool_calls]

    if chunking == "whole":
        chunks = [output]
    elif chunking == "tokens":
        chunks = [tokenizer.decode([tid]) for tid in tokenizer.encode(output)]
    else:
        # Guided decoding can split even the structural markers across deltas.
        request.tool_choice = "required"
        tag = get_model_structural_tag(
            "hy_v4", request.tools, "required", False, token_suffix=suffix
        )
        assert tag is not None
        request.structured_outputs = StructuredOutputsParams(
            structural_tag=tag.model_dump_json()
        )
        chunks = list(output)

    arguments = [""] * len(calls)
    names = []
    previous_text = ""
    previous_ids: list[int] = []
    for chunk in chunks:
        delta_ids = tokenizer.encode(chunk)
        current_ids = previous_ids + delta_ids
        current_text = previous_text + chunk
        delta = parser.extract_tool_calls_streaming(
            previous_text,
            current_text,
            chunk,
            previous_ids,
            current_ids,
            delta_ids,
            request,
        )
        previous_text, previous_ids = current_text, current_ids
        if delta is None:
            continue
        assert not delta.content
        for tc in delta.tool_calls or []:
            assert tc.function is not None
            if tc.function.name is not None:
                names.append((tc.index, tc.function.name))
            arguments[tc.index] += tc.function.arguments or ""
            # A client cannot retract a prefix already sent on the wire.
            assert final_json[tc.index].startswith(arguments[tc.index])

    assert names == [(i, "record") for i in range(len(calls))]
    assert arguments == final_json
    assert [json.loads(args) for args in arguments] == expected


@pytest.mark.parametrize("chunking", ["whole", "tokens", "characters"])
@pytest.mark.parametrize("key", [" count ", "\tcount\n", " \t "])
@pytest.mark.parametrize(
    ("value_type", "raw", "value"),
    [
        ("string", '  a"b\\c  ', '  a"b\\c  '),
        ("integer", "7", 7),
        ("number", "1.5", 1.5),
        ("boolean", "true", True),
        ("array", "[1, 2]", [1, 2]),
        ("object", '{"x": 1}', {"x": 1}),
        ("null", "null", None),
    ],
)
def test_whitespace_argument_name_preserves_value_type(
    suffix, chunking, key, value_type, raw, value
):
    _assert_streaming_matches(
        {key: {"type": value_type}}, [[(key, raw)]], [{key: value}], suffix, chunking
    )


@pytest.mark.parametrize("chunking", ["whole", "tokens", "characters"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("padded_type", ["integer", "string"])
def test_whitespace_distinguishes_argument_names(
    suffix, chunking, reverse, padded_type
):
    properties = {
        "count": {"type": "string" if padded_type == "integer" else "integer"},
        " count ": {"type": padded_type},
    }
    pairs = [("count", "2"), (" count ", "7")]
    values = ["2", 7] if padded_type == "integer" else [2, "7"]
    if reverse:
        pairs.reverse()
        values.reverse()
    expected = {key: value for (key, _), value in zip(pairs, values)}
    _assert_streaming_matches(properties, [pairs], [expected], suffix, chunking)


@pytest.mark.parametrize("chunking", ["whole", "tokens", "characters"])
def test_whitespace_argument_state_resets_between_calls(suffix, chunking):
    _assert_streaming_matches(
        {" count ": {"type": "integer"}, " label ": {"type": "string"}},
        [
            [(" count ", "7"), (" label ", "first")],
            [(" count ", "9")],
            [(" label ", "last")],
        ],
        [{" count ": 7, " label ": "first"}, {" count ": 9}, {" label ": "last"}],
        suffix,
        chunking,
    )


@pytest.mark.parametrize("chunking", ["whole", "tokens", "characters"])
@pytest.mark.parametrize(
    "schema,raw,expected",
    [
        ({"type": ["string", "null"]}, "null", None),
        ({"anyOf": [{"type": "string"}, {"type": "integer"}]}, "12", 12),
        ({"oneOf": [{"type": "array"}, {"type": "string"}]}, "[1,2]", [1, 2]),
        ({"type": "int64"}, "12", 12),
        ({"type": "dict"}, "{'x': 1}", {"x": 1}),
        ({}, "null", "null"),
        ({"type": "string"}, "", ""),
        ({"$ref": "#/$defs/value"}, "12", 12),
        ({"anyOf": [{"$ref": "#/$defs/value"}, {"type": "null"}]}, "null", None),
    ],
)
def test_schema_conversion_keeps_streamed_arguments_prefix_stable(
    suffix, chunking, schema, raw, expected
):
    _assert_streaming_matches(
        {"value": schema},
        [[("value", raw)]],
        [{"value": expected}],
        suffix,
        chunking,
        definitions={"value": {"type": "integer"}},
    )


@pytest.mark.parametrize("chunking", ["whole", "tokens", "characters"])
def test_duplicate_argument_keys_preserve_streamed_json(suffix, chunking):
    _assert_streaming_matches(
        {"value": {"type": "string"}},
        [[("value", "first"), ("value", "last")]],
        [{"value": "last"}],
        suffix,
        chunking,
    )


@pytest.mark.parametrize("chunk_size", [1, 7, 10000])
@pytest.mark.parametrize("delegating", [False, True], ids=["engine", "registered"])
@pytest.mark.parametrize("thinking", [False, True])
def test_reasoning_and_parallel_tools_share_streaming_boundaries(
    suffix, chunk_size, delegating, thinking
):
    """The public parser must route a batched reasoning end and two tool calls."""
    tokenizer = _tokenizer(suffix)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "record",
                    "parameters": {
                        "type": "object",
                        "properties": {"count": {"type": "integer"}},
                    },
                },
            }
        ],
    )
    kwargs = {
        "chat_template_kwargs": {"reasoning_effort": "high" if thinking else "no_think"}
    }
    parser_cls = (
        ParserManager.get_parser("hy_v4", "hy_v4", enable_auto_tools=True)
        if delegating
        else HYV4Parser
    )
    assert parser_cls is not None
    reasoning = "Let me check. \n" if thinking else ""
    content = "Checking now. \n"
    output = (
        (reasoning + f"</think{suffix}>" if thinking else "")
        + content
        + f"<tool_calls{suffix}>"
        + _tool_call([("count", "7")], suffix)
        + _tool_call([("count", "9")], suffix)
        + f"</tool_calls{suffix}>"
    )
    parser = parser_cls(tokenizer, request.tools, **kwargs)
    r, c, calls = parser.parse(
        output,
        request,
        enable_auto_tools=True,
        model_output_token_ids=tokenizer.encode(output),
    )
    assert r == (reasoning or None)
    assert c == content
    assert calls is not None
    assert [json.loads(call.arguments) for call in calls] == [
        {"count": 7},
        {"count": 9},
    ]
    parser = parser_cls(tokenizer, request.tools, **kwargs)
    accumulated_reasoning = accumulated_content = ""
    arguments = ["", ""]
    names, ids = [], []
    output_ids = tokenizer.encode(output)
    for start in range(0, len(output_ids), chunk_size):
        delta_ids = output_ids[start : start + chunk_size]
        text = tokenizer.decode(delta_ids)
        delta = parser.parse_delta(
            text,
            delta_ids,
            request,
            finished=start + chunk_size >= len(output_ids),
        )
        if delta is None:
            continue
        accumulated_reasoning += delta.reasoning or ""
        accumulated_content += delta.content or ""
        for call in delta.tool_calls or []:
            assert call.function is not None
            if call.id:
                ids.append(call.id)
            if call.function.name:
                names.append((call.index, call.function.name))
            arguments[call.index] += call.function.arguments or ""
    assert accumulated_reasoning == reasoning
    assert accumulated_content == content
    assert names == [(0, "record"), (1, "record")]
    assert len(set(ids)) == 2
    assert [json.loads(arg) for arg in arguments] == [{"count": 7}, {"count": 9}]


@pytest.mark.parametrize("explicit_start", [False, True])
def test_reasoning_gate_and_token_count_agree_with_extraction(suffix, explicit_start):
    tokenizer = _tokenizer(suffix)
    parser = HYV4ReasoningParser(tokenizer)
    request = ChatCompletionRequest(model="test-model", messages=[])
    prefix = f"<think{suffix}>" if explicit_start else ""
    reasoning_ids = tokenizer.encode(prefix + "abc")
    end_id = tokenizer.get_vocab()[f"</think{suffix}>"]
    content_ids = tokenizer.encode("OK")
    all_ids = reasoning_ids + [end_id] + content_ids
    assert not parser.is_reasoning_end(reasoning_ids)
    assert parser.extract_content_ids(reasoning_ids) == []
    assert parser.is_reasoning_end(all_ids)
    assert parser.is_reasoning_end_streaming(all_ids, [end_id])
    assert not parser.is_reasoning_end_streaming(all_ids, content_ids)
    assert parser.extract_content_ids(all_ids) == content_ids
    assert parser.extract_reasoning(tokenizer.decode(all_ids), request) == ("abc", "OK")
    assert parser.count_reasoning_tokens(all_ids) == 3
    assert parser.count_reasoning_tokens(reasoning_ids) == 3


@pytest.mark.parametrize(
    "kwargs,thinking",
    [
        ({}, True),
        ({"reasoning_effort": "no_think"}, False),
        (
            {
                "reasoning_effort": "high",
                "chat_template_kwargs": {"reasoning_effort": "no_think"},
            },
            False,
        ),
        (
            {
                "reasoning_effort": "no_think",
                "chat_template_kwargs": {"reasoning_effort": "high"},
            },
            True,
        ),
    ],
)
def test_reasoning_effort_precedence_matches_template(suffix, kwargs, thinking):
    tokenizer = _tokenizer(suffix)
    parser = HYV4ReasoningParser(tokenizer, **kwargs)
    request = ChatCompletionRequest(model="test-model", messages=[])
    output = f"abc</think{suffix}>OK"
    expected = ("abc", "OK") if thinking else (None, output)
    assert parser.extract_reasoning(output, request) == expected
    assert parser.is_reasoning_end([]) is not thinking


@pytest.mark.parametrize(
    "body",
    [
        "<tool_calls><tool_call>record</tool_call>",
        "<tool_calls><tool_call></tool_call></tool_calls>",
        "<tool_call>record<arg_key>x</arg_key><arg_value>7</tool_call>",
    ],
)
def test_malformed_tools_remain_content(body):
    tokenizer = _tokenizer("")
    request = ChatCompletionRequest(model="test-model", messages=[])
    result = HYV4ToolParser(tokenizer).extract_tool_calls(body, request)
    assert not result.tools_called
    assert result.content == body


def test_responses_function_schema_preserves_argument_types(suffix):
    request = ResponsesRequest(
        model="test-model",
        input="Record a count.",
        tools=[
            {
                "type": "function",
                "name": "record",
                "parameters": {
                    "type": "object",
                    "properties": {"count": {"type": "integer"}},
                },
            }
        ],
    )
    parser = HYV4Parser(_tokenizer(suffix), request.tools, thinking=False)
    _, _, calls = parser.parse(_tool_call([("count", "7")], suffix), request)
    assert calls is not None
    assert [(c.name, json.loads(c.arguments)) for c in calls] == [
        ("record", {"count": 7})
    ]
