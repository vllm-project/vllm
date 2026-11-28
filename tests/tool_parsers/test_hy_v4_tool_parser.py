# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from unittest.mock import MagicMock, patch

import pytest
import regex as re

import vllm.envs as envs
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.sampling_params import StructuredOutputsParams
from vllm.tool_parsers.hy_v4_tool_parser import HYV4ToolExtractor, HYV4ToolParser
from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag


class _StubTokenizer:
    def __init__(self, suffix):
        self.vocab = {
            f"<{closing}{name}{suffix}>": i
            for i, (name, closing) in enumerate(
                (name, closing)
                for name in ("tool_calls", "tool_call", "arg_key", "arg_value")
                for closing in ("", "/")
            )
        }
        self.pattern = "|".join(map(re.escape, self.vocab)) + r"|[\s\S]"

    def get_vocab(self):
        return self.vocab

    def pieces(self, text):
        return re.findall(self.pattern, text)

    def encode(self, text):
        return [
            self.vocab[piece] if piece in self.vocab else 256 + ord(piece)
            for piece in self.pieces(text)
        ]


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


def _assert_streaming_matches(properties, calls, expected, suffix, chunking):
    tokenizer = _StubTokenizer(suffix)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "record",
                    "parameters": {"type": "object", "properties": properties},
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
        chunks = tokenizer.pieces(output)
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
    assert parser.prev_tool_call_arr == [
        {"name": "record", "arguments": args} for args in expected
    ]
    assert parser.streamed_args_for_tool == final_json


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


def _hy_v4_extractor() -> HYV4ToolExtractor:
    vocab = {
        "<tool_calls>": 1,
        "</tool_calls>": 2,
        "<tool_call>": 3,
        "</tool_call>": 4,
        "<arg_key>": 5,
        "</arg_key>": 6,
        "<arg_value>": 7,
        "</arg_value>": 8,
    }
    return HYV4ToolExtractor(vocab, token_suffix="", strict=False)


def test_regex_timeout_treated_as_no_tool_call():
    extractor = _hy_v4_extractor()
    model_output = "<tool_calls><tool_call>" + "c" * 100 + "</tool_calls>"
    mock_regex = MagicMock()
    mock_regex.findall.side_effect = TimeoutError("Regex timeout")

    with patch.object(extractor, "tool_call_regex", mock_regex):
        result = extractor.extract_tool_calls(model_output, None)

    assert result["tools_called"] is False
    assert result["tool_calls"] == []
    assert result["content"] == model_output
    mock_regex.findall.assert_called_once()
    assert (
        mock_regex.findall.call_args.kwargs["timeout"]
        == envs.VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS
    )


def test_streaming_args_regex_timeout_skips_later_deltas():
    extractor = _hy_v4_extractor()
    extractor._streaming_tool_name = "get_weather"
    extractor.current_tool_id = 0
    extractor._buffer = "<arg_key>" + "c" * 100
    mock_regex = MagicMock()
    mock_regex.findall.side_effect = TimeoutError("Regex timeout")

    with patch.object(extractor, "func_args_regex", mock_regex):
        first = extractor._extract_streaming_incremental(False, None, None)
        second = extractor.extract_tool_calls_streaming(
            "prev",
            "prevx",
            "x",
            [1],
            [1, 3],
            [3],
            None,
        )

    assert first is None
    assert second is None
    mock_regex.findall.assert_called_once()
    assert (
        mock_regex.findall.call_args.kwargs["timeout"]
        == envs.VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS
    )
