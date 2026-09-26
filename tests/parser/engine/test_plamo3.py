# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PLaMo3 grammar, truncation, and serving-adapter regressions."""

from __future__ import annotations

import json

import pytest
import regex as re

from tests.parser.engine.conftest import make_mock_tokenizer
from tests.parser.engine.replay_harness import (
    DUMMY_TOOLS,
    _test_request,
    collect_output,
    replay_streaming,
    replay_with_text_holdback,
)
from vllm.parser.parser_manager import ParserManager
from vllm.parser.plamo3 import (
    BEGIN_THINK,
    BEGIN_TOOL_ARGUMENTS,
    BEGIN_TOOL_NAME,
    BEGIN_TOOL_REQUEST,
    BEGIN_TOOL_REQUESTS,
    END_THINK,
    END_TOOL_ARGUMENTS,
    END_TOOL_NAME,
    END_TOOL_REQUEST,
    END_TOOL_REQUESTS,
    EOT,
    PLAMO_MARKER_TOKENS,
    Plamo3Parser,
)


def _tool_header(name="weather"):
    return (
        BEGIN_TOOL_REQUEST
        + BEGIN_TOOL_NAME
        + name
        + END_TOOL_NAME
        + BEGIN_TOOL_ARGUMENTS
    )


def _tool_call(name, arguments):
    return _tool_header(name) + arguments + END_TOOL_ARGUMENTS + END_TOOL_REQUEST


@pytest.fixture
def mock_tokenizer():
    # Wrapper markers span several IDs; only the atomic pieces are in vocab.
    special = [*sorted(PLAMO_MARKER_TOKENS), EOT, "<|plamo:bos|>"]
    atoms = [*special, "<|plamo:constrain|>", "<|plamo:msg|>"]
    vocab = {token: 100 + i for i, token in enumerate(atoms)}
    inverse = {i: token for token, i in vocab.items()}
    pattern = re.compile("|".join(map(re.escape, atoms)) + r"|[\s\S]")
    tokenizer = make_mock_tokenizer(vocab, special_tokens=special)
    tokenizer.encode.side_effect = lambda text, **kwargs: [
        vocab[piece] if piece in vocab else 1000 + ord(piece)
        for piece in pattern.findall(text)
    ]
    tokenizer.decode.side_effect = lambda ids, **kwargs: "".join(
        inverse[i] if i in inverse else chr(i - 1000) for i in ids
    )
    return tokenizer


@pytest.fixture(params=["direct", "registered"])
def parser_cls(request):
    if request.param == "direct":
        return Plamo3Parser
    return ParserManager.get_parser(
        tool_parser_name="plamo3",
        reasoning_parser_name="plamo3",
        enable_auto_tools=True,
    )


def _stream(parser, tokenizer, output, chunk_size=1, finished_on_last=False):
    tokens = [
        (i, tokenizer.decode([i]))
        for i in tokenizer.encode(output, add_special_tokens=False)
    ]
    deltas = replay_streaming(
        parser,
        tokens,
        chunk_size=chunk_size,
        finished_on_last=finished_on_last,
        tools=DUMMY_TOOLS,
    )
    if not finished_on_last:
        deltas.append(
            parser.parse_delta("", [], _test_request(DUMMY_TOOLS), finished=True)
        )
    return deltas


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("with_markers", [False, True])
def test_reasoning_mode(
    parser_cls, mock_tokenizer, mock_request, thinking, with_markers
):
    output = (
        BEGIN_THINK + "Reasoning." + END_THINK + "Answer."
        if with_markers
        else "Answer."
    )
    if thinking:
        expected = ("Reasoning.", "Answer.") if with_markers else ("Answer.", "")
    else:
        expected = ("", output)
    kwargs = {"chat_template_kwargs": {"enable_thinking": thinking}}
    reasoning, content, calls = parser_cls(mock_tokenizer, **kwargs).parse(
        output, mock_request
    )
    assert (reasoning or "", content or "") == expected
    assert not calls
    result = collect_output(
        _stream(parser_cls(mock_tokenizer, **kwargs), mock_tokenizer, output)
    )
    assert (result.reasoning, result.content) == expected
    assert not result.tool_calls


@pytest.mark.parametrize("chunk_size", [1, 7, None])
@pytest.mark.parametrize("finished_on_last", [False, True])
@pytest.mark.parametrize(
    "closing",
    [
        "",
        END_TOOL_ARGUMENTS + END_TOOL_REQUEST,
        END_TOOL_ARGUMENTS + END_TOOL_REQUEST + END_TOOL_REQUESTS + EOT,
    ],
)
def test_complete_arguments_survive_missing_closers(
    parser_cls, mock_tokenizer, mock_request, chunk_size, finished_on_last, closing
):
    output = (
        BEGIN_THINK
        + "Reasoning. "
        + END_THINK
        + "Before tools. "
        + BEGIN_TOOL_REQUESTS
        + _tool_header()
        + '{"city":"東京"}'
        + closing
    )
    reasoning, content, calls = parser_cls(mock_tokenizer).parse(
        output, mock_request, enable_auto_tools=True
    )
    assert (reasoning, content) == ("Reasoning. ", "Before tools. ")
    assert calls and [(c.name, json.loads(c.arguments)) for c in calls] == [
        ("weather", {"city": "東京"})
    ]
    result = collect_output(
        _stream(
            parser_cls(mock_tokenizer),
            mock_tokenizer,
            output,
            chunk_size,
            finished_on_last,
        )
    )
    assert (result.reasoning, result.content) == (reasoning, content)
    assert result.tool_calls == [{"name": "weather", "arguments": '{"city":"東京"}'}]


def test_tool_name_cutoffs_do_not_leak_markers(
    parser_cls, mock_tokenizer, mock_request
):
    header = BEGIN_TOOL_REQUEST + BEGIN_TOOL_NAME + "weather" + END_TOOL_NAME
    header_ids = mock_tokenizer.encode(header, add_special_tokens=False)
    name_start = len(
        mock_tokenizer.encode(
            BEGIN_TOOL_REQUEST + BEGIN_TOOL_NAME, add_special_tokens=False
        )
    )
    for end in range(len(header_ids)):
        partial = mock_tokenizer.decode(header_ids[:end])
        output = END_THINK + "Before tools." + BEGIN_TOOL_REQUESTS + partial
        reasoning, content, calls = parser_cls(mock_tokenizer).parse(
            output, mock_request, enable_auto_tools=True
        )
        assert (reasoning, content) == (None, "Before tools."), end
        if end <= name_start:
            expected_name = ""
        else:
            name_text = mock_tokenizer.decode(header_ids[name_start:end])
            if "<|plamo:" in name_text:
                name_text = name_text[: name_text.rfind("<|plamo:")]
            expected_name = name_text[: len("weather")]
        assert [call.name for call in calls or []] == (
            [expected_name] if expected_name else []
        ), end
        result = collect_output(
            _stream(parser_cls(mock_tokenizer), mock_tokenizer, output)
        )
        assert [call["name"] for call in result.tool_calls] == (
            [expected_name] if expected_name else []
        ), end
        assert all("<" not in call["name"] for call in result.tool_calls), end
        assert result.content == "Before tools.", end


@pytest.mark.parametrize("chunk_size", [1, 7, None])
@pytest.mark.parametrize("finished_on_last", [False, True])
def test_all_argument_closer_cutoffs_do_not_leak(
    parser_cls, mock_tokenizer, mock_request, chunk_size, finished_on_last
):
    head = END_THINK + BEGIN_TOOL_REQUESTS + _tool_header()
    closer_ids = mock_tokenizer.encode(END_TOOL_ARGUMENTS, add_special_tokens=False)
    for end in range(len(closer_ids)):
        partial = mock_tokenizer.decode(closer_ids[:end])
        output = head + '{"city":"東京"}' + partial
        _, _, calls = parser_cls(mock_tokenizer).parse(
            output, mock_request, enable_auto_tools=True
        )
        assert calls and calls[0].arguments == '{"city":"東京"}', end
        result = collect_output(
            _stream(
                parser_cls(mock_tokenizer),
                mock_tokenizer,
                output,
                chunk_size,
                finished_on_last,
            )
        )
        assert result.tool_calls == [
            {"name": "weather", "arguments": '{"city":"東京"}'}
        ], end


@pytest.mark.parametrize(
    "arguments",
    [
        "{}",
        '{"name":"Alice","arguments":{"x":1},"parameters":true}',
        '{"text":"literal <|plamo:end_',
    ],
)
def test_arguments_are_not_rewritten(
    parser_cls, mock_tokenizer, mock_request, arguments
):
    output = END_THINK + BEGIN_TOOL_REQUESTS + _tool_header() + arguments
    _, _, calls = parser_cls(mock_tokenizer).parse(
        output, mock_request, enable_auto_tools=True
    )
    assert calls and calls[0].arguments == arguments
    result = collect_output(_stream(parser_cls(mock_tokenizer), mock_tokenizer, output))
    assert result.tool_calls == [{"name": "weather", "arguments": arguments}]


@pytest.mark.parametrize("truncated_second", [False, True])
def test_multiple_calls_keep_indices_ids_and_arguments_separate(
    parser_cls, mock_tokenizer, mock_request, truncated_second
):
    second = (
        BEGIN_TOOL_REQUEST + BEGIN_TOOL_NAME + "clo"
        if truncated_second
        else _tool_call("clock", '{"timezone":"Asia/Tokyo"}')
    )
    output = (
        END_THINK
        + BEGIN_TOOL_REQUESTS
        + _tool_call("weather", '{"city":"東京"}')
        + second
    )
    expected = [{"name": "weather", "arguments": '{"city":"東京"}'}]
    if truncated_second:
        expected.append({"name": "clo", "arguments": "{}"})
        stream_expected = [
            *expected[:-1],
            {"name": "clo", "arguments": ""},
        ]
    else:
        output += END_TOOL_REQUESTS
        expected.append({"name": "clock", "arguments": '{"timezone":"Asia/Tokyo"}'})
        stream_expected = expected
    _, content, calls = parser_cls(mock_tokenizer).parse(
        output, mock_request, enable_auto_tools=True
    )
    assert (
        calls
        and [{"name": c.name, "arguments": c.arguments} for c in calls] == expected
    )
    deltas = _stream(parser_cls(mock_tokenizer), mock_tokenizer, output)
    result = collect_output(deltas)
    assert result.tool_calls == stream_expected
    assert result.content == (content or "") == ""
    announced = [
        call for delta in deltas if delta for call in delta.tool_calls or [] if call.id
    ]
    assert [call.index for call in announced] == list(range(len(expected)))
    assert len({call.id for call in announced}) == len(expected)


def test_arguments_are_available_before_outer_closer(mock_tokenizer, mock_request):
    parser = Plamo3Parser(mock_tokenizer)
    output = END_THINK + BEGIN_TOOL_REQUESTS + _tool_call("weather", '{"city":"Tokyo"}')
    first = parser.parse_delta(output, [], mock_request, finished=False)
    assert collect_output([first]).tool_calls == [
        {"name": "weather", "arguments": '{"city":"Tokyo"}'}
    ]
    assert (
        parser.parse_delta(END_TOOL_REQUESTS[:10], [], mock_request, finished=False)
        is None
    )
    assert (
        parser.parse_delta(END_TOOL_REQUESTS[10:], [], mock_request, finished=True)
        is None
    )


def test_reusing_parser_drops_pending_names(mock_tokenizer, mock_request):
    parser = Plamo3Parser(mock_tokenizer)
    parser.parse(
        END_THINK
        + BEGIN_TOOL_REQUESTS
        + BEGIN_TOOL_REQUEST
        + BEGIN_TOOL_NAME
        + "partial",
        mock_request,
    )
    output = (
        END_THINK
        + BEGIN_TOOL_REQUESTS
        + _tool_call("weather", "{}")
        + END_TOOL_REQUESTS
    )
    _, _, calls = parser.parse(output, mock_request)
    assert calls and calls[0].name == "weather"


@pytest.mark.parametrize(
    "text", ["", "a < b", "literal <", "literal <|plamo:unknown", "  text\n"]
)
def test_plain_content_is_preserved(parser_cls, mock_tokenizer, mock_request, text):
    output = END_THINK + text + EOT
    _, content, calls = parser_cls(mock_tokenizer).parse(output, mock_request)
    assert (content or "") == text
    assert not calls
    result = collect_output(_stream(parser_cls(mock_tokenizer), mock_tokenizer, output))
    assert (result.content, result.tool_calls) == (text, [])


@pytest.mark.parametrize(
    "text, ended",
    [
        ("", False),
        (BEGIN_THINK + "reasoning", False),
        (END_THINK + "answer", True),
        (END_THINK + "answer" + BEGIN_THINK + "new reasoning", False),
        (END_THINK + "answer" + EOT + "assistant", False),
    ],
)
def test_reasoning_end_checks_the_latest_turn(mock_tokenizer, text, ended):
    parser = Plamo3Parser(mock_tokenizer)
    ids = mock_tokenizer.encode(text)
    assert parser.is_reasoning_end(ids) == ended
    if ended:
        assert mock_tokenizer.decode(parser.extract_content_ids(ids)) == "answer"


@pytest.mark.parametrize("suffix", ["<|plamo:end_", "<|plamo:end_think"])
def test_partial_end_think_is_consistent_between_streaming_and_non_streaming(
    parser_cls, mock_tokenizer, mock_request, suffix
):
    output = "visible" + suffix
    reasoning, content, calls = parser_cls(mock_tokenizer).parse(output, mock_request)
    assert (reasoning, content) == ("visible", None)
    assert not calls
    result = collect_output(_stream(parser_cls(mock_tokenizer), mock_tokenizer, output))
    assert (result.reasoning, result.content, result.tool_calls) == ("visible", "", [])


def test_tool_block_can_end_reasoning_implicitly(
    parser_cls, mock_tokenizer, mock_request
):
    output = (
        "reasoning"
        + BEGIN_TOOL_REQUESTS
        + _tool_call("weather", "{}")
        + END_TOOL_REQUESTS
    )
    reasoning, content, calls = parser_cls(mock_tokenizer).parse(
        output, mock_request, enable_auto_tools=True
    )
    assert (reasoning, content) == ("reasoning", None)
    assert calls and calls[0].name == "weather"
    result = collect_output(_stream(parser_cls(mock_tokenizer), mock_tokenizer, output))
    assert (result.reasoning, result.content) == ("reasoning", "")
    assert result.tool_calls == [{"name": "weather", "arguments": "{}"}]


@pytest.mark.parametrize(
    "choice",
    [
        "auto",
        "required",
        "none",
        {"type": "function", "function": {"name": "weather"}},
    ],
)
def test_tool_choice_uses_the_plamo_format(parser_cls, mock_tokenizer, choice):
    request = _test_request(
        [
            {
                "type": "function",
                "function": {"name": "weather", "parameters": {"type": "object"}},
            },
        ]
    )
    request = type(request).model_validate(
        request.model_dump() | {"tool_choice": choice}
    )
    output = (
        END_THINK
        + "Before tools."
        + BEGIN_TOOL_REQUESTS
        + _tool_call("weather", "{}")
        + END_TOOL_REQUESTS
    )
    _, content, calls = parser_cls(mock_tokenizer).parse(
        output, request, enable_auto_tools=True
    )
    assert content == "Before tools."
    assert [c.name for c in calls or []] == ([] if choice == "none" else ["weather"])
    parser = parser_cls(mock_tokenizer)
    delta = parser.parse_delta(output, [], request, finished=True)
    result = collect_output([delta])
    assert result.content == "Before tools."
    assert result.tool_calls == (
        [] if choice == "none" else [{"name": "weather", "arguments": "{}"}]
    )


@pytest.mark.parametrize("delay", [1, 3])
def test_delayed_text_keeps_split_markers_intact(parser_cls, mock_tokenizer, delay):
    output = (
        BEGIN_THINK
        + "reasoning"
        + END_THINK
        + "content"
        + BEGIN_TOOL_REQUESTS
        + _tool_call("weather", "{}")
        + END_TOOL_REQUESTS
        + EOT
    )
    tokens = [(i, mock_tokenizer.decode([i])) for i in mock_tokenizer.encode(output)]
    result = collect_output(
        replay_with_text_holdback(
            parser_cls(mock_tokenizer), tokens, text_delay=delay, tools=DUMMY_TOOLS
        )
    )
    assert (result.reasoning, result.content) == ("reasoning", "content")
    assert result.tool_calls == [{"name": "weather", "arguments": "{}"}]
