# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the engine-based MuseGlimmer reasoning parser.

MuseGlimmer output is a sequence of channel-scoped messages
(``to=self`` reasoning / ``to=user`` answer / ``to=<tool>`` ATEM tool call).
The engine drives the reasoning side; the tool channel is forwarded intact
(framing included) to the legacy ATEM tool parser, so the handoff contract
is exercised here against ``MuseGlimmerToolParser`` directly and through
``DelegatingParser.parse_delta`` — the mixed configuration serving uses.
"""

import json
from unittest.mock import MagicMock

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from tests.parser.engine.streaming_helpers import simulate_reasoning_streaming
from vllm.parser.abstract_parser import DelegatingParser, StreamState
from vllm.parser.engine.registered_adapters import MuseGlimmerParserReasoningAdapter
from vllm.tool_parsers.muse_glimmer_tool_parser import MuseGlimmerToolParser

TURN_START = "<|start|>"
MESSAGE = "<|message|>"
EOM = "<|eom|>"
EOT = "<|eot|>"

_VOCAB = {
    TURN_START: 200001,
    MESSAGE: 200002,
    EOM: 200003,
    EOT: 200004,
}

# Framing that must never surface in reasoning or content.
_FRAMING = [TURN_START, MESSAGE, EOM, EOT, "to=self", "to=user", "<atem:"]

RAW_TOOLCALL = (
    " to=self<|message|>I should read the hostname file.<|eom|>"
    "<|start|>assistant to=read.read<|message|>"
    '<atem:function_calls>\n<atem:invoke name="read.read">\n'
    '<atem:parameter name="path">/etc/hostname</atem:parameter>\n'
    "</atem:invoke>\n</atem:function_calls>"
)

RAW_ANSWER = (
    " to=self<|message|>Think about it.<|eom|>"
    "<|start|>assistant to=user<|message|>The answer is 42.<|eot|>"
)

# No closing <|eom|>: truncated CoT quoting ATEM markup.
RAW_TRUNCATED = (
    " to=self<|message|>Maybe I should call "
    '<atem:function_calls>\n<atem:invoke name="read.read">\n'
    '<atem:parameter name="path">/etc/hostname</atem:parameter>\n'
    "</atem:invoke>\n</atem:function_calls> but wait"
)


@pytest.fixture
def mock_tokenizer():
    return make_mock_tokenizer(_VOCAB)


@pytest.fixture
def reasoning_parser(mock_tokenizer):
    return MuseGlimmerParserReasoningAdapter(mock_tokenizer)


def _tool_parser():
    return MuseGlimmerToolParser(object())


def _assert_no_framing(text, where):
    for marker in _FRAMING:
        assert marker not in text, f"framing {marker!r} leaked into {where}: {text!r}"


class TestNonStreaming:
    def test_reasoning_then_answer(self, reasoning_parser):
        reasoning, content = reasoning_parser.extract_reasoning(RAW_ANSWER, None)
        assert reasoning == "Think about it."
        assert content == "The answer is 42."

    def test_content_only(self, reasoning_parser):
        reasoning, content = reasoning_parser.extract_reasoning(
            " to=user<|message|>Just a direct answer.<|eot|>", None
        )
        assert reasoning is None
        assert content == "Just a direct answer."

    def test_unframed_output_passes_through(self, reasoning_parser):
        reasoning, content = reasoning_parser.extract_reasoning(
            "Just a direct answer.", None
        )
        assert reasoning is None
        assert content == "Just a direct answer."

    def test_bare_header_is_untagged_content(self, reasoning_parser):
        reasoning, content = reasoning_parser.extract_reasoning(
            " <|message|>public text<|eot|>", None
        )
        assert reasoning is None
        assert content == "public text"

    def test_multiple_reasoning_blocks_concatenate(self, reasoning_parser):
        reasoning, content = reasoning_parser.extract_reasoning(
            " to=self<|message|>A<|eom|>"
            "<|start|>assistant to=self<|message|>B<|eom|>"
            "<|start|>assistant to=user<|message|>C<|eot|>",
            None,
        )
        assert reasoning == "AB"
        assert content == "C"

    def test_tool_handoff_preserves_channel_framing(self, reasoning_parser):
        """The forwarded tool channel must start at its header — the ATEM
        tool parser needs the recipient for channel scoping."""
        reasoning, content = reasoning_parser.extract_reasoning(RAW_TOOLCALL, None)
        assert reasoning == "I should read the hostname file."
        assert content is not None
        assert content.startswith("to=read.read<|message|>"), repr(content)
        out = _tool_parser().extract_tool_calls(content, None)
        assert out.tools_called and len(out.tool_calls) == 1
        assert out.tool_calls[0].function.name == "read.read"
        assert json.loads(out.tool_calls[0].function.arguments) == {
            "path": "/etc/hostname"
        }

    def test_parallel_calls_forwarded(self, reasoning_parser):
        raw = (
            " to=self<|message|>need two calls<|eom|>"
            "<|start|>assistant to=math.add<|message|>"
            '<atem:function_calls>\n<atem:invoke name="math.add">\n'
            '<atem:parameter name="a">1</atem:parameter>\n</atem:invoke>\n'
            "</atem:invoke>\n</atem:function_calls><|eom|>"
            "<|start|>assistant to=math.mul<|message|>"
            '<atem:function_calls>\n<atem:invoke name="math.mul">\n'
            '<atem:parameter name="a">3</atem:parameter>\n</atem:invoke>\n'
            "</atem:function_calls><|eot|>"
        )
        reasoning, content = reasoning_parser.extract_reasoning(raw, None)
        assert reasoning == "need two calls"
        out = _tool_parser().extract_tool_calls(content, None)
        assert [t.function.name for t in out.tool_calls] == ["math.add", "math.mul"]

    def test_truncated_cot_keeps_echoed_atem_as_reasoning(self, reasoning_parser):
        reasoning, content = reasoning_parser.extract_reasoning(RAW_TRUNCATED, None)
        assert reasoning is not None and "Maybe I should call" in reasoning
        assert content is None

    def test_prose_to_in_reasoning_is_not_a_channel_switch(self, reasoning_parser):
        """``to=`` needs a complete ``to=<name><|message|>`` header to switch
        channels; prose mentioning ``to=`` stays reasoning."""
        reasoning, content = reasoning_parser.extract_reasoning(
            " to=self<|message|>set flag to=5 and send it to=alice<|eom|>"
            "<|start|>assistant to=user<|message|>done<|eot|>",
            None,
        )
        assert reasoning == "set flag to=5 and send it to=alice"
        assert content == "done"

    def test_bare_tool_header_without_eom_routes_tool_call(self, reasoning_parser):
        """Known model defect: the analysis channel is left without <|eom|>,
        writing a bare tool header instead. The tool call must survive."""
        raw = (
            " to=self<|message|>calling now to=weather.get<|message|>"
            '<atem:function_calls>\n<atem:invoke name="weather.get">\n'
            '<atem:parameter name="city">Paris</atem:parameter>\n'
            "</atem:invoke>\n</atem:function_calls>"
        )
        reasoning, content = reasoning_parser.extract_reasoning(raw, None)
        assert content is not None
        assert content.startswith("to=weather.get<|message|>"), repr(content)
        out = _tool_parser().extract_tool_calls(content, None)
        assert out.tools_called
        assert out.tool_calls[0].function.name == "weather.get"


class TestStreaming:
    @pytest.mark.parametrize("chunk_size", [1, 3, 17])
    def test_chunk_invariant_reasoning_then_answer(self, mock_tokenizer, chunk_size):
        parser = MuseGlimmerParserReasoningAdapter(mock_tokenizer)
        chunks = [
            RAW_ANSWER[i : i + chunk_size]
            for i in range(0, len(RAW_ANSWER), chunk_size)
        ]
        reasoning, content = simulate_reasoning_streaming(parser, chunks)
        assert reasoning == "Think about it."
        assert content == "The answer is 42."
        _assert_no_framing(reasoning, "reasoning")
        _assert_no_framing(content, "content")
        assert not parser.has_engine_confirmed_reasoning_end()

    @pytest.mark.parametrize("chunk_size", [1, 3, 17])
    def test_chunk_invariant_tool_handoff(self, mock_tokenizer, chunk_size):
        parser = MuseGlimmerParserReasoningAdapter(mock_tokenizer)
        chunks = [
            RAW_TOOLCALL[i : i + chunk_size]
            for i in range(0, len(RAW_TOOLCALL), chunk_size)
        ]
        reasoning, content = simulate_reasoning_streaming(parser, chunks)
        assert reasoning == "I should read the hostname file."
        _assert_no_framing(reasoning, "reasoning")
        # The forwarded tool channel (framing included) is the handoff to the
        # tool parser, not client-visible content.
        assert content.startswith("to=read.read<|message|>"), repr(content)
        assert parser.has_engine_confirmed_reasoning_end()

    def test_truncated_stream_stays_in_reasoning(self, reasoning_parser):
        chunks = [RAW_TRUNCATED[i : i + 3] for i in range(0, len(RAW_TRUNCATED), 3)]
        reasoning, content = simulate_reasoning_streaming(reasoning_parser, chunks)
        assert "Maybe I should call" in reasoning
        assert content == ""
        assert not reasoning_parser.has_engine_confirmed_reasoning_end()

    def test_prose_to_streams_as_reasoning(self, reasoning_parser):
        raw = " to=self<|message|>send it to=alice then stop<|eom|>"
        chunks = list(raw)
        reasoning, content = simulate_reasoning_streaming(reasoning_parser, chunks)
        assert reasoning == "send it to=alice then stop"
        assert content == ""
        assert not reasoning_parser.has_engine_confirmed_reasoning_end()


class _MuseGlimmerDelegatingParser(DelegatingParser):
    reasoning_parser_cls = MuseGlimmerParserReasoningAdapter
    tool_parser_cls = MuseGlimmerToolParser


class _Req:
    tools = None
    tool_choice = None
    include_reasoning = True


def _tokenize(text):
    """Markers are atomic special tokens; other text is one token per char,
    matching the mock tokenizer's chr()-based decode."""
    markers = sorted(_VOCAB, key=len, reverse=True)
    tokens = []
    i = 0
    while i < len(text):
        for marker in markers:
            if text.startswith(marker, i):
                tokens.append((_VOCAB[marker], marker))
                i += len(marker)
                break
        else:
            tokens.append((ord(text[i]), text[i]))
            i += 1
    return tokens


def _drive(mock_tokenizer, gen_text, req=None):
    """Feed gen_text token-by-token through parse_delta, as serving does."""
    parser = _MuseGlimmerDelegatingParser(mock_tokenizer)
    parser._stream_state = StreamState()
    if req is None:
        req = _Req()
    tokens = _tokenize(gen_text)
    prompt_ids = [900001, 900002]
    reasoning, content, tools = [], [], []
    for i, (tid, text) in enumerate(tokens):
        dm = parser.parse_delta(
            text,
            [tid],
            req,
            prompt_token_ids=prompt_ids if i == 0 else None,
            finished=(i == len(tokens) - 1),
        )
        if dm is None:
            continue
        if getattr(dm, "reasoning", None):
            reasoning.append(dm.reasoning)
        if getattr(dm, "content", None):
            content.append(dm.content)
        for tc in getattr(dm, "tool_calls", None) or []:
            fn = tc.function
            name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
            args = (
                fn.get("arguments")
                if isinstance(fn, dict)
                else getattr(fn, "arguments", None)
            )
            tools.append((tc.index, name, args))
    return "".join(reasoning), "".join(content), tools


class TestParseDelta:
    """The mixed configuration serving builds for PR-stage MuseGlimmer:
    engine reasoning adapter + legacy ATEM tool parser."""

    def test_reasoning_then_answer(self, mock_tokenizer):
        reasoning, content, tools = _drive(mock_tokenizer, RAW_ANSWER)
        assert reasoning == "Think about it."
        assert content == "The answer is 42."
        assert tools == []
        _assert_no_framing(content, "content")

    def test_content_only(self, mock_tokenizer):
        reasoning, content, tools = _drive(
            mock_tokenizer, " to=user<|message|>Just a direct answer.<|eot|>"
        )
        assert content == "Just a direct answer."
        assert tools == []

    def test_tool_call(self, mock_tokenizer):
        reasoning, content, tools = _drive(mock_tokenizer, RAW_TOOLCALL)
        assert reasoning == "I should read the hostname file."
        _assert_no_framing(content, "content")
        assert len(tools) == 1, tools
        idx, name, args = tools[0]
        assert idx == 0 and name == "read.read"
        assert json.loads(args) == {"path": "/etc/hostname"}

    def test_truncated_cot_no_toolcall(self, mock_tokenizer):
        reasoning, content, tools = _drive(mock_tokenizer, RAW_TRUNCATED)
        assert tools == [], f"contemplated invoke leaked as tool call: {tools}"
        _assert_no_framing(content, "content")
        assert "Maybe I should call" in reasoning

    def test_reasoning_suppressed_when_not_requested(self, mock_tokenizer):
        class _NoReasonReq(_Req):
            include_reasoning = False

        reasoning, content, tools = _drive(
            mock_tokenizer, RAW_ANSWER, req=_NoReasonReq()
        )
        assert reasoning == ""
        assert content == "The answer is 42."


class TestReasoningTokenCounts:
    def test_streaming_counts_only_reasoning_body(self, mock_tokenizer):
        parser = MuseGlimmerParserReasoningAdapter(mock_tokenizer)
        tokens = _tokenize(RAW_ANSWER)
        prev_text = ""
        prev_ids: list[int] = []
        all_ids: list[int] = []
        for tid, text in tokens:
            cur_text = prev_text + text
            cur_ids = prev_ids + [tid]
            parser.extract_reasoning_streaming(
                prev_text, cur_text, text, tuple(prev_ids), tuple(cur_ids), (tid,)
            )
            prev_text, prev_ids = cur_text, cur_ids
            all_ids.append(tid)
        # One token per reasoning-body char; headers and framing excluded.
        assert parser.count_reasoning_tokens(all_ids) == len("Think about it.")


class TestAdjustRequest:
    def test_special_tokens_preserved(self, reasoning_parser):
        req = MagicMock()
        req.skip_special_tokens = True
        reasoning_parser.adjust_request(req)
        assert req.skip_special_tokens is False
