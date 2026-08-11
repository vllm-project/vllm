# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the MuseGlimmer ATEM tool parser and reasoning parser.

MuseGlimmer writes every turn as a sequence of channel-scoped messages rather
than JSON, so the two parsers are tested together: the reasoning parser strips
the reasoning span and forwards the remaining channels as content, and the tool
parser reads ATEM markup out of those channels.

Four areas, in order:

  1. non-streaming tool-call extraction, including channel scoping (an
     ``<atem:invoke>`` echoed inside reasoning must never become a call);
  2. the reasoning -> tool-parser handoff, which regressed once by returning
     ``content=None`` and starving the tool parser;
  3. streaming, where markers routinely straddle chunk boundaries, plus
     truncation isolation for an unterminated ``to=self`` block;
  4. tool-name normalization against the tools registered on the request.

These drive the parsers directly and need no checkpoint. The tests that require
a real tokenizer live in ``test_muse_glimmer_parse_delta.py``.
"""

import json
from types import SimpleNamespace

from vllm.reasoning.muse_glimmer_reasoning_parser import MuseGlimmerReasoningParser
from vllm.tool_parsers.muse_glimmer_tool_parser import MuseGlimmerToolParser

# __init__ needs a tokenizer; every method under test is reachable without one.
R = MuseGlimmerReasoningParser.__new__(MuseGlimmerReasoningParser)
T = MuseGlimmerToolParser.__new__(MuseGlimmerToolParser)

# Any framing token that must NEVER appear in surfaced reasoning/content.
_FRAMING = [
    "<|start|>",
    "<|message|>",
    "<|eom|>",
    "<|eot|>",
    "to=self",
    "to=user",
    "to=read.read",
    "<atem:",
]


class _FakeReq:
    """Minimal ChatCompletionRequest stand-in (no registered tools)."""

    tools = None


def _req(*names):
    """A request with ``names`` registered as tools."""
    return SimpleNamespace(
        tools=[SimpleNamespace(function=SimpleNamespace(name=n)) for n in names]
    )


def _call(name):
    """One tool-call turn invoking ``name``."""
    return (
        f"<|start|>assistant to={name}<|message|>"
        f'<atem:function_calls>\n<atem:invoke name="{name}">\n'
        f'<atem:parameter name="city">Paris</atem:parameter>\n'
        f"</atem:invoke>\n</atem:function_calls>"
    )


# ---------------------------------------------------------------- tool calls


def test_single_tool_call_after_reasoning():
    raw = (
        "to=self<|message|>Let me check the weather.<|eom|>"
        "<|start|>assistant to=weather.get<|message|>"
        '<atem:function_calls>\n<atem:invoke name="weather.get">\n'
        '<atem:parameter name="city">Paris</atem:parameter>\n'
        '<atem:parameter name="units">celsius</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls><|eot|>"
    )
    out = MuseGlimmerToolParser.extract_tool_calls(T, raw, None)
    assert out.tools_called and len(out.tool_calls) == 1
    assert out.tool_calls[0].function.name == "weather.get"
    assert json.loads(out.tool_calls[0].function.arguments) == {
        "city": "Paris",
        "units": "celsius",
    }


def test_parallel_calls_across_eom_boundaries():
    raw = (
        "<|start|>assistant to=math.add<|message|>"
        '<atem:function_calls>\n<atem:invoke name="math.add">\n'
        '<atem:parameter name="a">1</atem:parameter>\n'
        '<atem:parameter name="b">2</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls><|eom|>"
        "<|start|>assistant to=math.mul<|message|>"
        '<atem:function_calls>\n<atem:invoke name="math.mul">\n'
        '<atem:parameter name="a">3</atem:parameter>\n'
        '<atem:parameter name="b">4</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls><|eot|>"
    )
    out = MuseGlimmerToolParser.extract_tool_calls(T, raw, None)
    assert out.tools_called and len(out.tool_calls) == 2, len(out.tool_calls)
    assert [t.function.name for t in out.tool_calls] == ["math.add", "math.mul"]
    # JSON-typed values decode to ints
    assert json.loads(out.tool_calls[0].function.arguments) == {"a": 1, "b": 2}


def test_echoed_invoke_in_reasoning_is_not_parsed():
    """Channel scoping: an invoke quoted inside reasoning is not a call."""
    raw = (
        'to=self<|message|>I could call <atem:invoke name="evil.fn">'
        '<atem:parameter name="x">1</atem:parameter></atem:invoke> '
        "but I will not.<|eom|>"
        "<|start|>assistant to=user<|message|>The answer is 42.<|eot|>"
    )
    out = MuseGlimmerToolParser.extract_tool_calls(T, raw, None)
    assert not out.tools_called, "channel scoping failed -- echoed invoke parsed!"
    assert out.content == "The answer is 42.", repr(out.content)


def test_plain_answer_yields_no_tool_calls():
    out = MuseGlimmerToolParser.extract_tool_calls(
        T, "to=user<|message|>Just a plain answer.<|eot|>", None
    )
    assert not out.tools_called


def test_json_object_array_and_bool_params_decode():
    raw = (
        "<|start|>assistant to=api.call<|message|>"
        '<atem:function_calls>\n<atem:invoke name="api.call">\n'
        '<atem:parameter name="payload">{"nested": [1, 2, 3]}</atem:parameter>\n'
        '<atem:parameter name="flag">true</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls><|eot|>"
    )
    out = MuseGlimmerToolParser.extract_tool_calls(T, raw, None)
    assert json.loads(out.tool_calls[0].function.arguments) == {
        "payload": {"nested": [1, 2, 3]},
        "flag": True,
    }


# ------------------------------------------------- reasoning -> tool handoff


def test_reasoning_to_toolcall_handoff():
    """The regression: content=None here starved the tool parser."""
    raw = (
        " to=self<|message|>Let me call the tool.<|eom|>"
        "<|start|>assistant to=weather.get<|message|>"
        '<atem:function_calls>\n<atem:invoke name="weather.get">\n'
        '<atem:parameter name="city">Paris</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls>"
    )
    reasoning, content = MuseGlimmerReasoningParser.extract_reasoning(R, raw, None)
    assert reasoning == "Let me call the tool.", repr(reasoning)
    assert content is not None and "<atem:invoke" in content, repr(content)
    out = MuseGlimmerToolParser.extract_tool_calls(T, content, None)
    assert out.tools_called and len(out.tool_calls) == 1
    assert out.tool_calls[0].function.name == "weather.get"


def test_reasoning_then_user_answer():
    raw = (
        " to=self<|message|>thinking<|eom|>"
        "<|start|>assistant to=user<|message|>The answer is 42.<|eot|>"
    )
    reasoning, content = MuseGlimmerReasoningParser.extract_reasoning(R, raw, None)
    assert reasoning == "thinking", repr(reasoning)
    assert content == "The answer is 42.", repr(content)
    assert not MuseGlimmerToolParser.extract_tool_calls(T, content, None).tools_called


def test_plain_content_without_framing_passes_through():
    reasoning, content = MuseGlimmerReasoningParser.extract_reasoning(
        R, "Just a direct answer.", None
    )
    assert reasoning is None and content == "Just a direct answer.", (
        reasoning,
        content,
    )


def test_reasoning_then_parallel_calls():
    raw = (
        " to=self<|message|>need two calls<|eom|>"
        "<|start|>assistant to=math.add<|message|>"
        '<atem:function_calls>\n<atem:invoke name="math.add">\n'
        '<atem:parameter name="a">1</atem:parameter>\n</atem:invoke>\n'
        "</atem:function_calls><|eom|>"
        "<|start|>assistant to=math.mul<|message|>"
        '<atem:function_calls>\n<atem:invoke name="math.mul">\n'
        '<atem:parameter name="a">3</atem:parameter>\n</atem:invoke>\n'
        "</atem:function_calls><|eot|>"
    )
    reasoning, content = MuseGlimmerReasoningParser.extract_reasoning(R, raw, None)
    assert reasoning == "need two calls", repr(reasoning)
    out = MuseGlimmerToolParser.extract_tool_calls(T, content, None)
    assert [t.function.name for t in out.tool_calls] == ["math.add", "math.mul"], (
        out.tool_calls
    )


# ----------------------------------------------------------------- streaming


def _stream(raw: str, chunk: int):
    """Feed ``raw`` incrementally in ``chunk``-char steps through BOTH streaming
    parsers; return (reasoning, content, tool_calls)."""
    reasoning, content, toolcalls = [], [], []
    prev = ""
    i = 0
    while i < len(raw):
        cur = raw[: i + chunk]
        delta = cur[len(prev) :]
        dm = MuseGlimmerReasoningParser.extract_reasoning_streaming(
            R, prev, cur, delta, [], [], []
        )
        if dm is not None:
            if getattr(dm, "reasoning", None):
                reasoning.append(dm.reasoning)
            if getattr(dm, "content", None):
                content.append(dm.content)
        dt = MuseGlimmerToolParser.extract_tool_calls_streaming(
            T, prev, cur, delta, [], [], [], _FakeReq()
        )
        if dt is not None and dt.tool_calls:
            toolcalls.extend(dt.tool_calls)
        prev = cur
        i += chunk
    return "".join(reasoning), "".join(content), toolcalls


def _fn_of(tc):
    fn = tc.function
    if isinstance(fn, dict):
        return fn.get("name"), fn.get("arguments")
    return fn.name, fn.arguments


RAW_TOOLCALL = (
    " to=self<|message|>I should read the hostname file to answer.<|eom|>"
    "<|start|>assistant to=read.read<|message|>"
    '<atem:function_calls>\n<atem:invoke name="read.read">\n'
    '<atem:parameter name="path">/etc/hostname</atem:parameter>\n'
    "</atem:invoke>\n</atem:function_calls>"
)

RAW_ANSWER = (
    " to=self<|message|>Think about it.<|eom|>"
    "<|start|>assistant to=user<|message|>The answer is 42.<|eot|>"
)

# NO closing <|eom|> -> truncated CoT
RAW_TRUNCATED = (
    " to=self<|message|>Maybe I should call "
    '<atem:function_calls>\n<atem:invoke name="read.read">\n'
    '<atem:parameter name="path">/etc/hostname</atem:parameter>\n'
    "</atem:invoke>\n</atem:function_calls> but wait"
)


def _check_toolcall_stream(chunk):
    reasoning, content, tcs = _stream(RAW_TOOLCALL, chunk)
    # (a) no framing token leaks into reasoning or content
    for f in _FRAMING:
        assert f not in reasoning, (
            f"framing {f!r} leaked into reasoning (chunk={chunk})"
        )
        assert f not in content, f"framing {f!r} leaked into content (chunk={chunk})"
    # (b) exactly one tool_call with correct name + args
    assert len(tcs) == 1, f"expected 1 tool_call, got {len(tcs)} (chunk={chunk})"
    name, args = _fn_of(tcs[0])
    assert name == "read.read", name
    assert json.loads(args) == {"path": "/etc/hostname"}, args
    assert tcs[0].index == 0 and tcs[0].type == "function" and tcs[0].id
    # (c) reasoning captured separately and clean
    assert reasoning == "I should read the hostname file to answer.", repr(reasoning)
    assert content == "", repr(content)


def test_streaming_toolcall_chunk3():
    _check_toolcall_stream(3)


def test_streaming_toolcall_charwise():
    # worst case: markers arrive one char at a time (mid-marker deltas)
    _check_toolcall_stream(1)


def test_streaming_toolcall_bigchunks():
    _check_toolcall_stream(17)


def test_streaming_reasoning_then_content():
    reasoning, content, tcs = _stream(RAW_ANSWER, 3)
    for f in _FRAMING:
        assert f not in reasoning and f not in content, f
    assert reasoning == "Think about it.", repr(reasoning)
    assert content == "The answer is 42.", repr(content)
    assert tcs == []


def test_truncated_cot_no_toolcall_nonstreaming():
    out = MuseGlimmerToolParser.extract_tool_calls(T, RAW_TRUNCATED, _FakeReq())
    assert not out.tools_called and out.tool_calls == []
    # partial reasoning must still be recovered by the reasoning parser
    reasoning, _ = MuseGlimmerReasoningParser.extract_reasoning(
        R, RAW_TRUNCATED, _FakeReq()
    )
    assert reasoning and "Maybe I should call" in reasoning, repr(reasoning)


def test_truncated_cot_no_toolcall_streaming():
    _, _, tcs = _stream(RAW_TRUNCATED, 3)
    assert tcs == [], f"truncated CoT invoke leaked as streaming tool call: {tcs}"


# ------------------------------------------------------- name normalization
#
# MuseGlimmer emits `get_weather.get_weather` for a bare-registered
# `get_weather`, and `weather.get` verbatim for a namespaced one. The parser
# normalizes against the tools actually registered on the request.


def test_doubled_bare_name_collapses():
    out = MuseGlimmerToolParser.extract_tool_calls(
        T, _call("get_weather.get_weather"), _req("get_weather")
    )
    assert out.tools_called and out.tool_calls[0].function.name == "get_weather", (
        out.tool_calls[0].function.name
    )


def test_namespaced_name_preserved():
    out = MuseGlimmerToolParser.extract_tool_calls(
        T, _call("weather.get"), _req("weather.get")
    )
    assert out.tool_calls[0].function.name == "weather.get"


def test_trailing_segment_unambiguous():
    # emitted foo.get_weather; registered get_weather (bare) -> bind to get_weather
    out = MuseGlimmerToolParser.extract_tool_calls(
        T, _call("foo.get_weather"), _req("get_weather")
    )
    assert out.tool_calls[0].function.name == "get_weather"


def test_trailing_segment_ambiguous_left_alone():
    # two registered tools share leaf 'get' -> ambiguous -> do NOT rewrite
    out = MuseGlimmerToolParser.extract_tool_calls(
        T, _call("x.get"), _req("weather.get", "time.get")
    )
    assert out.tool_calls[0].function.name == "x.get"


def test_no_registered_tools_passthrough():
    out = MuseGlimmerToolParser.extract_tool_calls(
        T, _call("get_weather.get_weather"), None
    )
    assert out.tool_calls[0].function.name == "get_weather.get_weather"


def test_exact_match_kept():
    out = MuseGlimmerToolParser.extract_tool_calls(
        T, _call("get_weather"), _req("get_weather")
    )
    assert out.tool_calls[0].function.name == "get_weather"
