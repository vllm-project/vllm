# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Low-level tests for MuseGlimmer reasoning and ATEM tool parsing.

End-to-end streaming and tokenizer coverage lives in
``test_muse_glimmer_parse_delta.py``.
"""

import json
from types import SimpleNamespace

import pytest

from vllm.reasoning.muse_glimmer_reasoning_parser import MuseGlimmerReasoningParser
from vllm.tool_parsers.muse_glimmer_tool_parser import MuseGlimmerToolParser

R: MuseGlimmerReasoningParser
T: MuseGlimmerToolParser


@pytest.fixture(autouse=True)
def _fresh_parsers():
    """Give each test request-scoped parser state through the real constructors."""
    global R, T
    R = MuseGlimmerReasoningParser(object())
    T = MuseGlimmerToolParser(object())


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


def test_sequential_tool_channels_across_eom_boundaries():
    # The model emits one tool call per channel; a turn may carry several
    # consecutive tool channels (e.g. in assistant history). Not parallel
    # generation -- the parser must segment each channel into its own call.
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
    reasoning, framed = MuseGlimmerReasoningParser.extract_reasoning(R, raw, None)
    assert reasoning == "thinking", repr(reasoning)
    out = MuseGlimmerToolParser.extract_tool_calls(T, framed, None)
    assert out.content == "The answer is 42.", repr(out.content)
    assert not out.tools_called


def test_plain_content_without_framing_passes_through():
    reasoning, content = MuseGlimmerReasoningParser.extract_reasoning(
        R, "Just a direct answer.", None
    )
    assert reasoning is None and content == "Just a direct answer.", (
        reasoning,
        content,
    )


def test_reasoning_then_sequential_tool_channels():
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


# ---------------------------------------------------------- reasoning boundary


def test_reasoning_end_requires_a_post_reasoning_channel():
    """``is_reasoning_end`` must fire once the turn leaves ``to=self`` for the
    ``to=user`` answer OR a tool channel -- this is the gate the structured-
    outputs backend uses to start applying the JSON grammar. It must NOT fire
    while still reasoning, nor for a channel header the model merely quotes
    inside an open reasoning span.
    """
    tokenizer = SimpleNamespace(decode=lambda token_ids: "".join(map(chr, token_ids)))
    parser = MuseGlimmerReasoningParser(tokenizer)

    def is_end(text):
        return parser.is_reasoning_end(list(map(ord, text)))

    reasoning = " to=self<|message|>thinking"
    answer = "<|eom|><|start|>assistant to=user<|message|>"
    tool = "<|eom|><|start|>assistant to=weather.get<|message|>"
    echoed = ' to=user<|message|> <atem:invoke name="weather.get">'

    assert is_end(reasoning + answer)
    assert is_end(reasoning + tool)
    assert not is_end(reasoning)
    assert not is_end(reasoning + echoed)


# NO closing <|eom|> -> truncated CoT
RAW_TRUNCATED = (
    " to=self<|message|>Maybe I should call "
    '<atem:function_calls>\n<atem:invoke name="read.read">\n'
    '<atem:parameter name="path">/etc/hostname</atem:parameter>\n'
    "</atem:invoke>\n</atem:function_calls> but wait"
)


def test_truncated_cot_no_toolcall_nonstreaming():
    out = MuseGlimmerToolParser.extract_tool_calls(T, RAW_TRUNCATED, _FakeReq())
    assert not out.tools_called and out.tool_calls == []
    # partial reasoning must still be recovered by the reasoning parser
    reasoning, _ = MuseGlimmerReasoningParser.extract_reasoning(
        R, RAW_TRUNCATED, _FakeReq()
    )
    assert reasoning and "Maybe I should call" in reasoning, repr(reasoning)


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


def test_unregistered_namespace_is_preserved():
    # Suffix-only matching can silently dispatch a tool from the wrong namespace.
    out = MuseGlimmerToolParser.extract_tool_calls(
        T, _call("foo.get_weather"), _req("get_weather")
    )
    assert out.tool_calls[0].function.name == "foo.get_weather"


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
