# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the MuseGlimmer ATEM tool parser and reasoning parser.

MuseGlimmer writes every turn as a sequence of channel-scoped messages rather
than JSON, so the two parsers are tested together: the reasoning parser strips
the reasoning span and forwards the remaining channels as content, and the tool
parser reads ATEM markup out of those channels.

Five areas, in order:

  1. non-streaming tool-call extraction, including channel scoping (an
     ``<atem:invoke>`` echoed inside reasoning must never become a call);
  2. the reasoning -> tool-parser handoff, which regressed once by returning
     ``content=None`` and starving the tool parser;
  3. streaming, where markers routinely straddle chunk boundaries, plus
     truncation isolation for an unterminated ``to=self`` block;
  4. tool-name normalization against the tools registered on the request;
  5. the grammar-gate prefilter, which must answer `is_reasoning_end` per
     decode step without re-decoding the sequence, and must never skip a
     False->True transition of the full check.

These drive the parsers directly and need no checkpoint. The tests that require
a real tokenizer live in ``test_muse_glimmer_parse_delta.py``.
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


# ---------------------------------------------------------- grammar-gate prefilter
#
# `is_reasoning_end_streaming` runs once per decode step per running request
# while a structured-output request is still reasoning. Answering it with a
# full-sequence decode makes decoding quadratic in sequence length, so the
# parser prefilters on the delta's token ids and only decodes on steps that
# could have completed a channel header. These tests pin the two properties
# that make the prefilter sound: no False->True transition of the full check
# is ever skipped, and ordinary steps decode nothing.


class _FakeTokenizer:
    """Concatenative tokenizer offering several spellings per marker.

    Byte-level BPE lets one marker be spelled by many token-id sequences, so
    `<|message|>` is available here as one piece, as two, and as four. Tokens
    decode by concatenation, which is what the parser relies on.
    """

    _PIECES = (
        "<|message|>",
        "<|mes",
        "sage|>",
        "<|",
        "mess",
        "age",
        "|>",
        "<|eom|>",
        "<|eot|>",
        "<|start|>",
        "assistant",
        " to=self",
        " to=user",
        " to=weather.get",
        "<atem:function_calls>",
        '<atem:invoke name="weather.get">',
    )

    def __init__(self):
        pieces = list(self._PIECES)
        pieces += [chr(c) for c in range(32, 127)]
        self._id_to_text = pieces
        self._text_to_id = {text: i for i, text in enumerate(pieces)}
        self.decode_calls = 0

    def get_vocab(self):
        return dict(self._text_to_id)

    def decode(self, token_ids, **kwargs):
        self.decode_calls += 1
        return "".join(self._id_to_text[i] for i in token_ids)

    def ids(self, *pieces):
        """Token ids for an explicit spelling; unknown pieces go char by char."""
        out = []
        for piece in pieces:
            if piece in self._text_to_id:
                out.append(self._text_to_id[piece])
            else:
                out.extend(self._text_to_id[ch] for ch in piece)
        return out


@pytest.fixture
def tok():
    return _FakeTokenizer()


def _first_fire(parser, ids):
    """Index of the step where the gate first opens, walking one token at a time.

    Mirrors `StructuredOutputManager._find_reasoning_end_index`, which is how
    the engine locates the boundary token.
    """
    for i in range(len(ids)):
        if parser.is_reasoning_end_streaming(ids[: i + 1], ids[i : i + 1]):
            return i
    return None


# `<|message|>` spellings, from one token to four.
_MARKER_SPELLINGS = [
    ("<|message|>",),
    ("<|mes", "sage|>"),
    ("<|", "mess", "age", "|>"),
]


@pytest.mark.parametrize("marker", _MARKER_SPELLINGS, ids=lambda m: str(len(m)))
def test_gate_opens_when_the_answer_header_completes(tok, marker):
    """The gate must open at `to=user`, exactly when its header completes."""
    parser = MuseGlimmerReasoningParser(tok)
    prefix = (" to=self", "<|message|>", "h", "m", "<|eom|>", "<|start|>", "assistant")
    ids = tok.ids(*prefix, " to=user", *marker, "4", "2", "<|eot|>")
    assert _first_fire(parser, ids) == len(tok.ids(*prefix, " to=user", *marker)) - 1


@pytest.mark.parametrize("marker", _MARKER_SPELLINGS, ids=lambda m: str(len(m)))
def test_prefilter_never_misses_a_transition(tok, marker):
    """Every False->True step of the full check must survive the prefilter.

    The caller latches the first True (`StructuredOutputManager.should_advance`
    sets `reasoning_ended`), so the prefilter may answer False on later steps
    but must never skip a transition. The reasoning body quotes a channel
    header character by character so the confirm-tail path runs too.
    """
    parser = MuseGlimmerReasoningParser(tok)
    ids = tok.ids(
        " to=self",
        *marker,
        "quoting a header: to=user<|message|> ...but only quoting",
        "<|eom|>",
        "<|start|>",
        "assistant",
        " to=user",
        *marker,
        "short answer",
        "<|eot|>",
        "<|start|>",
        "assistant",
        " to=self",
        *marker,
        "more thought",
        "<|eom|>",
        "<|start|>",
        "assistant",
        " to=user",
        *marker,
        "done",
        "<|eot|>",
    )
    previous = False
    transitions = 0
    for i in range(len(ids)):
        prefix = ids[: i + 1]
        reference = parser.is_reasoning_end(prefix)
        got = parser.is_reasoning_end_streaming(prefix, ids[i : i + 1])
        if reference and not previous:
            transitions += 1
            assert got, f"missed transition at step {i}: {tok.decode(prefix)!r}"
        assert not got or reference, f"false positive at step {i}"
        previous = reference
    assert transitions >= 2, "test lost its multi-transition coverage"


def test_prefilter_skips_decoding_on_ordinary_steps(tok):
    """The regression guard: no full-sequence decode per generated token."""
    parser = MuseGlimmerReasoningParser(tok)
    ids = tok.ids(" to=self", "<|message|>", "a long chain of thought " * 20)
    tok.decode_calls = 0
    for i in range(len(ids)):
        parser.is_reasoning_end_streaming(ids[: i + 1], ids[i : i + 1])
    # Only the step completing `<|message|>` may decode; the body must not.
    assert tok.decode_calls == 1, tok.decode_calls


def test_gate_opens_at_a_tool_header_too(tok):
    """A tool channel ends reasoning for the grammar, same as an answer."""
    parser = MuseGlimmerReasoningParser(tok)
    prefix = (
        " to=self",
        "<|message|>",
        "call it",
        "<|eom|>",
        "<|start|>",
        "assistant",
        " to=weather.get",
        "<|message|>",
    )
    ids = tok.ids(
        *prefix,
        "<atem:function_calls>",
        '<atem:invoke name="weather.get">',
    )
    assert _first_fire(parser, ids) == len(tok.ids(*prefix)) - 1


def test_prefilter_disables_itself_without_a_vocabulary():
    """A tokenizer with no vocabulary falls back to the full check unchanged."""
    parser = MuseGlimmerReasoningParser(object())
    assert parser._channel_marker_completers == frozenset()


@pytest.mark.skipif(
    "MUSE_GLIMMER_CKPT" not in __import__("os").environ,
    reason="needs a MuseGlimmer checkpoint for the real tokenizer",
)
@pytest.mark.parametrize(
    "text",
    [
        " to=self<|message|>hm<|eom|><|start|>assistant to=user<|message|>Because",
        " to=self<|message|>hm<|eom|><|start|>assistant"
        " to=weather.get<|message|><atem:function_calls>",
    ],
    ids=["answer", "tool"],
)
def test_prefilter_matches_full_check_on_the_real_tokenizer(text):
    """Replay a channel switch token by token against the real vocabulary."""
    import os

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(os.environ["MUSE_GLIMMER_CKPT"])
    parser = MuseGlimmerReasoningParser(tokenizer)
    ids = tokenizer.encode(text, add_special_tokens=False)
    previous = False
    for i in range(len(ids)):
        prefix = ids[: i + 1]
        reference = parser.is_reasoning_end(prefix)
        got = parser.is_reasoning_end_streaming(prefix, ids[i : i + 1])
        if reference and not previous:
            assert got, f"missed transition at step {i}"
        assert not got or reference, f"false positive at step {i}"
        previous = reference
