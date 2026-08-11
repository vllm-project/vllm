# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Streaming + truncation regression tests for the MuseGlimmer parsers.

Covers the streaming-mode defects fixed in
``fix(muse_glimmer): channel-aware streaming tool-call + reasoning parser``:

  1. Streaming tool calls were dropped (stub returned None) and the reasoning
     parser leaked raw harmony framing tokens into ``content``.
  2. Truncation isolation (an unterminated ``to=self`` block containing an
     ``<atem:invoke>`` must NOT produce a tool call) — streaming AND non-streaming.
"""

import json

from vllm.reasoning.muse_glimmer_reasoning_parser import MuseGlimmerReasoningParser
from vllm.tool_parsers.muse_glimmer_tool_parser import MuseGlimmerToolParser

R = MuseGlimmerReasoningParser.__new__(
    MuseGlimmerReasoningParser
)  # skip __init__ (needs tokenizer)
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


# ---- captured raw framing string (tool-calling prompt) ----
RAW_TOOLCALL = (
    " to=self<|message|>I should read the hostname file to answer.<|eom|>"
    "<|start|>assistant to=read.read<|message|>"
    '<atem:function_calls>\n<atem:invoke name="read.read">\n'
    '<atem:parameter name="path">/etc/hostname</atem:parameter>\n'
    "</atem:invoke>\n</atem:function_calls>"
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


# ---- streaming with a to=user final answer (content channel) ----
RAW_ANSWER = (
    " to=self<|message|>Think about it.<|eom|>"
    "<|start|>assistant to=user<|message|>The answer is 42.<|eot|>"
)


def test_streaming_reasoning_then_content():
    reasoning, content, tcs = _stream(RAW_ANSWER, 3)
    for f in _FRAMING:
        assert f not in reasoning and f not in content, f
    assert reasoning == "Think about it.", repr(reasoning)
    assert content == "The answer is 42.", repr(content)
    assert tcs == []


# ---- truncation: unterminated to=self containing an invoke ----
RAW_TRUNCATED = (
    " to=self<|message|>Maybe I should call "
    '<atem:function_calls>\n<atem:invoke name="read.read">\n'
    '<atem:parameter name="path">/etc/hostname</atem:parameter>\n'
    "</atem:invoke>\n</atem:function_calls> but wait"
)  # NO closing <|eom|> -> truncated CoT


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


if __name__ == "__main__":
    test_streaming_toolcall_chunk3()
    test_streaming_toolcall_charwise()
    test_streaming_toolcall_bigchunks()
    test_streaming_reasoning_then_content()
    test_truncated_cot_no_toolcall_nonstreaming()
    test_truncated_cot_no_toolcall_streaming()
    print("ALL STREAMING/TRUNCATION TESTS PASSED")
