# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Bug-confirmation tests for the merged Qwen 3.5 parser changes.

Each test is a minimal reproducer of a real issue; they are meant to FAIL
until the corresponding bug is fixed.  Each scenario is also contrasted
against the Coder parser (for XML bugs) or the XML parser (for Coder bugs)
when one of the two already behaves correctly, which helps narrow down
where the fix belongs.

Run with:
    .venv/bin/python -m pytest tests/tool_parsers/test_qwen36_bugs.py -v
"""
import json

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.tokenizers import get_tokenizer
from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser
from vllm.tool_parsers.qwen3xml_tool_parser import Qwen3XMLToolParser

MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"


@pytest.fixture(scope="module")
def qwen3_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL)


def _stream(parser, chunks, request):
    """Feed pre-shaped string chunks and collect emitted tool-call pieces.

    Returns (content_str, tool_calls_dict_by_index).
    """
    prev_text = ""
    prev_ids: list[int] = []
    content_out = ""
    events: list[tuple] = []
    for chunk in chunks:
        cur_text = prev_text + chunk
        # Approximate: tokenize incrementally.
        dt_ids = parser.model_tokenizer.encode(chunk, add_special_tokens=False)
        cur_ids = prev_ids + dt_ids
        msg = parser.extract_tool_calls_streaming(
            prev_text, cur_text, chunk, prev_ids, cur_ids, dt_ids, request
        )
        if msg is not None:
            if msg.content:
                content_out += msg.content
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    events.append((
                        tc.index,
                        tc.function.name if tc.function else None,
                        tc.function.arguments if tc.function else None,
                    ))
        prev_text, prev_ids = cur_text, cur_ids
    tcs: dict[int, dict] = {}
    for idx, name, args in events:
        tcs.setdefault(idx, {"name": name, "args": ""})
        if args:
            tcs[idx]["args"] += args
    return content_out, tcs


# ---------------------------------------------------------------------------
# BUG 1: XML parser -- array parameter containing JSON true/false/null is
# emitted as a JSON string instead of being parsed as a JSON array.
#
# Root cause: in _end_element the deferred parser calls ast.literal_eval on
# the raw text.  ast.literal_eval does NOT understand JSON tokens `true`,
# `false`, `null` (Python uses True/False/None), so it raises and the fallback
# path emits the raw string wrapped with json.dumps.
#
# The Coder parser uses json.loads first, so it gets this scenario right --
# the test contrasts the two parsers to prove the bug is XML-specific.
# ---------------------------------------------------------------------------

_ARRAY_TOOLS = [
    ChatCompletionToolsParam(
        type="function",
        function={
            "name": "pick",
            "parameters": {
                "type": "object",
                "properties": {"items": {"type": "array"}},
            },
        },
    )
]

_ARRAY_WITH_JSON_BOOL_OUTPUT = (
    "<tool_call>\n<function=pick>\n"
    '<parameter=items>\n["a", "b", 1, true]\n</parameter>\n'
    "</function>\n</tool_call>"
)


def test_xml_array_with_json_bool_nonstreaming(qwen3_tokenizer):
    """XML non-streaming: array containing `true` must be parsed as a list."""
    parser = Qwen3XMLToolParser(qwen3_tokenizer, tools=_ARRAY_TOOLS)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=_ARRAY_TOOLS)
    result = parser.extract_tool_calls(_ARRAY_WITH_JSON_BOOL_OUTPUT, request=request)

    assert result.tools_called
    args = json.loads(result.tool_calls[0].function.arguments)
    assert isinstance(args["items"], list), (
        f"XML parser emitted items as {type(args['items']).__name__} "
        f"({args['items']!r}). ast.literal_eval cannot parse JSON `true` and "
        "the exception fallback wraps the raw string with json.dumps. "
        "Use json.loads first (see the Coder parser)."
    )
    assert args["items"] == ["a", "b", 1, True]


def test_coder_array_with_json_bool_nonstreaming(qwen3_tokenizer):
    """Contrast: Coder parser handles the same input correctly."""
    parser = Qwen3CoderToolParser(qwen3_tokenizer, tools=_ARRAY_TOOLS)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=_ARRAY_TOOLS)
    result = parser.extract_tool_calls(_ARRAY_WITH_JSON_BOOL_OUTPUT, request=request)

    assert result.tools_called
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args["items"] == ["a", "b", 1, True]


# ---------------------------------------------------------------------------
# BUG 2: Coder parser -- when two complete <tool_call>...</tool_call>
# blocks arrive in a SINGLE streaming delta (typical for speculative
# decoding), only the first tool call is emitted, the second is dropped.
#
# Root cause: extract_tool_calls_streaming advances current_tool_index by
# one per delta.  When a delta flushes two complete tool calls the parser
# processes call #0, sees tool_ends > current_tool_index, advances to #1,
# and returns None without re-processing the same delta.  The XML parser
# processes all complete elements in a loop and does not drop the second.
# ---------------------------------------------------------------------------

_WEATHER_TOOLS = [
    ChatCompletionToolsParam(
        type="function",
        function={
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    )
]

_TWO_TOOL_CALLS_IN_ONE_CHUNK = (
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
    "</function>\n</tool_call>\n"
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nLondon\n</parameter>\n"
    "</function>\n</tool_call>"
)


def test_coder_two_tool_calls_in_one_streaming_chunk(qwen3_tokenizer):
    """Coder streaming: a single delta that contains TWO complete tool calls
    must emit both, not just the first."""
    parser = Qwen3CoderToolParser(qwen3_tokenizer, tools=_WEATHER_TOOLS)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=_WEATHER_TOOLS)
    _, tcs = _stream(parser, [_TWO_TOOL_CALLS_IN_ONE_CHUNK], request)
    assert len(tcs) == 2, (
        f"Expected 2 tool calls, got {len(tcs)}. "
        "The Coder parser drops the second tool call when both complete in "
        "the same delta (speculative decoding scenario)."
    )
    args0 = json.loads(tcs[0]["args"])
    args1 = json.loads(tcs[1]["args"])
    assert args0 == {"city": "Paris"}
    assert args1 == {"city": "London"}


def test_xml_two_tool_calls_in_one_streaming_chunk(qwen3_tokenizer):
    """Contrast: XML parser already handles this case correctly."""
    parser = Qwen3XMLToolParser(qwen3_tokenizer, tools=_WEATHER_TOOLS)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=_WEATHER_TOOLS)
    _, tcs = _stream(parser, [_TWO_TOOL_CALLS_IN_ONE_CHUNK], request)
    assert len(tcs) == 2
    assert json.loads(tcs[0]["args"]) == {"city": "Paris"}
    assert json.loads(tcs[1]["args"]) == {"city": "London"}
