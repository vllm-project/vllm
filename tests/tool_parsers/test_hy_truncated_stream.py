# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Truncated HY tool-call streams must still end with parseable arguments.

These tests drive the production streaming path -- ``ParserManager.get_parser()``
plus ``Parser.parse_delta(..., finished=True)``, which is what
``vllm/entrypoints/openai/chat_completion/serving.py`` calls -- rather than the
``get_remaining_unstreamed_args()`` hook in isolation, so they fail when the
finalize step never reaches the hook.

HYV3/HYV4 stream tool-call arguments incrementally and withhold the closing
JSON brace (and, while a string value is still open, its closing quote) until
they see ``</tool_call>``.  When generation stops early -- max_tokens, a stop
sequence, a client disconnect -- that tag never arrives, and the withheld tail
has to be emitted at finalize time.  The last chunk of such a stream carries no
tool-call content of its own: the final token is usually the newline the chat
template places after ``</arg_value>``.

The mock vocabularies mirror the real ``tencent/Hy3-preview`` and
``tencent/Hy4-preview`` token shapes, including the ``:opensource`` suffix the
HY4 checkpoint puts on its structural tokens (HY4 reads the suffix out of the
vocab, so an unsuffixed test vocab exercises a different code path).
"""

import json
import re

import pytest

from tests.parser.engine.replay_harness import (
    MockTokenizer,
    accumulate_deltas,
    replay_streaming,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.parser.parser_manager import ParserManager

TOOLS = [
    ChatCompletionToolsParam(
        function=FunctionDefinition(
            name="get_weather",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}, "date": {"type": "string"}},
            },
        )
    ),
]
TOOLS_JSON = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "date": {"type": "string"},
                },
            },
        },
    },
]

# tencent/Hy3-preview: newline separated, `<tool_sep>` after the name.
HY3_MARKERS = [
    "<tool_calls>",
    "</tool_calls>",
    "<tool_call>",
    "</tool_call>",
    "<tool_sep>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
]

# tencent/Hy4-preview: no whitespace and no `<tool_sep>`; every structural
# token carries the checkpoint suffix.
HY4_MARKERS = [
    f"{marker.removesuffix('>')}:opensource>"
    for marker in HY3_MARKERS
    if marker != "<tool_sep>"
]

WORDS = ["get_weather", "city", "Beijing", "date", "2026", "\n"]


def _tokenizer(markers: list[str]) -> MockTokenizer:
    vocab = {tok: 1000 + i for i, tok in enumerate(markers + WORDS)}
    return MockTokenizer(vocab=vocab, tokens=[])


def _tokenize(text: str, vocab: dict[str, int]) -> list[tuple[int, str]]:
    """Split *text* into (id, text) pairs, one per mock token."""
    pattern = "|".join(re.escape(tok) for tok in sorted(vocab, key=len, reverse=True))
    parts = re.findall(pattern, text)
    assert "".join(parts) == text, f"mock vocab does not cover {text!r}"
    return [(vocab[part], part) for part in parts]


def _stream(
    tool_parser_name: str,
    markers: list[str],
    text: str,
    chunk_size: int | None = 1,
) -> dict:
    """Replay *text* through the production parser path, final chunk marked
    finished, and return the accumulated client-side output."""
    tokenizer = _tokenizer(markers)
    parser_cls = ParserManager.get_parser(
        tool_parser_name=tool_parser_name, enable_auto_tools=True
    )
    assert parser_cls is not None
    parser = parser_cls(tokenizer, TOOLS)
    tokens = _tokenize(text, tokenizer.get_vocab())
    results = replay_streaming(
        parser,
        tokens,
        chunk_size=chunk_size,
        finished_on_last=True,
        tools=TOOLS_JSON,
    )
    return accumulate_deltas(results)


def hy3_call(args: str) -> str:
    """A HY3 tool call truncated before ``</tool_call>``."""
    return "<tool_calls>\n<tool_call>get_weather<tool_sep>\n" + args


def hy4_call(args: str) -> str:
    """A HY4 tool call truncated before ``</tool_call:opensource>``."""
    return "<tool_calls:opensource>\n<tool_call:opensource>get_weather" + args


def assert_args(result: dict, index: int, expected: dict) -> None:
    assert len(result["tool_calls"]) > index, result
    raw = result["tool_calls"][index]["arguments"]
    # A JSONDecodeError here is the regression: the client is left holding an
    # unterminated fragment.
    assert json.loads(raw) == expected, raw


# ---------------------------------------------------------------------------
# HYV3 (tencent/Hy3-preview)
# ---------------------------------------------------------------------------
COMPLETE_ARGS = "<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>"


def test_hy3_complete_stream_is_unchanged():
    text = (
        "<tool_calls>\n<tool_call>get_weather<tool_sep>\n"
        + COMPLETE_ARGS
        + "\n</tool_call>\n</tool_calls>"
    )
    result = _stream("hy_v3", HY3_MARKERS, text)
    assert result["tool_calls"][0]["name"] == "get_weather"
    # An extra closing brace would make this invalid JSON.
    assert_args(result, 0, {"city": "Beijing"})


@pytest.mark.parametrize("tail", ["", "\n", "\n\n"])
def test_hy3_truncated_after_closed_argument(tail: str):
    """max_tokens stops right after the value, optionally on the template's
    trailing newline."""
    result = _stream("hy_v3", HY3_MARKERS, hy3_call(COMPLETE_ARGS) + tail)
    assert_args(result, 0, {"city": "Beijing"})


def test_hy3_truncated_mid_value():
    text = hy3_call(COMPLETE_ARGS) + "\n<arg_key>date</arg_key>\n<arg_value>2026"
    result = _stream("hy_v3", HY3_MARKERS, text)
    # The value the model actually emitted is kept, closed as a string.
    assert_args(result, 0, {"city": "Beijing", "date": "2026"})


def test_hy3_truncated_after_key_only():
    text = hy3_call("") + "<arg_key>city</arg_key>"
    result = _stream("hy_v3", HY3_MARKERS, text)
    assert_args(result, 0, {})


def test_hy3_second_call_truncated():
    first = "<tool_call>get_weather<tool_sep>\n" + COMPLETE_ARGS + "\n</tool_call>\n"
    second = (
        "<tool_call>get_weather<tool_sep>\n<arg_key>date</arg_key>\n<arg_value>2026"
    )
    result = _stream("hy_v3", HY3_MARKERS, "<tool_calls>\n" + first + second)
    names = [tc["name"] for tc in result["tool_calls"]]
    assert names == ["get_weather", "get_weather"]
    assert_args(result, 0, {"city": "Beijing"})
    assert_args(result, 1, {"date": "2026"})


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5])
def test_hy3_truncation_is_chunk_size_invariant(chunk_size: int):
    """The closing tail must be emitted whether the last batch contains the
    newline alone or the value tag together with it.

    (A single batch carrying the whole text is a different, pre-existing HY3
    path -- it never enters incremental streaming -- and is out of scope.)
    """
    result = _stream(
        "hy_v3",
        HY3_MARKERS,
        hy3_call(COMPLETE_ARGS) + "\n",
        chunk_size=chunk_size,
    )
    assert_args(result, 0, {"city": "Beijing"})


# ---------------------------------------------------------------------------
# HYV4 (tencent/Hy4-preview)
# ---------------------------------------------------------------------------
HY4_COMPLETE_ARGS = (
    "<arg_key:opensource>city</arg_key:opensource>"
    "<arg_value:opensource>Beijing</arg_value:opensource>"
)


def test_hy4_complete_stream_is_unchanged():
    text = (
        hy4_call(HY4_COMPLETE_ARGS)
        + "</tool_call:opensource>\n</tool_calls:opensource>"
    )
    result = _stream("hy_v4", HY4_MARKERS, text)
    assert result["tool_calls"][0]["name"] == "get_weather"
    assert_args(result, 0, {"city": "Beijing"})


@pytest.mark.parametrize("tail", ["", "\n"])
def test_hy4_truncated_after_closed_argument(tail: str):
    result = _stream("hy_v4", HY4_MARKERS, hy4_call(HY4_COMPLETE_ARGS) + tail)
    assert_args(result, 0, {"city": "Beijing"})


def test_hy4_truncated_mid_value():
    text = hy4_call(
        HY4_COMPLETE_ARGS
        + "<arg_key:opensource>date</arg_key:opensource>"
        + "<arg_value:opensource>2026"
    )
    result = _stream("hy_v4", HY4_MARKERS, text)
    assert_args(result, 0, {"city": "Beijing", "date": "2026"})


def test_hy4_truncated_with_whitespace_between_tags():
    """A model may emit a newline between `</arg_key>` and `<arg_value>` even
    though the checkpoint template does not; the value must still be parsed."""
    text = hy4_call(
        "<arg_key:opensource>city</arg_key:opensource>\n"
        "<arg_value:opensource>Beijing</arg_value:opensource>"
    )
    result = _stream("hy_v4", HY4_MARKERS, text)
    assert_args(result, 0, {"city": "Beijing"})
