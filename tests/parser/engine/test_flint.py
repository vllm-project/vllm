# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flint parser tests.

Flint's generation prompt ends with ``<|assistant|>`` and does not open
``<think>``: the model decides on each turn whether to reason.
"""

import json

import pytest

from tests.parser.engine.replay_harness import (
    MockTokenizer,
    _test_request,
    collect_output,
    replay_streaming,
)
from vllm.parser.parser_manager import ParserManager

THINK_S, THINK_E = 1000, 1001
TOOL_S, TOOL_E = 1002, 1003
USER, ASSISTANT = 1004, 1005
TEXT = 42

VOCAB = {
    "<think>": THINK_S,
    "</think>": THINK_E,
    "<tool_call>": TOOL_S,
    "</tool_call>": TOOL_E,
    "<|user|>": USER,
    "<|assistant|>": ASSISTANT,
}
PROMPT = [USER, TEXT, ASSISTANT]

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
        },
    },
}


def _parser(tokens: list[tuple[int, str]] | None = None, **kwargs):
    parser_cls = ParserManager.get_parser(
        tool_parser_name="flint",
        reasoning_parser_name="flint",
        enable_auto_tools=True,
    )
    assert parser_cls is not None
    return parser_cls(MockTokenizer(VOCAB, tokens or []), [WEATHER_TOOL], **kwargs)


@pytest.mark.parametrize(
    ("output", "chat_template_kwargs", "reasoning", "content"),
    [
        ("391", None, None, "391"),
        ("<think>Easy.</think>391", {"enable_thinking": False}, "Easy.", "391"),
    ],
    ids=["no-think", "think-enable_thinking_false"],
)
def test_reasoning_only_inside_think(output, chat_template_kwargs, reasoning, content):
    """A reply without ``<think>`` is content; Flint has no thinking switch."""
    parser = _parser(chat_template_kwargs=chat_template_kwargs)

    got_reasoning, got_content, _ = parser.parse(
        output, _test_request(tools=[WEATHER_TOOL]), enable_auto_tools=True
    )

    assert (got_reasoning, got_content) == (reasoning, content)


def test_streamed_tool_call_without_think():
    """A streamed reply that starts with text reaches the tool parser."""
    tokens = [
        (TEXT, "Checking."),
        (TOOL_S, "<tool_call>"),
        (43, "get_weather"),
        (44, "<arg_key>city</arg_key>"),
        (45, "<arg_value>Paris</arg_value>"),
        (TOOL_E, "</tool_call>"),
    ]
    parser = _parser(tokens)

    output = collect_output(
        replay_streaming(
            parser,
            tokens,
            chunk_size=1,
            finished_on_last=True,
            tools=[WEATHER_TOOL],
            prompt_token_ids=PROMPT,
        )
    )

    assert (output.reasoning, output.content) == ("", "Checking.")
    assert [(c["name"], json.loads(c["arguments"])) for c in output.tool_calls] == [
        ("get_weather", {"city": "Paris"})
    ]


def test_structured_outputs_wait_for_reasoning_to_end():
    """The tool-call grammar excludes ``<think>``, so it must not apply from
    the first generated token. It starts after ``</think>``, or at
    ``<tool_call>`` in a turn without reasoning."""
    parser = _parser()
    request = parser.adjust_request(_test_request(tools=[WEATHER_TOOL]))

    assert request.structured_outputs is not None
    assert request.resolve_reasoning_ended(parser, PROMPT) is False
    assert not parser.is_reasoning_end([*PROMPT, THINK_S, TEXT])
    assert parser.is_reasoning_end([*PROMPT, THINK_S, TEXT, THINK_E])
    assert parser.is_reasoning_end([*PROMPT, TEXT, TOOL_S])
