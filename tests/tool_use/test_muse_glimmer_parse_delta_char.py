# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Checkpoint-free MuseGlimmer parse_delta regression tests."""

from types import SimpleNamespace

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser import ParserManager
from vllm.reasoning.muse_glimmer_reasoning_parser import (
    MuseGlimmerReasoningParser,
)
from vllm.sampling_params import StructuredOutputsParams

_PROMPT = "<|start|>user<|message|>hi<|eom|><|start|>assistant"
_TOOL_XML = (
    "<atem:function_calls>\n"
    '<atem:invoke name="weather.get">\n'
    '<atem:parameter name="city">Paris</atem:parameter>\n'
    "</atem:invoke>\n"
    "</atem:function_calls>"
)


class _CharTokenizer:
    def decode(self, ids):
        return "".join(chr(token_id) for token_id in ids)

    def get_vocab(self):
        return {}


def _drive(chunks, *, prompt=_PROMPT, with_tool_parser=True):
    parser_kwargs = {"reasoning_parser_name": "muse_glimmer"}
    if with_tool_parser:
        parser_kwargs.update(
            tool_parser_name="muse_glimmer",
            enable_auto_tools=True,
        )
    parser = ParserManager.get_parser(**parser_kwargs)(_CharTokenizer())
    request = SimpleNamespace(
        tools=None,
        tool_choice="auto",
        include_reasoning=True,
    )
    prompt_ids = list(map(ord, prompt))
    messages = []
    for index, chunk in enumerate(chunks):
        message = parser.parse_delta(
            chunk,
            list(map(ord, chunk)),
            request,
            prompt_token_ids=prompt_ids if index == 0 else None,
            finished=index == len(chunks) - 1,
        )
        if message is not None:
            messages.append(message)

    reasoning = "".join(message.reasoning or "" for message in messages)
    content = "".join(message.content or "" for message in messages)
    tool_names = [
        tool.function.name for message in messages for tool in message.tool_calls or []
    ]
    return reasoning, content, tool_names


@pytest.mark.parametrize("with_tool_parser", [True, False])
def test_continued_user_channel_surfaces_clean_content(with_tool_parser):
    prompt = _PROMPT + " to=user<|message|>"

    reasoning, content, tools = _drive(
        ['{"value":"x"}<|eot|>'],
        prompt=prompt,
        with_tool_parser=with_tool_parser,
    )

    bare_reasoner = MuseGlimmerReasoningParser(_CharTokenizer())
    assert bare_reasoner.is_reasoning_end(list(map(ord, prompt)))
    assert reasoning == ""
    assert content == '{"value":"x"}'
    assert tools == []


def test_quoted_bare_tool_header_stays_in_reasoning():
    quoted = "I am quoting: to=weather.get<|message|>" + _TOOL_XML

    reasoning, content, tools = _drive([" to=self<|message|>" + quoted])

    assert reasoning == quoted
    assert content == ""
    assert tools == []


def test_fully_framed_tool_switch_after_open_reasoning_still_parses():
    reasoning, content, tools = _drive(
        [
            " to=self<|message|>Need weather.",
            "<|start|>assistant to=weather.get<|message|>" + _TOOL_XML,
        ]
    )

    assert reasoning == "Need weather."
    assert content == ""
    assert tools == ["weather.get"]


def test_answer_tail_is_emitted_before_straddled_tool_handoff():
    reasoning, content, tools = _drive(
        [
            " to=user<|message|>The answer",
            " tail.<|start|>assistant to=weather.get<|message|>",
            _TOOL_XML,
        ]
    )

    assert reasoning == ""
    assert content == "The answer tail."
    assert tools == ["weather.get"]


def test_final_delta_flushes_straddled_tool_handoff():
    reasoning, content, tools = _drive(
        [
            " to=self<|message|>think<|eom|>"
            "<|start|>assistant to=user<|message|>answer before ",
            "tool<|eom|><|start|>assistant to=weather.get<|message|>"
            + _TOOL_XML
            + "<|eot|>",
        ]
    )

    assert reasoning == "think"
    assert content == "answer before tool"
    assert tools == ["weather.get"]


@pytest.mark.parametrize("constraint", ["response_format", "structured_outputs"])
def test_active_tools_clear_caller_output_constraint(constraint):
    request_kwargs = {}
    if constraint == "response_format":
        request_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {"type": "object"},
            },
        }
    else:
        request_kwargs["structured_outputs"] = StructuredOutputsParams(
            json={"type": "object"}
        )
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "weather.get",
                    "parameters": {"type": "object"},
                },
            }
        ],
        tool_choice="auto",
        **request_kwargs,
    )
    parser = MuseGlimmerReasoningParser(_CharTokenizer())

    parser.adjust_request(request)

    assert request.response_format is None
    assert request.structured_outputs is None
