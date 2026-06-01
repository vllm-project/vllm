# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the HyperCLOVAX-SEED-Think-14B tool parser.

A mock tokenizer is used so the tests do not require the model weights.
Covers the templated header form, the bare-array hand-off (after the reasoning
parser strips the header), arguments/parameters aliasing, multiple calls, and
the abbreviated non-template form the model sometimes emits in
tool_choice="auto" - in both the non-streaming and streaming code paths.
"""

import json
from unittest.mock import MagicMock

import pytest

from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
from vllm.tool_parsers.hyperclovax_seed_think_14b_tool_parser import (
    HyperCLOVAXSeedThink14BToolParser,
)

PARSER_NAME = "hyperclovax_seed_think_14b"
HEADER = " -> tool/function_call\n"
END = "<|im_end|>"


@pytest.fixture
def tokenizer():
    tok = MagicMock()
    tok.get_vocab.return_value = {}
    return tok


@pytest.fixture
def tool_parser(tokenizer):
    return HyperCLOVAXSeedThink14BToolParser(tokenizer)


def make_request(tools=None, tool_choice="auto"):
    request = MagicMock()
    request.tools = tools
    request.tool_choice = tool_choice
    request.chat_template_kwargs = {}
    return request


_TOOLS = [{"type": "function", "function": {"name": "get_weather"}}]

# (label, output, needs_tools, expected_names, expected_args0)
EXTRACT_TOOL_CASES = [
    (
        "header_array_arguments",
        HEADER + '[{"name": "get_weather", "arguments": {"city": "Seoul"}}]' + END,
        False,
        ["get_weather"],
        {"city": "Seoul"},
    ),
    (
        "header_array_parameters_alias",
        HEADER + '[{"name": "get_weather", "parameters": {"a": 1}}]' + END,
        False,
        ["get_weather"],
        {"a": 1},
    ),
    (
        "bare_array_handoff",
        '[{"name": "get_weather", "parameters": {"city": "Busan"}}]',
        True,
        ["get_weather"],
        {"city": "Busan"},
    ),
    (
        "multiple_tool_calls",
        HEADER + '[{"name": "a", "arguments": {"x": 1}}, '
        '{"name": "b", "arguments": {"y": 2}}]' + END,
        False,
        ["a", "b"],
        {"x": 1},
    ),
]

# Outputs that must NOT be parsed as tool calls.
NON_TOOL_CASES = [
    # Abbreviated, non-template form sometimes emitted in tool_choice="auto":
    # "-> tool/<func>\n{...}" - a bare object with the real function name in
    # place of "function_call". The guarantee path is required + grammar.
    ("abbreviated_bare_object", '-> tool/get_current_weather\n{"city": "Seoul"}'),
    ("plain_text", "I cannot find the weather right now."),
]


class TestRegistration:
    def test_lazy_registered(self, tokenizer):
        cls = ToolParserManager.get_tool_parser(PARSER_NAME)
        assert cls is HyperCLOVAXSeedThink14BToolParser
        assert isinstance(cls(tokenizer), HyperCLOVAXSeedThink14BToolParser)


class TestExtractToolCalls:
    @pytest.mark.parametrize(
        "label,output,needs_tools,names,args0",
        EXTRACT_TOOL_CASES,
        ids=[c[0] for c in EXTRACT_TOOL_CASES],
    )
    def test_tool_call_forms(
        self, tool_parser, label, output, needs_tools, names, args0
    ):
        request = make_request(tools=_TOOLS if needs_tools else None)
        result = tool_parser.extract_tool_calls(output, request)
        assert result.tools_called
        assert [tc.function.name for tc in result.tool_calls] == names
        assert json.loads(result.tool_calls[0].function.arguments) == args0

    @pytest.mark.parametrize(
        "label,output", NON_TOOL_CASES, ids=[c[0] for c in NON_TOOL_CASES]
    )
    def test_non_tool_outputs_are_content(self, tool_parser, label, output):
        result = tool_parser.extract_tool_calls(output, make_request())
        assert not result.tools_called
        assert result.content == output


class TestStreaming:
    def _collect_tool_calls(self, parser, request, full_text):
        # Drive the streaming API one character per delta and collect deltas.
        previous_text = ""
        collected = []
        for i in range(1, len(full_text) + 1):
            msg = parser.extract_tool_calls_streaming(
                previous_text, full_text[:i], full_text[i - 1], [], [], [], request
            )
            if msg is not None and msg.tool_calls:
                collected.extend(msg.tool_calls)
            previous_text = full_text[:i]
        return [
            tc.function.name for tc in collected if tc.function and tc.function.name
        ]

    @pytest.mark.parametrize(
        "label,output,needs_tools",
        [
            (
                "header_array",
                HEADER + '[{"name": "get_weather", "arguments": {"city": "Seoul"}}]',
                False,
            ),
            (
                "bare_array",
                '[{"name": "get_weather", "parameters": {"city": "Busan"}}]',
                True,
            ),
        ],
        ids=["header_array", "bare_array"],
    )
    def test_streaming_tool_call_forms(self, tool_parser, label, output, needs_tools):
        request = make_request(tools=_TOOLS if needs_tools else None)
        names = self._collect_tool_calls(tool_parser, request, output)
        assert "get_weather" in names
