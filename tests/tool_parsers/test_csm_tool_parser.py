# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.entrypoints.openai.protocol import FunctionCall, ToolCall
from vllm.tokenizers import get_tokenizer
from vllm.tool_parsers.cwm_tool_parser import CwmToolParser

MODEL = "gpt2"


@pytest.fixture(scope="module")
def cwm_tokenizer():
    # The parser doesn't rely on tokenizer internals, but ToolParser requires it.
    return get_tokenizer(MODEL)


@pytest.fixture
def cwm_tool_parser(cwm_tokenizer):
    return CwmToolParser(cwm_tokenizer)


def assert_tool_calls(actual_tool_calls: list[ToolCall], expected: list[ToolCall]):
    assert len(actual_tool_calls) == len(expected)
    for actual, exp in zip(actual_tool_calls, expected):
        assert isinstance(actual.id, str)
        assert len(actual.id) > 16
        assert actual.type == "function"
        assert actual.function == exp.function


def test_extract_tool_calls_no_tools(cwm_tool_parser):
    model_output = "This is a test"
    extracted = cwm_tool_parser.extract_tool_calls(model_output, request=None)  # type: ignore[arg-type]
    assert not extracted.tools_called
    assert extracted.tool_calls == []
    assert extracted.content == model_output


def test_extract_tool_calls_single_terminal_inline(cwm_tool_parser):
    model_output = (
        "<think>Let me search.</think>\n"
        "<tool: function>\n"
        'terminal find . -name "*.py" | xargs grep -n "def bulk_create"\n'
        "</tool>\n"
    )
    extracted = cwm_tool_parser.extract_tool_calls(model_output, request=None)  # type: ignore[arg-type]
    assert extracted.tools_called
    expected = [
        ToolCall(
            function=FunctionCall(
                name="terminal",
                arguments=json.dumps(
                    {
                        "command": (
                            'find . -name "*.py" | xargs grep -n "def bulk_create"'
                        )
                    },
                    ensure_ascii=False,
                ),
            )
        )
    ]
    assert_tool_calls(extracted.tool_calls, expected)
    assert extracted.content == "<think>Let me search.</think>"


def test_extract_tool_calls_multiple_tools(cwm_tool_parser):
    model_output = (
        "preamble\n"
        "<tool: function>\n"
        "terminal echo hi\n"
        "</tool>\n"
        "<tool: function>\n"
        "terminal echo bye\n"
        "</tool>\n"
    )
    extracted = cwm_tool_parser.extract_tool_calls(model_output, request=None)  # type: ignore[arg-type]
    assert extracted.tools_called
    expected = [
        ToolCall(
            function=FunctionCall(
                name="terminal",
                arguments=json.dumps({"command": "echo hi"}, ensure_ascii=False),
            )
        ),
        ToolCall(
            function=FunctionCall(
                name="terminal",
                arguments=json.dumps({"command": "echo bye"}, ensure_ascii=False),
            )
        ),
    ]
    assert_tool_calls(extracted.tool_calls, expected)
    assert extracted.content == "preamble"


def test_extract_tool_calls_shorthand_missing_gt_uses_label_as_tool_name(
    cwm_tool_parser,
):
    # Shorthand format: `<tool: python ... </tool>` (missing `>` after label).
    model_output = (
        "prefix\n"
        "</think> "
        '<tool: python import os; print(os.listdir("_pytest")) </tool>\n'
    )
    extracted = cwm_tool_parser.extract_tool_calls(model_output, request=None)  # type: ignore[arg-type]
    assert extracted.tools_called
    expected = [
        ToolCall(
            function=FunctionCall(
                name="python",
                arguments=json.dumps(
                    {"command": 'import os; print(os.listdir("_pytest"))'},
                    ensure_ascii=False,
                ),
            )
        )
    ]
    assert_tool_calls(extracted.tool_calls, expected)
    assert extracted.content == "prefix\n</think>"
