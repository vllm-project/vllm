# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.tool_parsers.deepseekv32_tool_parser import DeepSeekV32ToolParser

pytestmark = pytest.mark.cpu_test


class FakeTokenizer:
    """Minimal fake tokenizer: a truthy model_tokenizer marker plus the DSML
    function-calls sentinel tokens."""

    model_tokenizer = True

    def __init__(self):
        self.vocab = {
            "<｜DSML｜function_calls>": 1,
            "</｜DSML｜function_calls>": 2,
        }

    def get_vocab(self):
        return self.vocab


@pytest.fixture
def parser():
    return DeepSeekV32ToolParser(FakeTokenizer())


def test_extract_tool_calls_keeps_unclosed_last_parameter(parser):
    # #57827: the model sometimes closes the invoke without closing the last
    # parameter; its trailing value must survive the final conversion.
    text = (
        '<｜DSML｜function_calls><｜DSML｜invoke name="get_weather">'
        '<｜DSML｜parameter name="city" string="true">Seattle</｜DSML｜parameter>'
        '<｜DSML｜parameter name="unit" string="true">celsius'
        "</｜DSML｜invoke></｜DSML｜function_calls>"
    )
    result = parser.extract_tool_calls(text, None)

    assert result.tools_called is True
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args == {"city": "Seattle", "unit": "celsius"}


def test_extract_tool_calls_fully_closed_output_unchanged(parser):
    text = (
        '<｜DSML｜function_calls><｜DSML｜invoke name="get_weather">'
        '<｜DSML｜parameter name="city" string="true">Seattle</｜DSML｜parameter>'
        '<｜DSML｜parameter name="unit" string="true">celsius</｜DSML｜parameter>'
        "</｜DSML｜invoke></｜DSML｜function_calls>"
    )
    result = parser.extract_tool_calls(text, None)

    args = json.loads(result.tool_calls[0].function.arguments)
    assert args == {"city": "Seattle", "unit": "celsius"}
