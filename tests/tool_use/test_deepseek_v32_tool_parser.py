# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.tool_parsers.deepseek_v32_tool_parser import (
    DeepSeekV32ToolParser,
)
from vllm.tokenizers import get_tokenizer

pytestmark = pytest.mark.cpu_test

MODEL = "deepseek-ai/DeepSeek-V3"


@pytest.fixture(scope="module")
def deepseek_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL)


@pytest.fixture
def deepseek_tool_parser(deepseek_tokenizer):
    return DeepSeekV32ToolParser(deepseek_tokenizer)


@pytest.fixture
def sample_tools():
    return [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_weather",
                "description": "Get the weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city name"},
                        "date": {"type": "string", "description": "The date"},
                    },
                    "required": ["location", "date"],
                },
            },
        ),
    ]


def assert_tool_calls(
    actual_tool_calls: list[ToolCall], expected_tool_calls: list[ToolCall]
):
    assert len(actual_tool_calls) == len(expected_tool_calls)

    for actual_tool_call, expected_tool_call in zip(
        actual_tool_calls, expected_tool_calls
    ):
        assert actual_tool_call.type == "function"
        assert actual_tool_call.function.name == expected_tool_call.function.name
        try:
            assert json.loads(actual_tool_call.function.arguments) == json.loads(
                expected_tool_call.function.arguments
            )
        except json.JSONDecodeError as e:
            print(e)
            print("actual_tool_call", actual_tool_call.function.arguments)
            print("expected_tool_call", expected_tool_call.function.arguments)


def test_extract_tool_calls_single_function(
    deepseek_tool_parser,
    sample_tools,
):
    """Test extracting a single function call"""
    model_output = """<｜DSML｜function_calls>
<｜DSML｜invoke name="get_weather">
<｜DSML｜parameter name="location" string="true">Hangzhou</｜DSML｜parameter>
<｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"""

    expected_tool_calls = [
        ToolCall(
            function=FunctionCall(
                name="get_weather",
                arguments=json.dumps({"location": "Hangzhou", "date": "2024-01-16"}),
            )
        ),
    ]

    request = ChatCompletionRequest(
        model="deepseek-v3",
        messages=[{"role": "user", "content": "What is the weather?"}],
        tools=sample_tools,
    )

    extracted = deepseek_tool_parser.extract_tool_calls(model_output, request)

    assert extracted.tools_called
    assert extracted.content is None
    assert_tool_calls(extracted.tool_calls, expected_tool_calls)


def test_extract_tool_calls_multiple_functions(
    deepseek_tool_parser,
    sample_tools,
):
    """Test extracting multiple function calls"""
    model_output = """<｜DSML｜function_calls>
<｜DSML｜invoke name="get_weather">
<｜DSML｜parameter name="location" string="true">Hangzhou</｜DSML｜parameter>
<｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>
</｜DSML｜invoke>
<｜DSML｜invoke name="get_weather">
<｜DSML｜parameter name="location" string="true">Beijing</｜DSML｜parameter>
<｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"""

    expected_tool_calls = [
        ToolCall(
            function=FunctionCall(
                name="get_weather",
                arguments=json.dumps({"location": "Hangzhou", "date": "2024-01-16"}),
            )
        ),
        ToolCall(
            function=FunctionCall(
                name="get_weather",
                arguments=json.dumps({"location": "Beijing", "date": "2024-01-16"}),
            )
        ),
    ]

    request = ChatCompletionRequest(
        model="deepseek-v3",
        messages=[{"role": "user", "content": "What is the weather?"}],
        tools=sample_tools,
    )

    extracted = deepseek_tool_parser.extract_tool_calls(model_output, request)

    assert extracted.tools_called
    assert extracted.content is None
    assert_tool_calls(extracted.tool_calls, expected_tool_calls)


def test_extract_tool_calls_with_end_of_sentence_token(
    deepseek_tool_parser,
    sample_tools,
):
    """Test extracting function calls with end-of-sentence token"""
    model_output = """<｜DSML｜function_calls>
<｜DSML｜invoke name="get_weather">
<｜DSML｜parameter name="location" string="true">Hangzhou</｜DSML｜parameter>
<｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls><｜end▁of▁sentence｜>"""

    expected_tool_calls = [
        ToolCall(
            function=FunctionCall(
                name="get_weather",
                arguments=json.dumps({"location": "Hangzhou", "date": "2024-01-16"}),
            )
        ),
    ]

    request = ChatCompletionRequest(
        model="deepseek-v3",
        messages=[{"role": "user", "content": "What is the weather?"}],
        tools=sample_tools,
    )

    extracted = deepseek_tool_parser.extract_tool_calls(model_output, request)

    assert extracted.tools_called
    assert extracted.content is None
    assert_tool_calls(extracted.tool_calls, expected_tool_calls)
