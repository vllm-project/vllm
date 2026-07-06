# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from tests.tool_parsers.common_tests import ToolParserTestConfig, ToolParserTests
from tests.tool_parsers.utils import run_tool_extraction
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParser, ToolParserManager


GCML_START = "<｜GCML｜tool_calls>"
GCML_END = "</｜GCML｜tool_calls>"
GCML_INVOKE_START = "<｜GCML｜invoke"
GCML_INVOKE_END = "</｜GCML｜invoke>"
GCML_PARAMETER_START = "<｜GCML｜parameter"
GCML_PARAMETER_END = "</｜GCML｜parameter>"


def invoke(name: str, *parameters: tuple[str, str, bool]) -> str:
    body = "".join(
        f'<｜GCML｜parameter name="{param_name}" '
        f'string="{str(is_string).lower()}">{value}</｜GCML｜parameter>'
        for param_name, value, is_string in parameters
    )
    return f'<｜GCML｜invoke name="{name}">{body}</｜GCML｜invoke>'


def tool_calls(*calls: str) -> str:
    return f"{GCML_START}{''.join(calls)}{GCML_END}"


@pytest.fixture
def gigachat35_tokenizer(default_tokenizer: TokenizerLike) -> TokenizerLike:
    default_tokenizer.add_tokens(
        [
            GCML_START,
            GCML_END,
            GCML_INVOKE_START,
            GCML_INVOKE_END,
            GCML_PARAMETER_START,
            GCML_PARAMETER_END,
        ]
    )
    return default_tokenizer


class TestGigaChat35ToolParser(ToolParserTests):
    @pytest.fixture
    def tokenizer(self, gigachat35_tokenizer: TokenizerLike) -> TokenizerLike:
        return gigachat35_tokenizer

    @pytest.fixture
    def test_config(self) -> ToolParserTestConfig:
        return ToolParserTestConfig(
            parser_name="gigachat35",
            no_tool_calls_output="Plain answer without a tool call.",
            single_tool_call_output=tool_calls(
                invoke("get_weather", ("city", "Tokyo", True))
            ),
            parallel_tool_calls_output=tool_calls(
                invoke("get_weather", ("city", "Tokyo", True)),
                invoke("get_time", ("timezone", "Asia/Tokyo", True)),
            ),
            various_data_types_output=tool_calls(
                invoke(
                    "typed_args",
                    ("string_field", '"hello"', False),
                    ("int_field", "7", False),
                    ("float_field", "3.5", False),
                    ("bool_field", "true", False),
                    ("null_field", "null", False),
                    ("array_field", "[1, 2, 3]", False),
                    ("object_field", '{"nested": true}', False),
                )
            ),
            empty_arguments_output=tool_calls(invoke("empty_args")),
            surrounding_text_output="I will check." + tool_calls(
                invoke("get_weather", ("city", "Tokyo", True))
            ),
            escaped_strings_output=tool_calls(
                invoke("escaped", ("text", 'line 1\\n"quoted"', True))
            ),
            malformed_input_outputs=[
                f"{GCML_START}<｜GCML｜invoke name=\"broken\">",
                f"{GCML_START}<｜GCML｜parameter name=\"x\">1</｜GCML｜parameter>",
                f"{GCML_START}{GCML_END}",
            ],
            single_tool_call_expected_name="get_weather",
            single_tool_call_expected_args={"city": "Tokyo"},
            parallel_tool_calls_names=["get_weather", "get_time"],
            single_tool_call_expected_content=None,
            parallel_tool_calls_expected_content=None,
            allow_empty_or_json_empty_args=False,
        )


@pytest.mark.parametrize("parser_name", ["gigachat35"])
def test_parser_aliases(parser_name: str, default_tokenizer: TokenizerLike) -> None:
    parser_cls = ToolParserManager.get_tool_parser(parser_name)
    parser: ToolParser = parser_cls(default_tokenizer)
    content, tool_calls_found = run_tool_extraction(
        parser,
        tool_calls(invoke("lookup", ("query", "vLLM", True))),
        streaming=False,
    )
    assert content is None
    assert tool_calls_found[0].function.name == "lookup"
    assert json.loads(tool_calls_found[0].function.arguments) == {"query": "vLLM"}


def test_string_false_json_object(gigachat35_tokenizer: TokenizerLike) -> None:
    parser = ToolParserManager.get_tool_parser("gigachat35")(gigachat35_tokenizer)
    _, tool_calls_found = run_tool_extraction(
        parser,
        tool_calls(invoke("store", ("payload", '{"a": 1}', False))),
        streaming=True,
    )
    assert json.loads(tool_calls_found[0].function.arguments) == {
        "payload": {"a": 1},
    }
