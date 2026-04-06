# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from tests.tool_parsers.common_tests import (
    ToolParserTestConfig,
    ToolParserTests,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
)
from vllm.tool_parsers.qwen3xml_tool_parser import StreamingXMLToolCallParser


class TestQwen3xmlToolParser(ToolParserTests):
    @pytest.fixture
    def test_config(self) -> ToolParserTestConfig:
        return ToolParserTestConfig(
            parser_name="qwen3_xml",
            # Test data
            no_tool_calls_output="This is a regular response without any tool calls.",
            single_tool_call_output="<tool_call>\n<function=get_weather>\n<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>",
            parallel_tool_calls_output="<tool_call>\n<function=get_weather>\n<parameter=city>Tokyo</parameter>\n</function>\n</tool_call><tool_call>\n<function=get_time>\n<parameter=timezone>Asia/Tokyo</parameter>\n</function>\n</tool_call>",
            various_data_types_output=(
                "<tool_call>\n<function=test_function>\n"
                "<parameter=string_field>hello</parameter>\n"
                "<parameter=int_field>42</parameter>\n"
                "<parameter=float_field>3.14</parameter>\n"
                "<parameter=bool_field>true</parameter>\n"
                "<parameter=null_field>null</parameter>\n"
                '<parameter=array_field>["a", "b", "c"]</parameter>\n'
                '<parameter=object_field>{"nested": "value"}</parameter>\n'
                "</function>\n</tool_call>"
            ),
            empty_arguments_output="<tool_call>\n<function=refresh>\n</function>\n</tool_call>",
            surrounding_text_output=(
                "Let me check the weather for you.\n\n"
                "<tool_call>\n<function=get_weather>\n"
                "<parameter=city>Tokyo</parameter>\n"
                "</function>\n</tool_call>\n\n"
                "I will get that information."
            ),
            escaped_strings_output=(
                "<tool_call>\n<function=test_function>\n"
                '<parameter=quoted>He said "hello"</parameter>\n'
                "<parameter=path>C:\\Users\\file.txt</parameter>\n"
                "<parameter=newline>line1\nline2</parameter>\n"
                "</function>\n</tool_call>"
            ),
            malformed_input_outputs=[
                "<tool_call><function=func>",
                "<tool_call><function=></function></tool_call>",
            ],
            # Expected results
            single_tool_call_expected_name="get_weather",
            single_tool_call_expected_args={"city": "Tokyo"},
            parallel_tool_calls_count=2,
            parallel_tool_calls_names=["get_weather", "get_time"],
            # xfail markers - Qwen3XML has systematic streaming issues
            xfail_streaming={
                "test_single_tool_call_simple_args": (
                    "Qwen3XML streaming has systematic issues"
                ),
                "test_parallel_tool_calls": "Qwen3XML streaming has systematic issues",
                "test_various_data_types": "Qwen3XML streaming has systematic issues",
                "test_empty_arguments": "Qwen3XML streaming has systematic issues",
                "test_surrounding_text": "Qwen3XML streaming has systematic issues",
                "test_escaped_strings": "Qwen3XML streaming has systematic issues",
                "test_malformed_input": (
                    "Qwen3XML parser is lenient with malformed input"
                ),
                "test_streaming_reconstruction": (
                    "Qwen3XML streaming reconstruction has known issues"
                ),
            },
            supports_typed_arguments=False,
        )


def test_get_param_type_anyof_type_conversion():
    """Test _get_param_type resolves anyOf/oneOf/type-as-array/$ref correctly.

    Pydantic v2 emits anyOf for Optional[T] fields. The previous implementation
    fell back to "string" for any param_def without a direct "type" key,
    causing incorrect type routing for nullable params and $ref schemas.
    """
    tools = [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "test_anyof",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "anyof_int": {
                            "anyOf": [{"type": "integer"}, {"type": "null"}],
                            "default": 5,
                        },
                        "anyof_str": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                        },
                        "anyof_array": {
                            "anyOf": [
                                {"type": "array", "items": {"type": "string"}},
                                {"type": "null"},
                            ],
                        },
                        "anyof_obj": {
                            "anyOf": [{"type": "object"}, {"type": "null"}],
                        },
                        "oneof_float": {
                            "oneOf": [{"type": "number"}, {"type": "null"}],
                        },
                        "type_as_array": {
                            "type": ["integer", "null"],
                        },
                        "multi_non_null": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "integer"},
                                {"type": "null"},
                            ],
                        },
                        "ref_param": {
                            "$ref": "#/$defs/ToolInput",
                        },
                        "anyof_ref": {
                            "anyOf": [
                                {"$ref": "#/$defs/ToolInput"},
                                {"type": "null"},
                            ],
                        },
                    },
                },
            },
        )
    ]

    parser = StreamingXMLToolCallParser()
    parser.set_tools(tools)
    parser.current_function_name = "test_anyof"

    assert parser._get_param_type("anyof_int") == "integer"
    assert parser._get_param_type("anyof_str") == "string"
    assert parser._get_param_type("anyof_array") == "array"
    assert parser._get_param_type("anyof_obj") == "object"
    assert parser._get_param_type("oneof_float") == "number"
    assert parser._get_param_type("type_as_array") == "integer"
    # Multi non-null: first non-null type is "string"
    assert parser._get_param_type("multi_non_null") == "string"
    # $ref: treated as object
    assert parser._get_param_type("ref_param") == "object"
    # anyOf[$ref, null]: Optional[BaseModel] pattern → object
    assert parser._get_param_type("anyof_ref") == "object"


def test_xml_parser_anyof_end_to_end():
    """End-to-end test: StreamingXMLToolCallParser converts anyOf params correctly.

    Verifies that parse_single_streaming_chunks produces properly typed argument
    values when the tool schema uses anyOf (Pydantic v2 Optional[T] pattern).
    """
    tools = [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "search_web",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                        },
                        "count": {
                            "anyOf": [{"type": "integer"}, {"type": "null"}],
                            "default": 5,
                        },
                        "filters": {
                            "$ref": "#/$defs/SearchFilters",
                        },
                    },
                },
            },
        )
    ]

    model_output = (
        "<tool_call>\n"
        "<function=search_web>\n"
        "<parameter=query>vllm tool parser</parameter>\n"
        "<parameter=count>10</parameter>\n"
        '<parameter=filters>{"site": "github.com"}</parameter>\n'
        "</function>\n"
        "</tool_call>"
    )

    parser = StreamingXMLToolCallParser()
    parser.set_tools(tools)
    result = parser.parse_single_streaming_chunks(model_output)

    assert result.tool_calls
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args["query"] == "vllm tool parser"
    assert isinstance(args["query"], str)
    assert args["count"] == 10
    assert isinstance(args["count"], int)
    assert args["filters"] == {"site": "github.com"}
    assert isinstance(args["filters"], dict)
