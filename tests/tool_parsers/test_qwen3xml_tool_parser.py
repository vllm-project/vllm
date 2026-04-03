# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest

from tests.tool_parsers.common_tests import (
    ToolParserTestConfig,
    ToolParserTests,
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


@pytest.mark.parametrize(
    "param_value,param_type",
    [
        ("not_a_number", "int"),
        ("not_a_number", "integer"),
        ("not_a_number", "uint"),
        ("not_a_number", "long"),
        ("not_a_number", "short"),
        ("not_a_number", "unsigned"),
        ("not_a_float", "num"),
        ("not_a_float", "float"),
    ],
)
def test_convert_param_value_invalid_emits_warning(
    param_value: str, param_type: str
) -> None:
    """_convert_param_value should emit a properly-formatted warning when a
    value cannot be converted to int or float.

    Previously the logger.warning() calls had 3 '%s' placeholders but only
    1 argument was passed, causing Python's logging machinery to raise a
    TypeError internally and print a '--- Logging error ---' traceback to
    stderr instead of the intended warning message.  After the fix the
    warning must be emitted without error and the raw value must appear in
    the message.
    """
    import logging

    parser = StreamingXMLToolCallParser()

    # vllm loggers have propagate=False, so we attach a handler directly.
    vllm_logger = logging.getLogger("vllm.tool_parsers.qwen3xml_tool_parser")
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Capture(level=logging.WARNING)
    vllm_logger.addHandler(handler)
    try:
        result = parser._convert_param_value(
            param_value, param_type, param_name="my_param", func_name="my_tool"
        )
    finally:
        vllm_logger.removeHandler(handler)

    # The value should be returned unchanged (degenerate-to-string path)
    assert result == param_value

    # Exactly one warning must have been captured — not zero (lost to TypeError)
    warning_records = [r for r in records if r.levelno == logging.WARNING]
    assert len(warning_records) == 1, f"Expected 1 warning, got {len(warning_records)}"
    # The message must format successfully and contain all three context fields
    msg = warning_records[0].getMessage()
    assert param_value in msg, (
        f"param_value '{param_value}' not found in warning: '{msg}'"
    )
    assert "my_param" in msg, f"param_name 'my_param' not found in warning: '{msg}'"
    assert "my_tool" in msg, f"func_name 'my_tool' not found in warning: '{msg}'"
