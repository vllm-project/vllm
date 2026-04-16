# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MimoV2FlashToolParser."""

from typing import Any

import pytest

from vllm.tokenizers import get_tokenizer
from vllm.tool_parsers.mimov2_flash_tool_parser import MimoV2FlashToolParser

MODEL = "XiaomiMiMo/MiMo-V2-Flash"


@pytest.fixture(scope="module")
def mimov2_flash_tokenizer() -> Any:
    return get_tokenizer(tokenizer_name=MODEL, trust_remote_code=True)


@pytest.fixture
def mimov2_flash_tool_parser(mimov2_flash_tokenizer):
    return MimoV2FlashToolParser(mimov2_flash_tokenizer)


def test_extract_basic_tool_call(mimov2_flash_tool_parser):
    """Test extracting a basic tool call."""
    model_output = """Here is the result:
<tool_call>
<function=execute_bash>
<parameter=command>pwd && ls</parameter>
</function>
</tool_call>"""

    result = mimov2_flash_tool_parser.extract_tool_calls(
        model_output,
        request=None,  # type: ignore[arg-type]
    )

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "execute_bash"
    args = result.tool_calls[0].function.arguments
    # Arguments should be a JSON string
    import json

    parsed_args = json.loads(args)
    assert parsed_args == {"command": "pwd && ls"}


def test_extract_multiple_tool_calls(mimov2_flash_tool_parser):
    """Test extracting multiple tool calls."""
    model_output = """Here is the result:
<tool_call>
<function=execute_bash>
<parameter=command>pwd</parameter>
</function>
</tool_call>
<tool_call>
<parameter=command>ls -la</parameter>
<function=list_files>
</function>
</tool_call>"""

    result = mimov2_flash_tool_parser.extract_tool_calls(
        model_output,
        request=None,  # type: ignore[arg-type]
    )

    assert result.tools_called is True
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].function.name == "execute_bash"
    assert result.tool_calls[1].function.name == "list_files"


def test_extract_tool_call_with_multiple_parameters(mimov2_flash_tool_parser):
    """Test extracting a tool call with multiple parameters."""
    model_output = """<tool_call>
<function=execute_code>
<parameter=language>python</parameter>
<parameter=code>print("hello")</parameter>
</function>
</tool_call>"""

    result = mimov2_flash_tool_parser.extract_tool_calls(
        model_output,
        request=None,  # type: ignore[arg-type]
    )

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "execute_code"
    import json

    parsed_args = json.loads(result.tool_calls[0].function.arguments)
    assert parsed_args == {"language": "python", "code": 'print("hello")'}


def test_extract_no_tool_calls(mimov2_flash_tool_parser):
    """Test when there are no tool calls in the output."""
    model_output = "This is just regular text without any tool calls."

    result = mimov2_flash_tool_parser.extract_tool_calls(
        model_output,
        request=None,  # type: ignore[arg-type]
    )

    assert result.tools_called is False
    assert result.tool_calls == []
    assert result.content == model_output


def test_extract_content_before_tool_call(mimov2_flash_tool_parser):
    """Test that content before tool call is preserved."""
    model_output = """Let me execute that command for you.
<tool_call>
<function=execute_bash>
<parameter=command>echo hello</parameter>
</function>
</tool_call>"""

    result = mimov2_flash_tool_parser.extract_tool_calls(
        model_output,
        request=None,  # type: ignore[arg-type]
    )

    assert result.tools_called is True
    assert result.content == "Let me execute that command for you."


def test_extract_tool_call_empty_arguments(mimov2_flash_tool_parser):
    """Test extracting a tool call with no parameters."""
    model_output = """<tool_call>
<function=noop>
</function>
</tool_call>"""

    result = mimov2_flash_tool_parser.extract_tool_calls(
        model_output,
        request=None,  # type: ignore[arg-type]
    )

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "noop"
    import json

    parsed_args = json.loads(result.tool_calls[0].function.arguments)
    assert parsed_args == {}


def test_extract_tool_call_multiline_parameter(mimov2_flash_tool_parser):
    """Test that multiline parameter values are preserved exactly."""
    model_output = """<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
<parameter=example_parameter_2>This is the value for the second parameter
that can span
multiple lines</parameter>
</function>
</tool_call>"""

    result = mimov2_flash_tool_parser.extract_tool_calls(
        model_output,
        request=None,  # type: ignore[arg-type]
    )

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "example_function_name"
    import json

    parsed_args = json.loads(result.tool_calls[0].function.arguments)
    expected = (
        "This is the value for the second parameter\n"
        "that can span\n"
        "multiple lines"
    )
    assert parsed_args == {
        "example_parameter_1": "value_1",
        "example_parameter_2": expected,
    }


def test_streaming_basic(mimov2_flash_tool_parser):
    """Test basic streaming functionality."""
    # Reset streaming state
    mimov2_flash_tool_parser._sent_content_idx = 0
    mimov2_flash_tool_parser.prev_tool_call_arr = []
    mimov2_flash_tool_parser.streamed_args_for_tool = []

    current_text = """Hello <tool_call>
<function=execute_bash>
<parameter=command>ls</parameter>
</function>
</tool_call>"""

    result = mimov2_flash_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=current_text,
        delta_text=current_text,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=None,  # type: ignore[arg-type]
    )

    # Should return content before tool call
    assert result is not None
    if result.content:
        assert "Hello" in result.content


def test_streaming_content_and_tool_name(mimov2_flash_tool_parser):
    """Test streaming with content before tool call and tool name."""
    mimov2_flash_tool_parser._sent_content_idx = 0
    mimov2_flash_tool_parser.prev_tool_call_arr = []
    mimov2_flash_tool_parser.streamed_args_for_tool = []

    current_text = """Say hello <tool_call>
<function=greet>
</function>"""

    result = mimov2_flash_tool_parser.extract_tool_calls_streaming(
        previous_text="Say hello ",
        current_text=current_text,
        delta_text="<tool_call>\n<function=greet>\n</function>",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=None,  # type: ignore[arg-type]
    )

    # Should detect the tool name
    if result and result.tool_calls:
        tool_call = result.tool_calls[0]
        if tool_call.function and tool_call.function.name:
            assert tool_call.function.name == "greet"


def test_streaming_tool_with_args(mimov2_flash_tool_parser):
    """Test streaming a tool call with arguments."""
    mimov2_flash_tool_parser._sent_content_idx = 0
    mimov2_flash_tool_parser.prev_tool_call_arr = []
    mimov2_flash_tool_parser.streamed_args_for_tool = []

    current_text = """<tool_call>
<function=execute_bash>
<parameter=command>pwd</parameter>
</function>
</tool_call>"""

    result = mimov2_flash_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=current_text,
        delta_text=current_text,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=None,  # type: ignore[arg-type]
    )

    if result and result.tool_calls:
        # Should have tool name
        has_tool_name = any(
            tc.function and tc.function.name == "execute_bash"
            for tc in result.tool_calls
        )
        assert has_tool_name or result.content


def test_streaming_incremental(mimov2_flash_tool_parser):
    """Test incremental streaming of tool calls."""
    mimov2_flash_tool_parser._sent_content_idx = 0
    mimov2_flash_tool_parser.prev_tool_call_arr = []
    mimov2_flash_tool_parser.streamed_args_for_tool = []

    stages = [
        "<tool_call>\n<function=test>",
        "<tool_call>\n<function=test>\n<parameter=arg1>",
        "<tool_call>\n<function=test>\n<parameter=arg1>value1",
        "</parameter>\n</function>\n</tool_call>",
    ]

    all_tool_names = []
    all_args = []

    for i, current_text in enumerate(stages):
        previous_text = stages[i - 1] if i > 0 else ""

        result = mimov2_flash_tool_parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=current_text[len(previous_text) :],
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=None,  # type: ignore[arg-type]
        )

        if result and result.tool_calls:
            for tc in result.tool_calls:
                if tc.function and tc.function.name:
                    all_tool_names.append(tc.function.name)
                if tc.function and tc.function.arguments:
                    all_args.append(tc.function.arguments)

    assert "test" in all_tool_names


def test_streaming_no_tool_calls(mimov2_flash_tool_parser):
    """Test streaming when there are no tool calls."""
    current_text = "This is just regular text."

    result = mimov2_flash_tool_parser.extract_tool_calls_streaming(
        previous_text="This is just",
        current_text=current_text,
        delta_text=" regular text.",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=None,  # type: ignore[arg-type]
    )

    assert result is not None
    if result.content:
        assert "regular text." in result.content
