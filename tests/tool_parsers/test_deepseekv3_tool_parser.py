# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest

from tests.tool_parsers.common_tests import (
    ToolParserTestConfig,
    ToolParserTests,
)
from vllm.tokenizers import TokenizerLike, get_tokenizer


class TestDeepSeekV3ToolParser(ToolParserTests):
    @pytest.fixture(scope="class")
    def tokenizer(self) -> TokenizerLike:
        return get_tokenizer("deepseek-ai/DeepSeek-V3")

    @pytest.fixture
    def test_config(self) -> ToolParserTestConfig:
        return ToolParserTestConfig(
            parser_name="deepseek_v3",
            # Test data
            no_tool_calls_output=(
                "How can I help you today? I can check weather for you."
            ),
            single_tool_call_output="""<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
```json
{"city": "Tokyo", "unit": "celsius"}
```<｜tool▁call▁end｜><｜tool▁calls▁end｜>""",
            parallel_tool_calls_output="""<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
```json
{"city": "Tokyo", "unit": "celsius"}
```<｜tool▁call▁end｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>search_hotels
```json
{"location": "Tokyo", "check_in": "2025-01-15"}
```<｜tool▁call▁end｜><｜tool▁calls▁end｜>""",
            various_data_types_output=(
                """<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>test_function
```json
"""
                """{"string_field": "hello", "int_field": 42, "float_field": 3.14, """
                """"bool_field": true, "null_field": null, """
                """"array_field": ["a", "b", "c"], """
                """"object_field": {"nested": "value"}, """
                """"empty_array": [], "empty_object": {}}
```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"""
            ),
            empty_arguments_output="""<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_current_time
```json
{}
```<｜tool▁call▁end｜><｜tool▁calls▁end｜>""",
            surrounding_text_output=(
                """Let me check the weather for you."""
                """<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
```json
{"city": "Paris"}
```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"""
            ),
            escaped_strings_output=(
                """<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>send_message
```json
"""
                """{"text": "He said \\"hello\\"", "path": "C:\\\\Users\\\\file", """
                """"newline": "line1\\nline2"}
```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"""
            ),
            malformed_input_outputs=[
                """<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
```json
{"city": "Tokyo"
```<｜tool▁call▁end｜><｜tool▁calls▁end｜>""",
                """<｜tool▁calls▁begin｜>function<｜tool▁sep｜>get_weather
```json
{"city": "Tokyo"}
```<｜tool▁calls▁end｜>""",
            ],
            # Expected results
            single_tool_call_expected_name="get_weather",
            single_tool_call_expected_args={"city": "Tokyo", "unit": "celsius"},
            single_tool_call_expected_content=None,
            parallel_tool_calls_count=2,
            parallel_tool_calls_names=["get_weather", "search_hotels"],
            # xfail markers
            xfail_streaming={},
            xfail_nonstreaming={
                "test_malformed_input": (
                    "Parser sets tools_called=True even when tool_calls is "
                    "empty (detects start token but fails to parse)"
                ),
            },
        )


TOOL_CALLS_BEGIN = "<｜tool▁calls▁begin｜>"
TOOL_CALLS_END = "<｜tool▁calls▁end｜>"
TOOL_CALL_BEGIN = "<｜tool▁call▁begin｜>"
TOOL_CALL_END = "<｜tool▁call▁end｜>"
TOOL_SEP = "<｜tool▁sep｜>"


@pytest.fixture(scope="module")
def deepseekv3_tokenizer():
    return get_tokenizer("deepseek-ai/DeepSeek-V3")


def _stream_deltas(tokenizer, deltas):
    """Drive the streaming parser with explicit multi-token text deltas,
    as produced by async scheduling / stream_interval > 1."""
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.tool_parsers.deepseekv3_tool_parser import DeepSeekV3ToolParser

    parser = DeepSeekV3ToolParser(tokenizer)
    request = ChatCompletionRequest(messages=[], model="test-model")

    name = None
    args = ""
    previous_text = ""
    previous_ids: list[int] = []
    for delta_text in deltas:
        delta_ids = tokenizer.encode(delta_text, add_special_tokens=False)
        current_text = previous_text + delta_text
        current_ids = previous_ids + delta_ids
        delta_message = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=previous_ids,
            current_token_ids=current_ids,
            delta_token_ids=delta_ids,
            request=request,
        )
        if delta_message is not None:
            for tool_call in delta_message.tool_calls:
                if tool_call.function:
                    if tool_call.function.name:
                        name = tool_call.function.name
                    if tool_call.function.arguments:
                        args += tool_call.function.arguments
        previous_text = current_text
        previous_ids = current_ids
    return name, args


@pytest.mark.parametrize(
    "final_args_deltas,expected_args",
    [
        # arguments ending with a number: the final characters and the
        # tool-call end token arrive in the same delta
        (['{"code": 1', "23}\n```" + TOOL_CALL_END], '{"code": 123}'),
        # arguments ending with a nested object
        (
            ['{"a": {"b": "x', '"}}\n```' + TOOL_CALL_END],
            '{"a": {"b": "x"}}',
        ),
        # arguments ending with a boolean
        (['{"flag": ', "true}\n```" + TOOL_CALL_END], '{"flag": true}'),
        # control: arguments ending with a quoted string (worked before)
        (['{"city": "Tok', 'yo"}\n```' + TOOL_CALL_END], '{"city": "Tokyo"}'),
    ],
    ids=["number_tail", "nested_object_tail", "boolean_tail", "string_tail"],
)
def test_streaming_final_args_chunk_shares_delta_with_end_token(
    deepseekv3_tokenizer, final_args_deltas, expected_args
):
    """The last characters of the arguments arriving in the same delta as
    the tool-call end token must not be dropped, regardless of what
    character the arguments end with.

    Regression: the closing branch reconstructed the unstreamed tail with
    a `"}`-based heuristic, dropping the tail entirely for arguments
    ending in a number/boolean/null and truncating nested objects.
    """
    deltas = [
        TOOL_CALLS_BEGIN + TOOL_CALL_BEGIN + "function" + TOOL_SEP + "get_code\n"
        "```json\n",
        *final_args_deltas,
        TOOL_CALLS_END,
    ]

    name, args = _stream_deltas(deepseekv3_tokenizer, deltas)

    assert name == "get_code"
    assert args == expected_args
