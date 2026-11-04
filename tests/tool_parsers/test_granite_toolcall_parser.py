# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from tests.tool_parsers.common_tests import ToolParserTestConfig, ToolParserTests
from tests.tool_parsers.utils import run_tool_extraction
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tokenizers import TokenizerLike, get_tokenizer
from vllm.tool_parsers import ToolParserManager

MODEL = "ibm-granite/granite-4.2-8b"


def _tool_call(name: str, params: dict[str, str]) -> str:
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


# Granite 4.2 uses its own tokenizer (not gpt2) so special-token boundaries
# match production. get_tokenizer fetches only the small tokenizer files, not
# the model weights.
@pytest.fixture(scope="module")
def granite_tokenizer() -> TokenizerLike:
    return get_tokenizer(tokenizer_name=MODEL)


class TestGraniteToolCallParser(ToolParserTests):
    @pytest.fixture
    def test_config(self) -> ToolParserTestConfig:
        return ToolParserTestConfig(
            parser_name="granite_toolcall_parser",
            no_tool_calls_output="This is a regular response without any tool calls.",
            single_tool_call_output=_tool_call("get_weather", {"city": "Tokyo"}),
            parallel_tool_calls_output=(
                _tool_call("get_weather", {"city": "Tokyo"})
                + "\n"
                + _tool_call("get_time", {"timezone": "Asia/Tokyo"})
            ),
            various_data_types_output=_tool_call(
                "test_function",
                {
                    "string_field": "hello",
                    "int_field": "42",
                    "float_field": "3.14",
                    "bool_field": "true",
                    "null_field": "null",
                    "array_field": '["a", "b", "c"]',
                    "object_field": '{"nested": "value"}',
                },
            ),
            empty_arguments_output="<tool_call>\n<function=refresh>\n</function>\n</tool_call>",
            surrounding_text_output=(
                "Let me check the weather for you."
                + _tool_call("get_weather", {"city": "Tokyo"})
            ),
            escaped_strings_output=_tool_call(
                "test_function",
                {
                    "quoted": 'He said "hello"',
                    "path": "C:\\Users\\file.txt",
                    "newline": "line1\nline2",
                },
            ),
            malformed_input_outputs=[
                # bare opener, never completed
                "<tool_call>\n<function=",
                # stray closing tags with no opener
                "</parameter>\n</function>\n</tool_call>",
                # tool-call markup embedded in plain prose, no function tag
                "Here is some text mentioning <tool_call> inline but not a call.",
            ],
            single_tool_call_expected_name="get_weather",
            single_tool_call_expected_args={"city": "Tokyo"},
            single_tool_call_expected_content=None,
            parallel_tool_calls_count=2,
            parallel_tool_calls_names=["get_weather", "get_time"],
            # Values are typed only when the request carries the tool's JSON
            # schema; the shared harness builds the parser with no tools, so
            # every <parameter> value stays a string (same as qwen3_coder).
            supports_typed_arguments=False,
        )

    # Granite 4.2 uses its own tokenizer (not gpt2) so special-token
    # boundaries match production. get_tokenizer fetches only the small
    # tokenizer files, not the model weights.
    @pytest.fixture
    def tokenizer(self, granite_tokenizer: TokenizerLike) -> TokenizerLike:
        return granite_tokenizer

    # --- Granite-specific tests ---

    @pytest.mark.parametrize("streaming", [True, False])
    def test_leading_newline_content_preserved(self, tokenizer, streaming):
        """A standalone tool parser must leave content untouched, matching
        qwen3_coder. Regression guard: an earlier version subclassed the
        reasoning parser and stripped the leading newline off plain content.
        """
        model_output = "\n\nHello, how can I help you today?"
        granite = ToolParserManager.get_tool_parser("granite_toolcall_parser")(
            tokenizer
        )
        qwen = ToolParserManager.get_tool_parser("qwen3_coder")(tokenizer)
        g_content, g_calls = run_tool_extraction(
            granite, model_output, streaming=streaming
        )
        q_content, q_calls = run_tool_extraction(
            qwen, model_output, streaming=streaming
        )
        assert g_content == q_content == model_output
        assert g_calls == q_calls == []

    def test_works_without_reasoning_parser(self, tokenizer):
        """Reasoning markup with no reasoning parser configured: the tool call
        is still extracted and the <think> block stays in content verbatim.
        """
        model_output = "<think>\nThe user wants the weather.\n</think>\n" + _tool_call(
            "get_weather", {"city": "Tokyo"}
        )
        parser = ToolParserManager.get_tool_parser("granite_toolcall_parser")(tokenizer)
        request = ChatCompletionRequest(model="test-model", messages=[])
        extracted = parser.extract_tool_calls(model_output, request=request)
        assert extracted.tools_called
        assert len(extracted.tool_calls) == 1
        assert extracted.tool_calls[0].function.name == "get_weather"
        assert json.loads(extracted.tool_calls[0].function.arguments) == {
            "city": "Tokyo"
        }
        assert "<think>" in (extracted.content or "")
