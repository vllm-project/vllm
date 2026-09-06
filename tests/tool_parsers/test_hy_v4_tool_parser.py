# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the HYV4 tool call parser."""

import json
from unittest.mock import Mock

import pytest
from transformers import AutoTokenizer

from tests.tool_parsers.common_tests import ToolParserTestConfig, ToolParserTests
from tests.tool_parsers.utils import (
    run_tool_extraction,
    run_tool_extraction_streaming,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.hy_v4_tool_parser import (
    HYV4ToolParser,
    build_tool_extractor,
    detect_token_suffix,
)

STRUCTURAL_TOKENS = [
    "<tool_calls>",
    "</tool_calls>",
    "<tool_call>",
    "</tool_call>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
]


def _tokenizer_with_structural_tokens(suffix: str = "") -> TokenizerLike:
    """gpt2 plus the HYV4 structural tokens, so they tokenize atomically.

    The extractor requires the tool-call tokens in ``get_vocab()`` and the
    streaming path detects markers on token ids, so the shared suite needs a
    real tokenizer whose vocab carries them.
    """
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.add_tokens([t.replace(">", f"{suffix}>") for t in STRUCTURAL_TOKENS])
    return tokenizer


class _FakeVocabTokenizer:
    """Minimal stand-in for ``detect_token_suffix``, which only reads a vocab."""

    def __init__(self, tokens: list[str], init_kwargs: dict | None = None):
        self._tokens = tokens
        self.init_kwargs = init_kwargs or {}

    def get_vocab(self) -> dict[str, int]:
        return {token: i for i, token in enumerate(self._tokens)}


class TestHYV4ToolParser(ToolParserTests):
    @pytest.fixture(scope="class")
    def tokenizer(self) -> TokenizerLike:
        return _tokenizer_with_structural_tokens()

    @pytest.fixture
    def test_config(self) -> ToolParserTestConfig:
        return ToolParserTestConfig(
            parser_name="hy_v4",
            no_tool_calls_output=(
                "How can I help you today? I can check the weather for you."
            ),
            single_tool_call_output=(
                "<tool_calls><tool_call>get_weather\n"
                "<arg_key>city</arg_key><arg_value>Tokyo</arg_value>\n"
                "<arg_key>unit</arg_key><arg_value>celsius</arg_value>\n"
                "</tool_call></tool_calls>"
            ),
            parallel_tool_calls_output=(
                "<tool_calls><tool_call>get_weather\n"
                "<arg_key>city</arg_key><arg_value>Tokyo</arg_value>\n"
                "</tool_call><tool_call>search_hotels\n"
                "<arg_key>location</arg_key><arg_value>Tokyo</arg_value>\n"
                "</tool_call></tool_calls>"
            ),
            various_data_types_output=(
                "<tool_calls><tool_call>test_function\n"
                "<arg_key>string_field</arg_key><arg_value>hello</arg_value>\n"
                "<arg_key>int_field</arg_key><arg_value>42</arg_value>\n"
                "<arg_key>float_field</arg_key><arg_value>3.14</arg_value>\n"
                "<arg_key>bool_field</arg_key><arg_value>true</arg_value>\n"
                "<arg_key>null_field</arg_key><arg_value>null</arg_value>\n"
                '<arg_key>array_field</arg_key><arg_value>["a", "b"]</arg_value>\n'
                "<arg_key>object_field</arg_key>"
                '<arg_value>{"nested": "value"}</arg_value>\n'
                "</tool_call></tool_calls>"
            ),
            empty_arguments_output=(
                "<tool_calls><tool_call>get_current_time</tool_call></tool_calls>"
            ),
            surrounding_text_output=(
                "Let me check the weather for you."
                "<tool_calls><tool_call>get_weather\n"
                "<arg_key>city</arg_key><arg_value>Paris</arg_value>\n"
                "</tool_call></tool_calls>"
            ),
            escaped_strings_output=(
                "<tool_calls><tool_call>send_message\n"
                "<arg_key>text</arg_key>"
                '<arg_value>He said "hello"</arg_value>\n'
                "<arg_key>path</arg_key>"
                "<arg_value>C:\\Users\\file</arg_value>\n"
                "</tool_call></tool_calls>"
            ),
            malformed_input_outputs=[
                # no inner <tool_call> tags -> nothing extracted
                "<tool_calls>get_weather\n"
                "<arg_key>city</arg_key><arg_value>Paris</arg_value>\n"
                "</tool_calls>",
                # tags present, function name absent
                "<tool_calls><tool_call>\n"
                "<arg_key>city</arg_key><arg_value>Paris</arg_value>\n"
                "</tool_call></tool_calls>",
                # unbalanced <tool_call> vs </tool_call>
                "<tool_calls><tool_call>get_weather\n"
                "<arg_key>city</arg_key><arg_value>Paris</arg_value>\n"
                "</tool_calls>",
                # unbalanced <arg_key> vs </arg_key>
                "<tool_calls><tool_call>get_weather\n"
                "<arg_key>city<arg_value>Paris</arg_value>\n"
                "</tool_call></tool_calls>",
            ],
            single_tool_call_expected_name="get_weather",
            single_tool_call_expected_args={"city": "Tokyo", "unit": "celsius"},
            single_tool_call_expected_content=None,
            parallel_tool_calls_count=2,
            parallel_tool_calls_names=["get_weather", "search_hotels"],
            # HYV4 types arguments from the tool's JSON Schema; the shared suite
            # sends no tools, so every value stays a string. Typed parsing is
            # covered by test_arguments_typed_from_tool_schema below.
            supports_typed_arguments=False,
            xfail_streaming={
                "test_malformed_input": (
                    "The shared tool_parser fixture reuses one instance across "
                    "every malformed input. The unbalanced-<tool_call> input "
                    "never closes its call, so the next input streams into "
                    "leaked state and trips the reconstructor's id/index "
                    "assertions. Harness artifact, not a serving path; #51559 "
                    "moves the suite to per-request instances."
                ),
            },
            xfail_nonstreaming={},
        )


@pytest.fixture(scope="module")
def hy_v4_tokenizer() -> TokenizerLike:
    return _tokenizer_with_structural_tokens()


@pytest.fixture
def hy_v4_tool_parser(hy_v4_tokenizer) -> HYV4ToolParser:
    return HYV4ToolParser(hy_v4_tokenizer)


@pytest.fixture
def typed_request() -> ChatCompletionRequest:
    request = Mock(spec=ChatCompletionRequest)
    request.tools = [
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="test_function",
                parameters={
                    "type": "object",
                    "properties": {
                        "count": {"type": "integer"},
                        "ratio": {"type": "number"},
                        "enabled": {"type": "boolean"},
                        "label": {"type": "string"},
                        "items": {"type": "array"},
                    },
                },
            ),
        )
    ]
    request.tool_choice = "auto"
    request.structured_outputs = None
    return request


class TestHYV4SuffixDetection:
    def test_unsuffixed_vocab(self):
        tokenizer = _FakeVocabTokenizer(STRUCTURAL_TOKENS)
        assert detect_token_suffix(tokenizer) == ""

    def test_suffixed_vocab(self):
        suffix = ":6124c78e"
        tokens = [t.replace(">", f"{suffix}>") for t in STRUCTURAL_TOKENS]
        assert detect_token_suffix(tokenizer=_FakeVocabTokenizer(tokens)) == suffix

    def test_no_structural_tokens(self):
        assert detect_token_suffix(_FakeVocabTokenizer(["hello", "world"])) == ""

    def test_transformers_5_model_specific_tokens_rejected(self):
        import transformers

        if int(transformers.__version__.split(".")[0]) < 5:
            pytest.skip("branch only reachable on transformers 5")
        tokenizer = _FakeVocabTokenizer(
            STRUCTURAL_TOKENS,
            init_kwargs={
                "model_specific_special_tokens": {"think_begin_token": "<think>"}
            },
        )
        with pytest.raises(RuntimeError, match="transformers 5 no longer supports"):
            detect_token_suffix(tokenizer)

    def test_missing_tool_call_tokens_raises(self):
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        with pytest.raises(RuntimeError, match="could not locate tool call"):
            build_tool_extractor(tokenizer, strict=True)


class TestHYV4Arguments:
    def test_arguments_typed_from_tool_schema(self, hy_v4_tool_parser, typed_request):
        output = (
            "<tool_calls><tool_call>test_function\n"
            "<arg_key>count</arg_key><arg_value>42</arg_value>\n"
            "<arg_key>ratio</arg_key><arg_value>3.14</arg_value>\n"
            "<arg_key>enabled</arg_key><arg_value>true</arg_value>\n"
            "<arg_key>label</arg_key><arg_value>hello</arg_value>\n"
            '<arg_key>items</arg_key><arg_value>["a", "b"]</arg_value>\n'
            "</tool_call></tool_calls>"
        )
        result = hy_v4_tool_parser.extract_tool_calls(output, request=typed_request)
        assert result.tools_called
        assert json.loads(result.tool_calls[0].function.arguments) == {
            "count": 42,
            "ratio": 3.14,
            "enabled": True,
            "label": "hello",
            "items": ["a", "b"],
        }


class TestHYV4Strictness:
    """The adapter hardcodes strict=True; both extractor paths stay reachable."""

    MALFORMED = (
        "<tool_calls><tool_call>get_weather\n"
        "<arg_key>city</arg_key><arg_value>Tokyo</arg_value>\n"
        "trailing junk\n"
        "</tool_call></tool_calls>"
    )

    def test_strict_rejects_unparsed_payload(self, hy_v4_tokenizer):
        extractor = build_tool_extractor(hy_v4_tokenizer, strict=True)
        result = extractor.extract_tool_calls(self.MALFORMED, None)
        assert not result["tools_called"]
        assert result["tool_calls"] == []
        assert result["content"] == self.MALFORMED

    def test_non_strict_accepts_unparsed_payload(self, hy_v4_tokenizer):
        extractor = build_tool_extractor(hy_v4_tokenizer, strict=False)
        result = extractor.extract_tool_calls(self.MALFORMED, None)
        assert result["tools_called"]
        assert result["tool_calls"][0]["name"] == "get_weather"

    def test_adapter_is_strict(self, hy_v4_tool_parser):
        request = ChatCompletionRequest(messages=[], model="test-model")
        result = hy_v4_tool_parser.extract_tool_calls(self.MALFORMED, request)
        assert not result.tools_called
        assert result.content == self.MALFORMED


class TestHYV4BatchedDelta:
    """One engine delta can carry several complete tool calls (batched or
    speculative decode). The name boundary and the buffer restart must both
    stop at the nearer tag, or the first call's name runs into the second and
    the second call's arguments land on the first."""

    @staticmethod
    def _assert_matches_nonstreaming(parser, output):
        streamed = run_tool_extraction_streaming(
            parser, [output], assert_one_tool_per_delta=False
        ).tool_calls
        _, expected = run_tool_extraction(parser, output, streaming=False)
        assert [
            (tc.function.name, json.loads(tc.function.arguments)) for tc in streamed
        ] == [(tc.function.name, json.loads(tc.function.arguments)) for tc in expected]
        return [tc.function.name for tc in streamed]

    def test_argless_call_then_call_with_args(self, hy_v4_tool_parser):
        output = (
            "<tool_calls><tool_call>get_time</tool_call>"
            "<tool_call>get_weather\n"
            "<arg_key>city</arg_key><arg_value>Tokyo</arg_value>\n"
            "</tool_call></tool_calls>"
        )
        names = self._assert_matches_nonstreaming(hy_v4_tool_parser, output)
        assert names == ["get_time", "get_weather"]

    def test_empty_name_then_call_with_args(self, hy_v4_tool_parser):
        output = (
            "<tool_calls><tool_call></tool_call>"
            "<tool_call>get_weather\n"
            "<arg_key>city</arg_key><arg_value>Tokyo</arg_value>\n"
            "</tool_call></tool_calls>"
        )
        streamed = run_tool_extraction_streaming(
            hy_v4_tool_parser, [output], assert_one_tool_per_delta=False
        ).tool_calls
        assert [tc.function.name for tc in streamed] == ["get_weather"]
        assert json.loads(streamed[0].function.arguments) == {"city": "Tokyo"}


class TestHYV4StreamingState:
    def test_state_is_aliased_to_extractor(self, hy_v4_tool_parser):
        assert hy_v4_tool_parser.prev_tool_call_arr is (
            hy_v4_tool_parser._extractor.prev_tool_call_arr
        )
        assert hy_v4_tool_parser.streamed_args_for_tool is (
            hy_v4_tool_parser._extractor.streamed_args_for_tool
        )

    def test_empty_function_name_is_not_streamed(self, hy_v4_tool_parser):
        """A nameless tool call is dropped, not emitted for the client to call."""
        output = (
            "<tool_calls><tool_call>\n"
            "<arg_key>city</arg_key><arg_value>Paris</arg_value>\n"
            "</tool_call></tool_calls>"
        )
        _, tool_calls = run_tool_extraction(hy_v4_tool_parser, output, streaming=True)
        assert tool_calls == []

    def test_state_populated_after_streaming(self, hy_v4_tool_parser):
        output = (
            "<tool_calls><tool_call>get_weather\n"
            "<arg_key>city</arg_key><arg_value>Tokyo</arg_value>\n"
            "</tool_call></tool_calls>"
        )
        _, tool_calls = run_tool_extraction(hy_v4_tool_parser, output, streaming=True)
        assert len(tool_calls) == 1
        assert hy_v4_tool_parser.prev_tool_call_arr
        assert hy_v4_tool_parser.prev_tool_call_arr[-1]["name"] == "get_weather"
        assert json.loads(hy_v4_tool_parser.streamed_args_for_tool[-1]) == {
            "city": "Tokyo"
        }
