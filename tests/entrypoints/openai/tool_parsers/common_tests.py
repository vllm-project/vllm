# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from tests.entrypoints.openai.tool_parsers.utils import run_tool_extraction
from vllm.entrypoints.openai.tool_parsers import ToolParserManager
from vllm.transformers_utils.tokenizer import AnyTokenizer


@dataclass
class ToolParserTestConfig:
    """Configuration for a tool parser's common tests.

    This dataclass contains all the test data and expected results needed
    to run the common test suite for a parser. Each parser test file
    creates one instance of this config with parser-specific values.

    Attributes:
        parser_name: Name used with ToolParserManager (e.g., "mistral")

        Test data (model outputs):
        no_tool_calls_output: Plain text without any tool syntax
        single_tool_call_output: One tool call with simple arguments
        parallel_tool_calls_output: Multiple tool calls in one response
        various_data_types_output: Tool with various data types
        empty_arguments_output: Tool call with no parameters
        surrounding_text_output: Tool call mixed with regular text
        escaped_strings_output: Tool call with escaped chars
        malformed_input_outputs: List of invalid inputs

        Expected results:
        single_tool_call_expected_name: Expected function name
        single_tool_call_expected_args: Expected arguments dict
        parallel_tool_calls_count: Number of tools in parallel test
        parallel_tool_calls_names: Function names in order
        single_tool_call_expected_content: Content field when tool called
        parallel_tool_calls_expected_content: Content for parallel test

        xfail markers:
        xfail_streaming: Mapping test name to xfail reason (streaming only)
        xfail_nonstreaming: Mapping test name to xfail reason (non-streaming)

        Special flags:
        allow_empty_or_json_empty_args: True if "" or "{}" both valid for empty args
    """

    # Parser identification
    parser_name: str

    # Test data - model outputs for each common test
    no_tool_calls_output: str
    single_tool_call_output: str
    parallel_tool_calls_output: str
    various_data_types_output: str
    empty_arguments_output: str
    surrounding_text_output: str
    escaped_strings_output: str
    malformed_input_outputs: list[str]

    # Expected results for specific tests (optional overrides)
    single_tool_call_expected_name: str = "get_weather"
    single_tool_call_expected_args: dict[str, Any] = field(
        default_factory=lambda: {"city": "Tokyo"}
    )
    parallel_tool_calls_count: int = 2
    parallel_tool_calls_names: list[str] = field(
        default_factory=lambda: ["get_weather", "get_time"]
    )

    # xfail configuration - maps test name to xfail reason
    xfail_streaming: dict[str, str] = field(default_factory=dict)
    xfail_nonstreaming: dict[str, str] = field(default_factory=dict)

    # Content expectations (some parsers strip content, others don't)
    single_tool_call_expected_content: str | None = None
    parallel_tool_calls_expected_content: str | None = None

    # Special assertions for edge cases
    allow_empty_or_json_empty_args: bool = True  # "{}" or "" for empty args


class ToolParserTests:
    """Mixin class providing common test suite for tool parsers.

    To use this mixin in a parser test file:

    1. Create a test_config fixture that returns a ToolParserTestConfig instance
    2. Inherit from this class
    3. Add parser-specific tests as additional methods

    Example:
        class TestMistralToolParser(ToolParserTests):
            @pytest.fixture
            def test_config(self) -> ToolParserTestConfig:
                return ToolParserTestConfig(
                    parser_name="mistral",
                    no_tool_calls_output="Plain text...",
                    # ... other config ...
                )

            # Parser-specific tests
            def test_mistral_specific_feature(self, tool_parser):
                # Custom test logic
                pass
    """

    @pytest.fixture
    def test_config(self) -> ToolParserTestConfig:
        """Override this to provide parser-specific configuration."""
        raise NotImplementedError(
            "Subclass must provide test_config fixture returning ToolParserTestConfig"
        )

    @pytest.fixture
    def tokenizer(self, default_tokenizer: AnyTokenizer) -> AnyTokenizer:
        """Override this to provide parser-specific tokenizer."""
        return default_tokenizer

    @pytest.fixture
    def tool_parser(self, test_config: ToolParserTestConfig, tokenizer: AnyTokenizer):
        return ToolParserManager.get_tool_parser(test_config.parser_name)(tokenizer)

    @pytest.fixture(params=[True, False])
    def streaming(self, request: pytest.FixtureRequest) -> bool:
        return request.param

    def test_no_tool_calls(
        self,
        request: pytest.FixtureRequest,
        tool_parser: Any,
        test_config: ToolParserTestConfig,
        streaming: bool,
    ):
        """Verify parser handles plain text without tool syntax."""
        # Apply xfail markers if configured
        test_name = "test_no_tool_calls"
        self.apply_xfail_mark(request, test_config, test_name, streaming)

        content, tool_calls = run_tool_extraction(
            tool_parser, test_config.no_tool_calls_output, streaming=streaming
        )
        assert content == test_config.no_tool_calls_output, (
            f"Expected content to match input, got {content}"
        )
        assert len(tool_calls) == 0, f"Expected no tool calls, got {len(tool_calls)}"

    def test_single_tool_call_simple_args(
        self,
        request: pytest.FixtureRequest,
        tool_parser: Any,
        test_config: ToolParserTestConfig,
        streaming: bool,
    ):
        """Verify parser extracts one tool with simple arguments."""
        # Apply xfail markers if configured
        test_name = "test_single_tool_call_simple_args"
        self.apply_xfail_mark(request, test_config, test_name, streaming)

        content, tool_calls = run_tool_extraction(
            tool_parser, test_config.single_tool_call_output, streaming=streaming
        )

        # Content check (some parsers strip it)
        if test_config.single_tool_call_expected_content is not None:
            assert content == test_config.single_tool_call_expected_content

        assert len(tool_calls) == 1, f"Expected 1 tool call, got {len(tool_calls)}"
        assert tool_calls[0].type == "function"
        assert tool_calls[0].function.name == test_config.single_tool_call_expected_name

        args = json.loads(tool_calls[0].function.arguments)
        for key, value in test_config.single_tool_call_expected_args.items():
            assert args.get(key) == value, (
                f"Expected {key}={value}, got {args.get(key)}"
            )

    def test_parallel_tool_calls(
        self,
        request: pytest.FixtureRequest,
        tool_parser: Any,
        test_config: ToolParserTestConfig,
        streaming: bool,
    ):
        """Verify parser handles multiple tools in one response."""
        # Apply xfail markers if configured
        test_name = "test_parallel_tool_calls"
        self.apply_xfail_mark(request, test_config, test_name, streaming)

        content, tool_calls = run_tool_extraction(
            tool_parser,
            test_config.parallel_tool_calls_output,
            streaming=streaming,
        )

        assert len(tool_calls) == test_config.parallel_tool_calls_count, (
            f"Expected {test_config.parallel_tool_calls_count} "
            f"tool calls, got {len(tool_calls)}"
        )

        # Verify tool names match expected
        for i, expected_name in enumerate(test_config.parallel_tool_calls_names):
            assert tool_calls[i].type == "function"
            assert tool_calls[i].function.name == expected_name

        # Verify unique IDs
        ids = [tc.id for tc in tool_calls]
        assert len(ids) == len(set(ids)), "Tool call IDs should be unique"

    def test_various_data_types(
        self,
        request: pytest.FixtureRequest,
        tool_parser: Any,
        test_config: ToolParserTestConfig,
        streaming: bool,
    ):
        """Verify parser handles all JSON types in arguments."""
        # Apply xfail markers if configured
        test_name = "test_various_data_types"
        self.apply_xfail_mark(request, test_config, test_name, streaming)

        content, tool_calls = run_tool_extraction(
            tool_parser,
            test_config.various_data_types_output,
            streaming=streaming,
        )
        assert len(tool_calls) == 1, f"Expected 1 tool call, got {len(tool_calls)}"

        args = json.loads(tool_calls[0].function.arguments)
        # Verify all expected fields present
        required_fields = [
            "string_field",
            "int_field",
            "float_field",
            "bool_field",
            "null_field",
            "array_field",
            "object_field",
        ]
        for required_field in required_fields:
            assert required_field in args, (
                f"Expected field '{required_field}' in arguments"
            )

    def test_empty_arguments(
        self,
        request: pytest.FixtureRequest,
        tool_parser: Any,
        test_config: ToolParserTestConfig,
        streaming: bool,
    ):
        """Verify parser handles parameterless tool calls."""
        # Apply xfail markers if configured
        test_name = "test_empty_arguments"
        self.apply_xfail_mark(request, test_config, test_name, streaming)

        content, tool_calls = run_tool_extraction(
            tool_parser, test_config.empty_arguments_output, streaming=streaming
        )
        assert len(tool_calls) == 1, f"Expected 1 tool call, got {len(tool_calls)}"

        args = tool_calls[0].function.arguments
        if test_config.allow_empty_or_json_empty_args:
            assert args in ["{}", ""], f"Expected empty args, got {args}"
        else:
            assert args == "{}", f"Expected {{}}, got {args}"

    def test_surrounding_text(
        self,
        request: pytest.FixtureRequest,
        tool_parser: Any,
        test_config: ToolParserTestConfig,
        streaming: bool,
    ):
        """Verify parser extracts tools from mixed content."""
        # Apply xfail markers if configured
        test_name = "test_surrounding_text"
        self.apply_xfail_mark(request, test_config, test_name, streaming)

        content, tool_calls = run_tool_extraction(
            tool_parser, test_config.surrounding_text_output, streaming=streaming
        )
        assert len(tool_calls) >= 1, (
            f"Expected at least 1 tool call, got {len(tool_calls)}"
        )

    def test_escaped_strings(
        self,
        request: pytest.FixtureRequest,
        tool_parser: Any,
        test_config: ToolParserTestConfig,
        streaming: bool,
    ):
        """Verify parser handles escaped characters in arguments."""
        # Apply xfail markers if configured
        test_name = "test_escaped_strings"
        self.apply_xfail_mark(request, test_config, test_name, streaming)

        content, tool_calls = run_tool_extraction(
            tool_parser, test_config.escaped_strings_output, streaming=streaming
        )
        assert len(tool_calls) == 1, f"Expected 1 tool call, got {len(tool_calls)}"

        args = json.loads(tool_calls[0].function.arguments)
        # At minimum, verify we can parse and have expected fields
        # Exact escaping behavior varies by parser
        assert len(args) > 0, "Expected some arguments with escaped strings"

    def test_malformed_input(
        self,
        request: pytest.FixtureRequest,
        tool_parser: Any,
        test_config: ToolParserTestConfig,
        streaming: bool,
    ):
        """Verify parser gracefully handles invalid syntax."""
        # Apply xfail markers if configured
        test_name = "test_malformed_input"
        self.apply_xfail_mark(request, test_config, test_name, streaming)

        for malformed_input in test_config.malformed_input_outputs:
            # Should not raise exception
            content, tool_calls = run_tool_extraction(
                tool_parser, malformed_input, streaming=streaming
            )
            # Parser should handle gracefully (exact behavior varies)

    def test_streaming_reconstruction(
        self,
        request: pytest.FixtureRequest,
        tool_parser: Any,
        test_config: ToolParserTestConfig,
    ):
        """Verify streaming produces same result as non-streaming."""
        test_name = "test_streaming_reconstruction"
        self.apply_xfail_mark(request, test_config, test_name, True)

        test_output = test_config.single_tool_call_output

        # Non-streaming result
        content_non, tools_non = run_tool_extraction(
            tool_parser, test_output, streaming=False
        )

        # Streaming result
        content_stream, tools_stream = run_tool_extraction(
            tool_parser, test_output, streaming=True
        )

        # Compare results
        assert content_non == content_stream, "Content should match between modes"
        assert len(tools_non) == len(tools_stream), "Tool count should match"
        if len(tools_non) > 0:
            assert tools_non[0].function.name == tools_stream[0].function.name
            assert tools_non[0].function.arguments == tools_stream[0].function.arguments

    def apply_xfail_mark(self, request, test_config, test_name, streaming):
        reason = None
        if streaming and test_name in test_config.xfail_streaming:
            reason = test_config.xfail_streaming[test_name]
        elif not streaming and test_name in test_config.xfail_nonstreaming:
            reason = test_config.xfail_nonstreaming[test_name]
        if reason is not None:
            mark = pytest.mark.xfail(reason=reason, strict=True)
            request.node.add_marker(mark)
