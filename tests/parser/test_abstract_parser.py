# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from xgrammar import Grammar
from xgrammar.testing import _is_grammar_accept_string

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.parser.abstract_parser import DelegatingParser
from vllm.tool_parsers.abstract_tool_parser import ToolParser
from vllm.tool_parsers.qwen3_engine_tool_parser import Qwen3EngineToolParser
from vllm.tool_parsers.structural_tag_registry import ToolChoice


class TestToolChoice_Plus_ResponseFormat:
    """Note(arpera):
    Test cases for tool_choice={auto,required} + response_format
    To keep it short:
    DelegatingParser.adjust_request behavior in some corner cases is checked there

    Initial bug report:
    https://github.com/vllm-project/vllm/issues/39929
    And PR that fixed this:
    https://github.com/vllm-project/vllm/pull/56086
    """

    # ================================
    # Helper methods
    # ================================

    @staticmethod
    def _tools(strict: bool) -> list[ChatCompletionToolsParam]:
        """Single get_weather tool, optionally marked as strict"""
        function: dict[str, Any] = {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
        if strict:
            function["strict"] = True
        return [ChatCompletionToolsParam(type="function", function=function)]

    @staticmethod
    def _qwen_tool_call() -> str:
        """Tool call for Qwen model that do support structural tag"""
        return (
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n"
            "</parameter>\n</function>\n</tool_call>"
        )

    @staticmethod
    def _json_schema_response_format() -> dict:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        }

    @staticmethod
    def _setup_request(
        tools: list[ChatCompletionToolsParam],
        tool_choice: ToolChoice,
        response_format: dict,
    ) -> ChatCompletionRequest:
        request = ChatCompletionRequest(
            messages=[],  # for our test cases it's always empty
            model="m",  # Just a placeholder, don't pay much attention
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
        )
        return request

    @staticmethod
    def _setup_abstract_parser(
        tools: list[ChatCompletionToolsParam],
    ) -> DelegatingParser:
        """Construct parser that does NOT support structural tag"""

        class TestParser(DelegatingParser):
            tool_parser_cls = ToolParser

        return TestParser(MagicMock(), tools=tools)

    @staticmethod
    def _setup_qwen_parser(
        tools: list[ChatCompletionToolsParam],
    ) -> DelegatingParser:
        """Construct parser that supports structural tag"""

        class TestParser(DelegatingParser):
            tool_parser_cls = Qwen3EngineToolParser

        return TestParser(MagicMock(), tools=tools)

    # ================================
    # Test cases
    # tool_choice=auto + response_format
    # ================================

    @pytest.mark.parametrize(
        # In this test we check that for response_format
        # resulting grammar accepts @compliant_output and rejects @non_compliant_output
        # You can add more examples here if you see some corner cases not covered
        ("response_format", "compliant_output", "non_compliant_output"),
        [
            (_json_schema_response_format(), '{"text": "hi"}', '{"foo": 1}'),
            ({"type": "json_object"}, '{"any": 1}', "[1, 2]"),
        ],
        # We test here two different response_format types:
        ids=["json_schema", "json_object"],
    )
    def test_auto_with_strict_tools(
        self,
        response_format: dict,
        compliant_output: str,
        non_compliant_output: str,
    ):
        tools = self._tools(strict=True)
        request = self._setup_request(
            tools=tools,
            tool_choice="auto",
            response_format=response_format,
        )
        parser = self._setup_qwen_parser(tools)
        out = parser.adjust_request(request)

        # Now check that request does not have response_format anymore
        # but instead has structured_outputs set as structural tag
        # And this structural tag is OR operation
        assert out.tool_choice == "auto"
        assert out.response_format is None
        assert out.structured_outputs is not None
        tag = json.loads(out.structured_outputs.structural_tag)
        assert tag["format"]["type"] == "or"
        grammar = Grammar.from_structural_tag(out.structured_outputs.structural_tag)

        assert _is_grammar_accept_string(grammar, compliant_output)
        assert not _is_grammar_accept_string(grammar, non_compliant_output)

        # Also tool call must be accepted by grammar
        assert _is_grammar_accept_string(grammar, self._qwen_tool_call())

        # IMPORTANT(arpera): Regression test
        # If we in adjust_request implementation by mistake
        # construct structural tag using tool_choice=auto
        # then such a structural tag would allow plain text as well
        # We need to be sure that plain text is NOT accepted in our case
        assert not _is_grammar_accept_string(grammar, "Hello")

    def test_auto_without_strict_tools(self):
        tools = self._tools(strict=False)
        request = self._setup_request(
            tools=tools,
            tool_choice="auto",
            response_format=self._json_schema_response_format(),
        )
        parser = self._setup_qwen_parser(tools)

        # There must be a warning that tool calls are disabled
        # Consume that warning
        with patch("vllm.parser.abstract_parser.logger.warning_once") as mock_warn:
            out = parser.adjust_request(request)

        assert out.response_format is not None
        assert out.structured_outputs is None
        mock_warn.assert_called_once()

    def test_when_model_does_not_have_structural_tag(self):
        """Note(arpera):
        When model does NOT have structural tag support
        we apply constraint only for response_format
        """
        tools = self._tools(strict=True)
        request = self._setup_request(
            tools=tools,
            tool_choice="auto",
            response_format=self._json_schema_response_format(),
        )
        # SIC! use parser whose model does NOT support structural tag
        parser = self._setup_abstract_parser(tools)

        with patch("vllm.parser.abstract_parser.logger.warning_once") as mock_warn:
            out = parser.adjust_request(request)

        assert out.response_format is not None
        assert out.structured_outputs is None
        mock_warn.assert_called_once()

    # ================================
    # Test cases
    # tool_choice=required + response_format
    # ================================

    def test_required(self):
        tools = self._tools(strict=True)
        request = self._setup_request(
            tools=tools,
            tool_choice="required",
            response_format=self._json_schema_response_format(),
        )
        parser = self._setup_qwen_parser(tools)

        with patch("vllm.parser.abstract_parser.logger.warning_once") as mock_warn:
            out = parser.adjust_request(request)

        assert out.response_format is None
        assert out.structured_outputs is not None
        grammar = Grammar.from_structural_tag(out.structured_outputs.structural_tag)
        assert _is_grammar_accept_string(grammar, self._qwen_tool_call())
        assert not _is_grammar_accept_string(grammar, '{"text": "hi"}')
        mock_warn.assert_called_once()
