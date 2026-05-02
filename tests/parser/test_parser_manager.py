# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.parser.abstract_parser import _WrappedParser
from vllm.parser.minimax_m2_parser import MiniMaxM2Parser
from vllm.parser.mistral_parser import MistralParser
from vllm.parser.parser_manager import ParserManager
from vllm.reasoning.minimax_m2_reasoning_parser import MiniMaxM2ReasoningParser
from vllm.reasoning.mistral_reasoning_parser import MistralReasoningParser
from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser
from vllm.tool_parsers.minimax_m2_tool_parser import MinimaxM2ToolParser
from vllm.tool_parsers.mistral_tool_parser import MistralToolParser


@pytest.fixture(autouse=True)
def reset_parser_classes():
    MistralParser.reasoning_parser_cls = None
    _WrappedParser.reasoning_parser_cls = None
    _WrappedParser.tool_parser_cls = None


@pytest.mark.parametrize(
    (
        "reasoning_parser_name",
        "tool_parser_name",
        "expected_parser_cls",
        "expected_reasoning_parser_cls",
        "expected_tool_parser_cls",
    ),
    [
        pytest.param(
            "minimax_m2",
            "minimax_m2",
            MiniMaxM2Parser,
            MiniMaxM2ReasoningParser,
            MinimaxM2ToolParser,
            id="minimax_m2_parser",
        ),
        pytest.param(
            "mistral",
            "mistral",
            MistralParser,
            MistralReasoningParser,
            MistralToolParser,
            id="mistral_parser_with_mistral_reasoning",
        ),
        pytest.param(
            "qwen3",
            "mistral",
            MistralParser,
            Qwen3ReasoningParser,
            MistralToolParser,
            id="mistral_parser_with_other_reasoning",
        ),
        pytest.param(
            "mistral",
            "minimax_m2",
            _WrappedParser,
            MistralReasoningParser,
            MinimaxM2ToolParser,
            id="wrapped_parser_fallback",
        ),
    ],
)
def test_get_parser(
    reasoning_parser_name: str,
    tool_parser_name: str,
    expected_parser_cls: type,
    expected_reasoning_parser_cls: type,
    expected_tool_parser_cls: type,
):
    parser_cls = ParserManager.get_parser(
        tool_parser_name=tool_parser_name,
        reasoning_parser_name=reasoning_parser_name,
        enable_auto_tools=True,
        model_name="test-model",
    )

    assert parser_cls is expected_parser_cls
    assert parser_cls.reasoning_parser_cls is expected_reasoning_parser_cls
    assert parser_cls.tool_parser_cls is expected_tool_parser_cls
