# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.parser.abstract_parser import DelegatingParser
from vllm.parser.minimax_m2_parser import MiniMaxM2Parser
from vllm.parser.mistral_parser import MistralParser
from vllm.parser.parser_manager import ParserManager


@pytest.mark.parametrize(
    (
        "reasoning_parser_name",
        "tool_parser_name",
        "expected_parser_cls",
        "enable_auto_tools",
    ),
    [
        pytest.param(
            "minimax_m2",
            "minimax_m2",
            MiniMaxM2Parser,
            True,
            id="minimax_m2_parser",
        ),
        pytest.param(
            "mistral",
            "mistral",
            MistralParser,
            True,
            id="mistral_parser_with_mistral_reasoning",
        ),
        pytest.param(
            "qwen3",
            "mistral",
            MistralParser,
            True,
            id="mistral_parser_with_other_reasoning",
        ),
        pytest.param(
            "mistral",
            "minimax_m2",
            DelegatingParser,
            True,
            id="delegating_parser_fallback",
        ),
        pytest.param(
            None,
            "minimax_m2",
            DelegatingParser,
            True,
            id="delegating_parser_fallback_no_reasoning",
        ),
        pytest.param(
            None,
            "mistral",
            MistralParser,
            True,
            id="mistral_parser_no_reasoning",
        ),
        pytest.param(
            "mistral",
            "mistral",
            DelegatingParser,
            False,
            id="mistral_parser_fallback_no_auto_tools",
        ),
    ],
)
def test_get_parser(
    reasoning_parser_name: str | None,
    tool_parser_name: str | None,
    expected_parser_cls: type,
    enable_auto_tools: bool,
):
    parser_cls = ParserManager.get_parser(
        tool_parser_name=tool_parser_name,
        reasoning_parser_name=reasoning_parser_name,
        enable_auto_tools=enable_auto_tools,
        model_name="test-model",
    )

    assert issubclass(parser_cls, expected_parser_cls)

    expected_reasoning_parser_cls = ParserManager.get_reasoning_parser(
        reasoning_parser_name
    )
    if expected_reasoning_parser_cls is not None:
        assert issubclass(
            parser_cls.reasoning_parser_cls, expected_reasoning_parser_cls
        )
    else:
        assert parser_cls.reasoning_parser_cls is None

    expected_tool_parser_cls = ParserManager.get_tool_parser(
        tool_parser_name=tool_parser_name,
        enable_auto_tools=enable_auto_tools,
    )
    if expected_tool_parser_cls is not None:
        assert issubclass(parser_cls.tool_parser_cls, expected_tool_parser_cls)
    else:
        assert parser_cls.tool_parser_cls is None
