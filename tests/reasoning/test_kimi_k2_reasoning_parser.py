# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Tests for KimiK2ReasoningParser.

The Kimi K2 model uses <think>...</think> tokens like DeepSeek R1, but
sometimes outputs tool calls without a proper </think> delimiter. This
parser handles that case by detecting tool call markers and splitting
the output at the tool marker boundary.
"""

import pytest
from transformers import AutoTokenizer

from tests.reasoning.utils import run_reasoning_extraction
from vllm.reasoning import ReasoningParser, ReasoningParserManager

parser_name = "kimi_k2"
start_token = "<think>"
end_token = "</think>"

# Use DeepSeek R1 tokenizer since Kimi K2 uses similar tokens
REASONING_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


@pytest.fixture(scope="module")
def kimi_k2_tokenizer():
    return AutoTokenizer.from_pretrained(REASONING_MODEL_NAME)


# Standard reasoning cases (inherited behavior from DeepSeek R1)
SIMPLE_REASONING = {
    "output": "This is a reasoning section</think>This is the rest",
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
}
COMPLETE_REASONING = {
    "output": "This is a reasoning section</think>",
    "reasoning": "This is a reasoning section",
    "content": None,
}
REASONING_WITH_THINK = {
    "output": "<think>This is a reasoning section</think>This is the rest",
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
}

# Kimi K2 specific: Tool call markers without </think>
TOOL_SECTION_NO_THINK_END = {
    "output": (
        "Let me check the weather"
        "<|tool_calls_section_begin|><|tool_call_begin|>"
        "get_weather:0<|tool_call_argument_begin|>{}"
        "<|tool_call_end|><|tool_calls_section_end|>"
    ),
    "reasoning": "Let me check the weather",
    "content": (
        "<|tool_calls_section_begin|><|tool_call_begin|>"
        "get_weather:0<|tool_call_argument_begin|>{}"
        "<|tool_call_end|><|tool_calls_section_end|>"
    ),
}
TOOL_CALL_NO_SECTION_NO_THINK_END = {
    "output": (
        "I'll help you with that"
        "<|tool_call_begin|>get_weather:0<|tool_call_argument_begin|>{}"
        "<|tool_call_end|>"
    ),
    "reasoning": "I'll help you with that",
    "content": (
        "<|tool_call_begin|>get_weather:0<|tool_call_argument_begin|>{}"
        "<|tool_call_end|>"
    ),
}
TOOL_SECTION_WITH_THINK = {
    "output": (
        "<think>Let me think about this</think>"
        "<|tool_calls_section_begin|><|tool_call_begin|>"
        "get_weather:0<|tool_call_argument_begin|>{}"
        "<|tool_call_end|><|tool_calls_section_end|>"
    ),
    "reasoning": "Let me think about this",
    "content": (
        "<|tool_calls_section_begin|><|tool_call_begin|>"
        "get_weather:0<|tool_call_argument_begin|>{}"
        "<|tool_call_end|><|tool_calls_section_end|>"
    ),
}
EMPTY_REASONING_WITH_TOOL = {
    "output": (
        "<|tool_calls_section_begin|><|tool_call_begin|>"
        "get_weather:0<|tool_call_argument_begin|>{}"
        "<|tool_call_end|><|tool_calls_section_end|>"
    ),
    "reasoning": None,
    "content": (
        "<|tool_calls_section_begin|><|tool_call_begin|>"
        "get_weather:0<|tool_call_argument_begin|>{}"
        "<|tool_call_end|><|tool_calls_section_end|>"
    ),
}
NO_TOOL_NO_THINK_END = {
    "output": "Just some reasoning without any markers",
    "reasoning": "Just some reasoning without any markers",
    "content": None,
}


@pytest.mark.parametrize(
    "param_dict",
    [
        pytest.param(SIMPLE_REASONING, id="simple_reasoning"),
        pytest.param(COMPLETE_REASONING, id="complete_reasoning"),
        pytest.param(REASONING_WITH_THINK, id="reasoning_with_think"),
        pytest.param(TOOL_SECTION_NO_THINK_END, id="tool_section_no_think_end"),
        pytest.param(
            TOOL_CALL_NO_SECTION_NO_THINK_END, id="tool_call_no_section_no_think_end"
        ),
        pytest.param(TOOL_SECTION_WITH_THINK, id="tool_section_with_think"),
        pytest.param(EMPTY_REASONING_WITH_TOOL, id="empty_reasoning_with_tool"),
        pytest.param(NO_TOOL_NO_THINK_END, id="no_tool_no_think_end"),
    ],
)
def test_kimi_k2_reasoning_extraction(param_dict: dict, kimi_k2_tokenizer):
    """Test that KimiK2ReasoningParser correctly extracts reasoning and content."""
    output = kimi_k2_tokenizer.tokenize(param_dict["output"])
    output_tokens: list[str] = [
        kimi_k2_tokenizer.convert_tokens_to_string([token]) for token in output
    ]
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        kimi_k2_tokenizer
    )

    reasoning, content = run_reasoning_extraction(
        parser, output_tokens, streaming=False
    )

    assert reasoning == param_dict["reasoning"]
    assert content == param_dict["content"]


def test_kimi_k2_tool_marker_priority(kimi_k2_tokenizer):
    """
    Test that tool_calls_section_begin takes priority over tool_call_begin.

    When both markers are present, we should split at the section marker.
    """
    # This shouldn't happen in practice, but tests the priority logic
    output = (
        "Reasoning here"
        "<|tool_calls_section_begin|>"
        "Some text"
        "<|tool_call_begin|>func:0<|tool_call_argument_begin|>{}<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    )

    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        kimi_k2_tokenizer
    )

    reasoning, content = parser.extract_reasoning(output, request=None)

    assert reasoning == "Reasoning here"
    assert content is not None
    assert content.startswith("<|tool_calls_section_begin|>")


def test_kimi_k2_inherits_deepseek_r1_behavior(kimi_k2_tokenizer):
    """
    Test that KimiK2ReasoningParser inherits standard DeepSeek R1 behavior
    for cases with proper </think> delimiters.
    """
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        kimi_k2_tokenizer
    )

    # Standard case with </think>
    reasoning, content = parser.extract_reasoning(
        "<think>My reasoning</think>The answer is 42", request=None
    )
    assert reasoning == "My reasoning"
    assert content == "The answer is 42"

    # Empty content after </think>
    reasoning, content = parser.extract_reasoning(
        "<think>Just thinking</think>", request=None
    )
    assert reasoning == "Just thinking"
    assert content is None


def test_kimi_k2_singular_section_variant(kimi_k2_tokenizer):
    """
    Test that the singular section variant <|tool_call_section_begin|> is
    correctly detected and used as split point.
    """
    output = (
        "I need to call a function"
        "<|tool_call_section_begin|><|tool_call_begin|>"
        "get_weather:0<|tool_call_argument_begin|>{}"
        "<|tool_call_end|><|tool_call_section_end|>"
    )

    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        kimi_k2_tokenizer
    )

    reasoning, content = parser.extract_reasoning(output, request=None)

    assert reasoning == "I need to call a function"
    assert content is not None
    assert content.startswith("<|tool_call_section_begin|>")


def test_kimi_k2_finds_first_marker(kimi_k2_tokenizer):
    """
    Test that when multiple tool markers are present, we split at the
    first occurring one (using min of positions).

    This handles edge cases where markers might appear in unexpected order.
    """
    # Hypothetical case: tool_call_begin without section wrapper
    output = (
        "Reasoning content"
        "<|tool_call_begin|>func:0<|tool_call_argument_begin|>{}<|tool_call_end|>"
    )

    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        kimi_k2_tokenizer
    )

    reasoning, content = parser.extract_reasoning(output, request=None)

    assert reasoning == "Reasoning content"
    assert content is not None
    assert content.startswith("<|tool_call_begin|>")


# Streaming tests
@pytest.mark.parametrize(
    "param_dict",
    [
        pytest.param(SIMPLE_REASONING, id="simple_reasoning"),
        pytest.param(COMPLETE_REASONING, id="complete_reasoning"),
        pytest.param(REASONING_WITH_THINK, id="reasoning_with_think"),
        pytest.param(TOOL_SECTION_NO_THINK_END, id="tool_section_no_think_end"),
        pytest.param(
            TOOL_CALL_NO_SECTION_NO_THINK_END, id="tool_call_no_section_no_think_end"
        ),
        pytest.param(TOOL_SECTION_WITH_THINK, id="tool_section_with_think"),
        pytest.param(EMPTY_REASONING_WITH_TOOL, id="empty_reasoning_with_tool"),
        pytest.param(NO_TOOL_NO_THINK_END, id="no_tool_no_think_end"),
    ],
)
def test_kimi_k2_reasoning_extraction_streaming(param_dict: dict, kimi_k2_tokenizer):
    """Test streaming extraction handles tool markers correctly."""
    output = kimi_k2_tokenizer.tokenize(param_dict["output"])
    output_tokens: list[str] = [
        kimi_k2_tokenizer.convert_tokens_to_string([token]) for token in output
    ]
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        kimi_k2_tokenizer
    )

    reasoning, content = run_reasoning_extraction(parser, output_tokens, streaming=True)

    assert reasoning == param_dict["reasoning"]
    assert content == param_dict["content"]
