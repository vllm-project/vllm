# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoTokenizer

from tests.reasoning.utils import run_reasoning_extraction
from vllm.reasoning import ReasoningParser, ReasoningParserManager

parser_name = "kimi_k25"

REASONING_MODEL_NAME = "moonshotai/Kimi-K2.5"


@pytest.fixture(scope="module")
def kimi_k2_tokenizer():
    return AutoTokenizer.from_pretrained(REASONING_MODEL_NAME, trust_remote_code=True)


# The following tests verify that standard reasoning behavior is preserved
# through KimiK2ReasoningParser's delegation to DeepSeekR1ReasoningParser
SIMPLE_REASONING = {
    "output": "This is a reasoning section</think>This is the rest",
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
    "is_reasoning_end": True,
}
COMPLETE_REASONING = {
    "output": "This is a reasoning section</think>",
    "reasoning": "This is a reasoning section",
    "content": None,
    "is_reasoning_end": True,
}
NO_CONTENT = {
    "output": "This is content",
    "reasoning": "This is content",
    "content": None,
    "is_reasoning_end": False,
}
NO_REASONING_STREAMING = {
    "output": "This is a reasoning section",
    "reasoning": "This is a reasoning section",
    "content": None,
    "is_reasoning_end": False,
}
MULTIPLE_LINES = {
    "output": "This\nThat</think>This is the rest\nThat",
    "reasoning": "This\nThat",
    "content": "This is the rest\nThat",
    "is_reasoning_end": True,
}
SHORTEST_REASONING_NO_STREAMING = {
    "output": "</think>This is the rest",
    "reasoning": "",
    "content": "This is the rest",
    "is_reasoning_end": True,
}
SHORTEST_REASONING = {
    "output": "</think>This is the rest",
    "reasoning": None,
    "content": "This is the rest",
    "is_reasoning_end": True,
}
REASONING_WITH_THINK = {
    "output": "<think>This is a reasoning section</think>This is the rest",
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
    "is_reasoning_end": True,
}
COMPLETE_REASONING_WITH_THINK = {
    "output": "<think>This is a reasoning section</think>",
    "reasoning": "This is a reasoning section",
    "content": None,
    "is_reasoning_end": True,
}
MULTIPLE_LINES_WITH_THINK = {
    "output": "<think>This\nThat</think>This is the rest\nThat",
    "reasoning": "This\nThat",
    "content": "This is the rest\nThat",
    "is_reasoning_end": True,
}
SHORTEST_REASONING_NO_STREAMING_WITH_THINK = {
    "output": "</think>This is the rest",
    "reasoning": "",
    "content": "This is the rest",
    "is_reasoning_end": True,
}
SHORTEST_REASONING_WITH_THINK = {
    "output": "</think>This is the rest",
    "reasoning": None,
    "content": "This is the rest",
    "is_reasoning_end": True,
}
THINK_NO_END = {
    "output": "<think>This is a reasoning section",
    "reasoning": "This is a reasoning section",
    "content": None,
    "is_reasoning_end": False,
}
EMPTY = {
    "output": "",
    "reasoning": "",
    "content": None,
    "is_reasoning_end": False,
}
EMPTY_STREAMING = {
    "output": "",
    "reasoning": None,
    "content": None,
    "is_reasoning_end": False,
}
NEW_LINE = {
    "output": "\n<think>This is a reasoning section</think>\nThis is the rest",
    "reasoning": "This is a reasoning section",
    "content": "\nThis is the rest",
    "is_reasoning_end": True,
}
# Streaming cannot handle new lines at the beginning of the output
# because we need to support <think>...</think> and </think>...
# We cannot know if the text before <think> is reasoning content
# or not.
NEW_LINE_STREAMING = {
    "output": "\n<think>This is a reasoning section</think>\nThis is the rest",
    "reasoning": "\nThis is a reasoning section",
    "content": "\nThis is the rest",
    "is_reasoning_end": True,
}

# The following tests verify Kimi-specific behavior: tool call tokens
# must not leak into reasoning_content when </think> is absent
TOOL_CALL_DURING_THINK = {
    "output": (
        "<think>"
        "I should read that file"
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>"
        "functions.read:0"
        "<|tool_call_argument_begin|>"
        "{'filePath':'file.txt'}"
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    ),
    "reasoning": "I should read that file",
    "content": (
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>"
        "functions.read:0"
        "<|tool_call_argument_begin|>"
        "{'filePath':'file.txt'}"
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    ),
    "is_reasoning_end": False,
    "skip_content_ids": True,
}

TOOL_CALL_IMMEDIATELY_AFTER_THINK = {
    "output": (
        "<think>"
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>"
        "functions.read:0"
        "<|tool_call_argument_begin|>"
        "{'filePath':'file.txt'}"
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    ),
    "reasoning": None,
    "content": (
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>"
        "functions.read:0"
        "<|tool_call_argument_begin|>"
        "{'filePath':'file.txt'}"
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    ),
    "is_reasoning_end": False,
    "skip_content_ids": True,
}

TOOL_CALL_AFTER_THINK_END = {
    "output": (
        "<think>I should read that file</think>"
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>"
        "functions.read:0"
        "<|tool_call_argument_begin|>"
        "{'filePath':'file.txt'}"
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    ),
    "reasoning": "I should read that file",
    "content": (
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>"
        "functions.read:0"
        "<|tool_call_argument_begin|>"
        "{'filePath':'file.txt'}"
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    ),
    "is_reasoning_end": True,
}

TEST_CASES = [
    pytest.param(
        False,
        SIMPLE_REASONING,
        id="simple_reasoning",
    ),
    pytest.param(
        True,
        SIMPLE_REASONING,
        id="simple_reasoning_streaming",
    ),
    pytest.param(
        False,
        COMPLETE_REASONING,
        id="complete_reasoning",
    ),
    pytest.param(
        True,
        COMPLETE_REASONING,
        id="complete_reasoning_streaming",
    ),
    pytest.param(
        False,
        NO_CONTENT,
        id="no_content_token",
    ),
    pytest.param(
        True,
        NO_REASONING_STREAMING,
        id="no_reasoning_token_streaming",
    ),
    pytest.param(
        False,
        MULTIPLE_LINES,
        id="multiple_lines",
    ),
    pytest.param(
        True,
        MULTIPLE_LINES,
        id="multiple_lines_streaming",
    ),
    pytest.param(
        True,
        SHORTEST_REASONING,
        id="shortest",
    ),
    pytest.param(
        False,
        SHORTEST_REASONING_NO_STREAMING,
        id="shortest_streaming",
    ),
    pytest.param(
        False,
        REASONING_WITH_THINK,
        id="reasoning_with_think",
    ),
    pytest.param(
        True,
        REASONING_WITH_THINK,
        id="reasoning_with_think_streaming",
    ),
    pytest.param(
        False,
        COMPLETE_REASONING_WITH_THINK,
        id="complete_reasoning_with_think",
    ),
    pytest.param(
        True,
        COMPLETE_REASONING_WITH_THINK,
        id="complete_reasoning_with_think_streaming",
    ),
    pytest.param(
        False,
        MULTIPLE_LINES_WITH_THINK,
        id="multiple_lines_with_think",
    ),
    pytest.param(
        True,
        MULTIPLE_LINES_WITH_THINK,
        id="multiple_lines_with_think_streaming",
    ),
    pytest.param(
        False,
        SHORTEST_REASONING_NO_STREAMING_WITH_THINK,
        id="shortest_with_think",
    ),
    pytest.param(
        True,
        SHORTEST_REASONING_WITH_THINK,
        id="shortest_with_think_streaming",
    ),
    pytest.param(
        False,
        THINK_NO_END,
        id="think_no_end",
    ),
    pytest.param(
        True,
        THINK_NO_END,
        id="think_no_end_streaming",
    ),
    pytest.param(
        False,
        EMPTY,
        id="empty",
    ),
    pytest.param(
        True,
        EMPTY_STREAMING,
        id="empty_streaming",
    ),
    pytest.param(
        False,
        NEW_LINE,
        id="new_line",
    ),
    pytest.param(
        True,
        NEW_LINE_STREAMING,
        id="new_line_streaming",
    ),
    pytest.param(
        True,
        TOOL_CALL_DURING_THINK,
        id="tool_call_during_think",
    ),
    pytest.param(
        True,
        TOOL_CALL_IMMEDIATELY_AFTER_THINK,
        id="tool_call_immediately_after_think",
    ),
    pytest.param(
        True,
        TOOL_CALL_AFTER_THINK_END,
        id="tool_call_after_think_end",
    ),
]


@pytest.mark.parametrize("streaming, param_dict", TEST_CASES)
def test_reasoning(
    streaming: bool,
    param_dict: dict,
    kimi_k2_tokenizer,
):
    output = kimi_k2_tokenizer.tokenize(param_dict["output"])
    # decode everything to tokens
    output_tokens: list[str] = [
        kimi_k2_tokenizer.convert_tokens_to_string([token]) for token in output
    ]
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        kimi_k2_tokenizer
    )

    reasoning, content = run_reasoning_extraction(
        parser, output_tokens, streaming=streaming
    )

    assert reasoning == param_dict["reasoning"]
    assert content == param_dict["content"]

    # Test is_reasoning_end
    output_ids = kimi_k2_tokenizer.convert_tokens_to_ids(output)
    is_reasoning_end = parser.is_reasoning_end(output_ids)
    assert is_reasoning_end == param_dict["is_reasoning_end"]

    # Test extract_content
    if param_dict["content"] is not None and not param_dict.get("skip_content_ids"):
        content = parser.extract_content_ids(output_ids)
        assert content == kimi_k2_tokenizer.convert_tokens_to_ids(
            kimi_k2_tokenizer.tokenize(param_dict["content"])
        )
    else:
        content = parser.extract_content_ids(output_ids)
        assert content == []


def test_kimi_k25_parser_registration(kimi_k2_tokenizer):
    """kimi_k2 must resolve to KimiK25ReasoningParser, not DeepSeek."""
    from vllm.reasoning.kimi_k25_reasoning_parser import (
        KimiK25ReasoningParser,
    )

    cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    assert cls is KimiK25ReasoningParser


def test_kimi_k25_thinking_always_enabled(kimi_k2_tokenizer):
    """Thinking mode must be forced even without chat_template_kwargs."""
    from vllm.reasoning.deepseek_r1_reasoning_parser import (
        DeepSeekR1ReasoningParser,
    )

    cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    parser = cls(kimi_k2_tokenizer)
    assert isinstance(parser._parser, DeepSeekR1ReasoningParser)


def test_kimi_k25_tool_token_ids_populated(kimi_k2_tokenizer):
    """Tool call token IDs must be set from the real tokenizer vocab."""
    cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    parser = cls(kimi_k2_tokenizer)
    # 163595: <|tool_calls_section_begin|>
    assert (
        kimi_k2_tokenizer.convert_tokens_to_ids("<|tool_calls_section_begin|>")
        in parser._reasoning_end_token_ids
    )
    # 163597: <|tool_call_begin|>
    assert (
        kimi_k2_tokenizer.convert_tokens_to_ids("<|tool_call_begin|>")
        in parser._reasoning_end_token_ids
    )
