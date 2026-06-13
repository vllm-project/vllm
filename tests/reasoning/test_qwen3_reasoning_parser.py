# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoTokenizer

from tests.reasoning.utils import (
    StreamingReasoningReconstructor,
    run_reasoning_extraction,
    run_reasoning_extraction_streaming,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.reasoning import ReasoningParser, ReasoningParserManager

parser_name = "qwen3"
start_token = "<think>"
end_token = "</think>"

REASONING_MODEL_NAMES = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3.5-397B-A17B",
    "Qwen/Qwen3-4B-Thinking-2507",
]


@pytest.fixture(scope="module", params=REASONING_MODEL_NAMES)
def qwen3_tokenizer(request):
    return AutoTokenizer.from_pretrained(request.param)


# --- <think> in prompt, only </think> in output (typical) ---

WITHOUT_START_TOKEN = {
    "output": "This is a reasoning section</think>This is the rest",
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
}
WITHOUT_START_TOKEN_STREAM = {
    "output": "This is a reasoning section</think>This is the rest",
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
}
WITHOUT_START_TOKEN_COMPLETE_REASONING = {
    "output": "This is a reasoning section</think>",
    "reasoning": "This is a reasoning section",
    "content": None,
}

# --- <think> present in output (old template / edge case) ---

WITH_THINK = {
    "output": "<think>This is a reasoning section</think>This is the rest",
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
}
WITH_THINK_STREAM = {
    "output": "<think>This is a reasoning section</think>This is the rest",
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
}

# --- No think tokens at all (thinking enabled, truncated) ---

# With thinking enabled (default), no think tokens means the output was
# truncated before </think> could be generated. All output is reasoning.
WITHOUT_THINK = {
    "output": "This is the rest",
    "reasoning": "This is the rest",
    "content": None,
}
# In streaming, the parser cannot distinguish "thinking disabled" from
# "reasoning in progress" when no think tokens have appeared yet.
# It assumes reasoning. The serving layer handles the "thinking disabled"
# case by checking prompt_is_reasoning_end_arr before calling the parser.
WITHOUT_THINK_STREAM = {
    "output": "This is the rest",
    "reasoning": "This is the rest",
    "content": None,
}

# --- <tool_call> without </think> (implicit reasoning end) ---

TOOL_CALL_BODY = (
    "<tool_call>\n<function=bash>\n<parameter=command>"
    "\ncat /etc/hosts\n</parameter>\n</function>\n</tool_call>"
)

TOOL_CALL_NO_THINK_END = {
    "output": "I need to read the file.\n\n" + TOOL_CALL_BODY,
    "reasoning": "I need to read the file.\n\n",
    "content": TOOL_CALL_BODY,
}

TOOL_CALL_WITH_THINK_NO_END = {
    "output": "<think>I need to read the file.\n\n" + TOOL_CALL_BODY,
    "reasoning": "I need to read the file.\n\n",
    "content": TOOL_CALL_BODY,
}

# --- Edge cases ---

COMPLETE_REASONING = {
    "output": "<think>This is a reasoning section</think>",
    "reasoning": "This is a reasoning section",
    "content": None,
}
MULTILINE_REASONING = {
    "output": "<think>This is a reasoning\nsection</think>This is the rest\nThat",
    "reasoning": "This is a reasoning\nsection",
    "content": "This is the rest\nThat",
}
# Truncated output: <think> present but no </think> (thinking enabled).
# Everything is reasoning because the output was cut off mid-thought.
ONLY_OPEN_TAG = {
    "output": "<think>This is a reasoning section",
    "reasoning": "This is a reasoning section",
    "content": None,
}

ONLY_OPEN_TAG_STREAM = {
    "output": "<think>This is a reasoning section",
    "reasoning": "This is a reasoning section",
    "content": None,
}

# Truncated output without <think> prefix (Qwen3.5 style where <think>
# is in the prompt). No </think> means truncation — all is reasoning.
TRUNCATED_NO_START_TOKEN = {
    "output": "This is a reasoning section",
    "reasoning": "This is a reasoning section",
    "content": None,
}

TRUNCATED_NO_START_TOKEN_STREAM = {
    "output": "This is a reasoning section",
    "reasoning": "This is a reasoning section",
    "content": None,
}

TEST_CASES = [
    pytest.param(
        False,
        WITHOUT_START_TOKEN,
        id="without_start_token",
    ),
    pytest.param(
        True,
        WITHOUT_START_TOKEN_STREAM,
        id="without_start_token_stream",
    ),
    pytest.param(
        False,
        WITHOUT_START_TOKEN_COMPLETE_REASONING,
        id="without_start_token_complete_reasoning",
    ),
    pytest.param(
        True,
        WITHOUT_START_TOKEN_COMPLETE_REASONING,
        id="without_start_token_complete_reasoning_stream",
    ),
    pytest.param(
        False,
        WITH_THINK,
        id="with_think",
    ),
    pytest.param(
        True,
        WITH_THINK_STREAM,
        id="with_think_stream",
    ),
    pytest.param(
        False,
        WITHOUT_THINK,
        id="without_think",
    ),
    pytest.param(
        True,
        WITHOUT_THINK_STREAM,
        id="without_think_stream",
    ),
    pytest.param(
        False,
        COMPLETE_REASONING,
        id="complete_reasoning",
    ),
    pytest.param(
        True,
        COMPLETE_REASONING,
        id="complete_reasoning_stream",
    ),
    pytest.param(
        False,
        MULTILINE_REASONING,
        id="multiline_reasoning",
    ),
    pytest.param(
        True,
        MULTILINE_REASONING,
        id="multiline_reasoning_stream",
    ),
    pytest.param(
        False,
        ONLY_OPEN_TAG,
        id="only_open_tag",
    ),
    pytest.param(
        True,
        ONLY_OPEN_TAG_STREAM,
        id="only_open_tag_stream",
    ),
    pytest.param(
        False,
        TRUNCATED_NO_START_TOKEN,
        id="truncated_no_start_token",
    ),
    pytest.param(
        True,
        TRUNCATED_NO_START_TOKEN_STREAM,
        id="truncated_no_start_token_stream",
    ),
    pytest.param(
        False,
        TOOL_CALL_NO_THINK_END,
        id="tool_call_no_think_end",
    ),
    pytest.param(
        True,
        TOOL_CALL_NO_THINK_END,
        id="tool_call_no_think_end_stream",
    ),
    pytest.param(
        False,
        TOOL_CALL_WITH_THINK_NO_END,
        id="tool_call_with_think_no_end",
    ),
    pytest.param(
        True,
        TOOL_CALL_WITH_THINK_NO_END,
        id="tool_call_with_think_no_end_stream",
    ),
]


@pytest.mark.parametrize("streaming, param_dict", TEST_CASES)
def test_reasoning(
    streaming: bool,
    param_dict: dict,
    qwen3_tokenizer,
):
    output = qwen3_tokenizer.tokenize(param_dict["output"])
    output_tokens: list[str] = [
        qwen3_tokenizer.convert_tokens_to_string([token]) for token in output
    ]
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        qwen3_tokenizer
    )

    reasoning, content = run_reasoning_extraction(
        parser, output_tokens, streaming=streaming
    )

    assert reasoning == param_dict["reasoning"]
    assert content == param_dict["content"]


# Multi-token delta tests: simulate real-world streaming where a single
# delta can contain multiple tokens (e.g., speculative decoding).
MULTI_TOKEN_DELTA_CASES = [
    pytest.param(
        # <think> grouped with following text in one delta
        ["<think>This is a reasoning section", "</think>", "This is the rest"],
        "This is a reasoning section",
        "This is the rest",
        id="start_token_grouped_with_text",
    ),
    pytest.param(
        # </think> grouped with following content in one delta
        ["reasoning section", "</think>This is the rest"],
        "reasoning section",
        "This is the rest",
        id="end_token_grouped_with_content",
    ),
    pytest.param(
        # <think> and </think> in the same delta, no content after
        ["<think>reasoning</think>"],
        "reasoning",
        None,
        id="start_and_end_in_one_delta_no_content",
    ),
    pytest.param(
        # No start token, end grouped with content (Qwen3.5 style)
        ["reasoning section", "</think>content"],
        "reasoning section",
        "content",
        id="no_start_end_grouped_with_content",
    ),
    pytest.param(
        # <tool_call> arrives in a separate delta after reasoning text
        ["I need to read the file.\n\n", "<tool_call>\n<function=bash>"],
        "I need to read the file.\n\n",
        "<tool_call>\n<function=bash>",
        id="tool_call_implicit_reasoning_end",
    ),
]


@pytest.mark.parametrize(
    "deltas, expected_reasoning, expected_content", MULTI_TOKEN_DELTA_CASES
)
def test_reasoning_streaming_multi_token_deltas(
    deltas: list[str],
    expected_reasoning: str | None,
    expected_content: str | None,
    qwen3_tokenizer,
):
    """Test that multi-token deltas don't leak <think> into reasoning."""
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        qwen3_tokenizer
    )

    reconstructor: StreamingReasoningReconstructor = run_reasoning_extraction_streaming(
        parser, deltas
    )

    assert reconstructor.reasoning == expected_reasoning
    assert (reconstructor.other_content or None) == expected_content


# --- Tests for enable_thinking=False (thinking explicitly disabled) ---


THINKING_DISABLED_CASES = [
    pytest.param(
        "This is plain content",
        None,
        "This is plain content",
        id="thinking_disabled_plain_content",
    ),
    pytest.param(
        "Some output without think tokens",
        None,
        "Some output without think tokens",
        id="thinking_disabled_no_think_tokens",
    ),
    pytest.param(
        "I need to read the file.\n\n" + TOOL_CALL_BODY,
        None,
        "I need to read the file.\n\n" + TOOL_CALL_BODY,
        id="thinking_disabled_with_tool_call",
    ),
]


@pytest.mark.parametrize(
    "output, expected_reasoning, expected_content", THINKING_DISABLED_CASES
)
def test_reasoning_thinking_disabled(
    output: str,
    expected_reasoning: str | None,
    expected_content: str | None,
    qwen3_tokenizer,
):
    """When enable_thinking=False, output without </think> is all content."""
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        qwen3_tokenizer,
        chat_template_kwargs={"enable_thinking": False},
    )

    reasoning, content = parser.extract_reasoning(
        model_output=output,
        request=ChatCompletionRequest(messages=[], model="test-model"),
    )

    assert reasoning == expected_reasoning
    assert content == expected_content


SPLITTING_DELTAS_AND_EXPECTED = [
    pytest.param(
        ["This is reasoning text", "</", "think>", "This is content"],
        "This is reasoning text",
        "This is content",
        id="end_tag_split_across_two_chunks",
    ),
    pytest.param(
        ["This is reasoning text<", "/think", ">", "This is content"],
        "This is reasoning text",
        "This is content",
        id="end_tag_split_three_ways",
    ),
    pytest.param(
        # Splitting into "</t" + "hi" to circumvent typo linting rules
        ["First reasoning", " Second reasoning", " </t" + "hi", "nk>", "content"],
        "First reasoning Second reasoning",
        "content",
        id="multiple_reasoning_chunks_before_split_tag",
    ),
    pytest.param(
        ["This is reasoning", "This is content without end tag"],
        "This is reasoningThis is content without end tag",
        None,
        id="missing_end_tag_everything_is_reasoning",
    ),
    pytest.param(
        # Splitting into "</t" + "hi" to circumvent typo linting rules
        ["This is reasoning</t" + "hi", "nkThis is content"],
        "This is reasoning</thinkThis is content",
        None,
        id="partial_tag_not_recognized",
    ),
    pytest.param(
        ["This is reasoning</th", "ink>This is content"],
        "This is reasoning",
        "This is content",
        id="case_3a_tag_split_reasoning_then_content",
    ),
    pytest.param(
        ["This is reasoning", "</think>This is content"],
        "This is reasoning",
        "This is content",
        id="case_3b_complete_tag_in_content_chunk",
    ),
    pytest.param(
        ["This is", " reasoning", "</think>This is content"],
        "This is reasoning",
        "This is content",
        id="case_3c_tag_with_content_after_reasoning",
    ),
    pytest.param(
        ["This is reasoning", "</th", "i", "nk>", "content"],
        "This is reasoning",
        "content",
        id="case_3d_tag_split_multiple_ways",
    ),
    pytest.param(
        [
            "I'll check the weather",
            "</think><tool_call>",
            "<function=get_weather>",
            "<parameter=ci",
            "ty>Tokyo</parameter>",
            "</function></tool_call>",
        ],
        "I'll check the weather",
        "<tool_call><function=get_weather><parameter=city>Tokyo</parameter></function></tool_call>",
        id="tool_call_parameter_tag_split",
    ),
    pytest.param(
        [
            "Let me search",
            "</think>",
            "<tool_call><function=search>",
            "<parameter=query>how",
            " to cook pasta</parameter>",
            "</function></tool_call>",
        ],
        "Let me search",
        "<tool_call><function=search><parameter=query>how to cook pasta</parameter></function></tool_call>",  # noqa: E501
        id="tool_call_parameter_value_split",
    ),
    pytest.param(
        [
            "Getting data",
            "</think>",
            "<tool_call>",
            "<function=api_call>",
            "<parameter=url>http",
            "s://api.example.com</parameter>",
            "<parameter=method>G",
            "ET</parameter>",
            "</function></tool_call>",
        ],
        "Getting data",
        "<tool_call><function=api_call><parameter=url>https://api.example.com</parameter><parameter=method>GET</parameter></function></tool_call>",  # noqa: E501
        id="multiple_parameters_split",
    ),
]


@pytest.mark.parametrize(
    "deltas, expected_reasoning, expected_content", SPLITTING_DELTAS_AND_EXPECTED
)
def test_splitting_reasoning_tokens(
    deltas: list[str],
    expected_reasoning: str | None,
    expected_content: str | None,
    qwen3_tokenizer,
):
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        qwen3_tokenizer
    )

    reconstructor: StreamingReasoningReconstructor = run_reasoning_extraction_streaming(
        parser, deltas
    )

    assert reconstructor.reasoning == expected_reasoning, (
        f"Expected reasoning '{expected_reasoning}' but got '{reconstructor.reasoning}'"
    )
    assert (reconstructor.other_content or None) == expected_content, (
        f"Expected content '{expected_content}' but got '{reconstructor.other_content}'"
    )


@pytest.mark.parametrize(
    "stream_interval",
    [2, 3, 5, 10],
    ids=["interval_2", "interval_3", "interval_5", "interval_10"],
)
def test_with_stream_interval(stream_interval: int, qwen3_tokenizer):
    """
    Test that simulates real-world MTP or large stream-interval scenarios.

    This test batches tokens according to stream_interval and verifies that
    the parser correctly handles all edge cases when tags are split across
    batch boundaries.
    """
    full_output = (
        "I need to analyze this carefully. "
        "First, let me think about the approach. "
        "The solution requires checking multiple factors."
        "</think>"
        "Based on my analysis, I'll proceed with the following action: "
        "<tool_call>"
        "<function=execute_query>"
        "<parameter=database>production</parameter>"
        "<parameter=query>SELECT * FROM users WHERE status='active'</parameter>"
        "</function>"
        "</tool_call>"
    )

    # Tokenize the full output
    tokens = qwen3_tokenizer.tokenize(full_output)
    token_strings = [
        qwen3_tokenizer.convert_tokens_to_string([token]) for token in tokens
    ]

    # Batch tokens according to stream_interval
    batched_deltas = []
    for i in range(0, len(token_strings), stream_interval):
        batch = "".join(token_strings[i : i + stream_interval])
        batched_deltas.append(batch)

    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        qwen3_tokenizer
    )

    reconstructor: StreamingReasoningReconstructor = run_reasoning_extraction_streaming(
        parser, batched_deltas
    )

    # Expected results
    expected_reasoning = (
        "I need to analyze this carefully. "
        "First, let me think about the approach. "
        "The solution requires checking multiple factors."
    )
    expected_content = (
        "Based on my analysis, I'll proceed with the following action: "
        "<tool_call>"
        "<function=execute_query>"
        "<parameter=database>production</parameter>"
        "<parameter=query>SELECT * FROM users WHERE status='active'</parameter>"
        "</function>"
        "</tool_call>"
    )

    assert reconstructor.reasoning == expected_reasoning, (
        f"With interval {stream_interval}, reasoning was incorrect"
    )
    assert reconstructor.other_content == expected_content, (
        f"With interval {stream_interval}, content was incorrect"
    )
