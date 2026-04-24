# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import pytest
from transformers import AutoTokenizer

from tests.reasoning.utils import (
    StreamingReasoningReconstructor,
    run_reasoning_extraction,
    run_reasoning_extraction_streaming,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    ExtractedToolCallInformation,
)
from vllm.parser.parser_manager import ParserManager
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.tool_parsers.abstract_tool_parser import ToolParser

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


def test_is_reasoning_end_complete_tool_call_in_single_delta(
    qwen3_tokenizer,
) -> None:
    """
    Regression – is_reasoning_end must return True when a complete
    <tool_call>…</tool_call> arrives in a single delta (MTP / speculative-
    decoding scenario). The paired-token guard was incorrectly skipping this
    case, preventing reasoning from being marked as ended.
    """
    vocab = qwen3_tokenizer.get_vocab()
    tool_call_id = vocab["<tool_call>"]
    tool_call_end_id = vocab["</tool_call>"]
    # Use a regular content token (first token ID from any word)
    content_token = qwen3_tokenizer.encode("hello", add_special_tokens=False)[0]

    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(
        parser_name
    )(qwen3_tokenizer)

    delta_ids = [tool_call_id, content_token, tool_call_end_id]
    assert parser.is_reasoning_end(delta_ids), (
        "is_reasoning_end returned False for a complete <tool_call>…</tool_call> "
        "pair in a single delta — the paired-token guard incorrectly skips real "
        "output tool calls when both opening and closing tokens arrive together."
    )


def test_is_reasoning_end_false_after_complete_tool_call_in_output(
    qwen3_tokenizer,
) -> None:
    """
    is_reasoning_end must return True for accumulated
    output that contains a complete <tool_call>…</tool_call> (no explicit
    </think>). Once reasoning ended via <tool_call>, the invariant must hold
    for any superset of those tokens.
    """
    vocab = qwen3_tokenizer.get_vocab()
    tool_call_id = vocab["<tool_call>"]
    tool_call_end_id = vocab["</tool_call>"]
    reasoning_token = qwen3_tokenizer.encode("thinking", add_special_tokens=False)[0]
    content_token = qwen3_tokenizer.encode("hello", add_special_tokens=False)[0]

    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(
        parser_name
    )(qwen3_tokenizer)

    output_ids = [
        reasoning_token,
        reasoning_token,
        tool_call_id,
        content_token,
        tool_call_end_id,
    ]
    assert parser.is_reasoning_end(output_ids), (
        "is_reasoning_end returned False after <tool_call>…</tool_call> is "
        "complete in the accumulated output — the paired-token guard violates "
        "the invariant that reasoning stays ended once it has ended."
    )


def test_extract_reasoning_streaming_fragmented_tool_call(qwen3_tokenizer):
    """
    Test streaming reasoning extraction when the <tool_call> tag is fragmented
    across multiple deltas and delta_token_ids does not contain the special token ID.
    """
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        qwen3_tokenizer
    )

    # Force the parser to have a valid tool call tag
    # (Qwen3 uses <tool_call>)
    assert parser._tool_call_tag == "<tool_call>"
    
    # Delta 1: "<to"
    previous_text = "Here is my reasoning. "
    current_text = "Here is my reasoning. <to"
    msg1 = parser.extract_reasoning_streaming(
        previous_text=previous_text,
        current_text=current_text,
        delta_text="<to",
        previous_token_ids=[1, 2],
        current_token_ids=[1, 2, 3],
        delta_token_ids=[3],
    )
    assert msg1 is not None
    assert msg1.reasoning == "<to"
    assert msg1.content is None

    # Delta 2: "ol_"
    previous_text = current_text
    current_text = current_text + "ol_"
    msg2 = parser.extract_reasoning_streaming(
        previous_text=previous_text,
        current_text=current_text,
        delta_text="ol_",
        previous_token_ids=[1, 2, 3],
        current_token_ids=[1, 2, 3, 4],
        delta_token_ids=[4],
    )
    assert msg2 is not None
    assert msg2.reasoning == "ol_"
    assert msg2.content is None

    # Delta 3: "call>\n<function=bash>"
    previous_text = current_text
    current_text = current_text + "call>\n<function=bash>"
    msg3 = parser.extract_reasoning_streaming(
        previous_text=previous_text,
        current_text=current_text,
        delta_text="call>\n<function=bash>",
        previous_token_ids=[1, 2, 3, 4],
        current_token_ids=[1, 2, 3, 4, 5, 6],
        delta_token_ids=[5, 6],
    )
    assert msg3 is not None
    assert msg3.reasoning is None
    assert msg3.content == "call>\n<function=bash>"


class _StubToolParser(ToolParser):
    """Always returns DeltaMessage(content="[tool]") to detect overwrites."""

    def extract_tool_calls(
        self, model_output: str, _request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=model_output
        )

    def extract_tool_calls_streaming(
        self,
        _previous_text: str,
        _current_text: str,
        _delta_text: str,
        _previous_token_ids: Sequence[int],
        _current_token_ids: Sequence[int],
        _delta_token_ids: Sequence[int],
        _request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        return DeltaMessage(content="[tool]")


def test_reasoning_preserved_when_transition_and_tool_call_in_same_delta(
    qwen3_tokenizer,
) -> None:
    """
    When a delta contains both the last reasoning fragment AND the <tool_call>
    token (implicit reasoning end), parse_delta must not lose the reasoning
    text by overwriting delta_message with the result of
    extract_tool_calls_streaming.

    Sequence inside parse_delta (before fix):
      1. extract_reasoning_streaming  → DeltaMessage(reasoning="last text",
                                                      content="<tool_call>…")
      2. Transition: reasoning_ended = True, current_text = "<tool_call>…"
      3. extract_tool_calls_streaming → DeltaMessage(content="[tool]")
      4. delta_message = step-3 result  ← reasoning "last text" silently lost
    """
    vocab = qwen3_tokenizer.get_vocab()
    tool_call_id = vocab["<tool_call>"]
    reasoning_token = qwen3_tokenizer.encode("reasoning", add_special_tokens=False)[0]

    parser_cls = ParserManager.get_parser(
        reasoning_parser_name="qwen3",
        tool_parser_name="qwen3_coder",
    )
    # Swap in the stub so the test doesn't depend on qwen3_coder streaming logic
    parser = parser_cls(qwen3_tokenizer)
    parser._tool_parser = _StubToolParser(qwen3_tokenizer)

    request = ChatCompletionRequest(messages=[], model="test-model")
    delta_text = "last reasoning text\n<tool_call>\n<function=test_func>"

    result = parser.parse_delta(
        delta_text=delta_text,
        delta_token_ids=[reasoning_token, tool_call_id],
        request=request,
    )

    assert result is not None, (
        "parse_delta returned None — both reasoning and tool content were lost."
    )
    assert result.reasoning == "last reasoning text\n", (
        f"last reasoning fragment lost in transition delta. "
        f"Got reasoning={result.reasoning!r}, expected 'last reasoning text\\n'. "
        f"delta_message was overwritten by extract_tool_calls_streaming."
    )
