# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.reasoning.utils import run_reasoning_extraction
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.reasoning import ReasoningParser, ReasoningParserManager

# Using mistral tokenizer as a generic mock since the actual model is not on HF
from vllm.tokenizers.registry import get_tokenizer

parser_name = "gemma4"


@pytest.fixture(scope="module")
def generic_tokenizer():
    return get_tokenizer("google/gemma-4-E2B-it")


INVALID_SIMPLE_NONSTREAMING = {
    "output": "This is a reasoning section<channel|>This is the rest",
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
    "is_reasoning_end": True,
}
INVALID_SIMPLE_STREAMING = {
    "output": "This is a reasoning section<channel|>This is the rest",
    "reasoning": None,
    "content": "This is a reasoning sectionThis is the rest",
    "is_reasoning_end": True,
}
INVALID_COMPLETE_NONSTREAMING = {
    "output": "This is a reasoning section<channel|>",
    "reasoning": "This is a reasoning section",
    "content": None,
    "is_reasoning_end": True,
}
INVALID_COMPLETE_STREAMING = {
    "output": "This is a reasoning section<channel|>",
    "reasoning": None,
    "content": "This is a reasoning section",
    "is_reasoning_end": True,
}
NO_CONTENT = {
    "output": "<|channel>This is reasoning",
    "reasoning": "This is reasoning",
    "content": None,
    "is_reasoning_end": False,
}
NO_REASONING = {
    "output": "This is content",
    "reasoning": None,
    "content": "This is content",
    "is_reasoning_end": False,
}
REASONING_WITH_CHANNEL = {
    "output": "<|channel>This is a reasoning section<channel|>This is the rest",
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
    "is_reasoning_end": True,
}
COMPLETE_REASONING_WITH_CHANNEL = {
    "output": "<|channel>This is a reasoning section<channel|>",
    "reasoning": "This is a reasoning section",
    "content": None,
    "is_reasoning_end": True,
}
MULTIPLE_LINES_WITH_CHANNEL = {
    "output": "<|channel>This\nThat<channel|>This is the rest\nThat",
    "reasoning": "This\nThat",
    "content": "This is the rest\nThat",
    "is_reasoning_end": True,
}
CHANNEL_NO_END = {
    "output": "<|channel>This is a reasoning section",
    "reasoning": "This is a reasoning section",
    "content": None,
    "is_reasoning_end": False,
}
EMPTY = {
    "output": "",
    "reasoning": None,
    "content": "",
    "is_reasoning_end": False,
}
NEW_LINE_NONSTREAMING = {
    "output": (
        "Before\n<|channel>This is a reasoning section<channel|>\nThis is the rest"
    ),
    "reasoning": "This is a reasoning section",
    "content": "\nThis is the rest",
    "is_reasoning_end": True,
}
NEW_LINE_STREAMING = {
    "output": (
        "Before\n<|channel>This is a reasoning section<channel|>\nThis is the rest"
    ),
    "reasoning": "This is a reasoning section",
    "content": "Before\n\nThis is the rest",
    "is_reasoning_end": True,
}

THOUGHT_PREFIX = {
    "output": "<|channel>thought\nActual reasoning here<channel|>Final answer",
    "reasoning": "Actual reasoning here",
    "content": "Final answer",
    "is_reasoning_end": True,
}
THOUGHT_PREFIX_ONLY = {
    "output": "<|channel>thought\n<channel|>",
    "reasoning": "",
    "content": None,
    "is_reasoning_end": True,
}
THOUGHT_PREFIX_MULTILINE = {
    "output": "<|channel>thought\nLine1\nLine2<channel|>Answer",
    "reasoning": "Line1\nLine2",
    "content": "Answer",
    "is_reasoning_end": True,
}
# "thousand" starts like "thought" but diverges — exercises Case 2→3 in streaming.
THOUGHT_PREFIX_DIVERGE = {
    "output": "<|channel>thousand reasons<channel|>Done",
    "reasoning": "thousand reasons",
    "content": "Done",
    "is_reasoning_end": True,
}
# The model isn't reasoning if we're generating tool calls.
TOOL_CALL_STARTED = {
    "output": "<|tool_call>",
    "reasoning": None,
    "content": "<|tool_call>",
    "is_reasoning_end": True,
}

TEST_CASES = [
    pytest.param(False, INVALID_SIMPLE_NONSTREAMING, id="invalid_simple"),
    pytest.param(True, INVALID_SIMPLE_STREAMING, id="invalid_simple_streaming"),
    pytest.param(False, INVALID_COMPLETE_NONSTREAMING, id="invalid_complete"),
    pytest.param(True, INVALID_COMPLETE_STREAMING, id="invalid_complete_streaming"),
    pytest.param(False, NO_CONTENT, id="no_content"),
    pytest.param(False, NO_REASONING, id="no_reasoning"),
    pytest.param(False, REASONING_WITH_CHANNEL, id="reasoning"),
    pytest.param(True, REASONING_WITH_CHANNEL, id="reasoning_streaming"),
    pytest.param(False, COMPLETE_REASONING_WITH_CHANNEL, id="complete_reasoning"),
    pytest.param(
        True, COMPLETE_REASONING_WITH_CHANNEL, id="complete_reasoning_streaming"
    ),
    pytest.param(False, MULTIPLE_LINES_WITH_CHANNEL, id="multiple_lines"),
    pytest.param(True, MULTIPLE_LINES_WITH_CHANNEL, id="multiple_lines_streaming"),
    pytest.param(False, CHANNEL_NO_END, id="no_end"),
    pytest.param(True, CHANNEL_NO_END, id="no_end_streaming"),
    pytest.param(False, EMPTY, id="empty"),
    pytest.param(False, NEW_LINE_NONSTREAMING, id="new_line"),
    pytest.param(True, NEW_LINE_STREAMING, id="new_line_streaming"),
    pytest.param(False, THOUGHT_PREFIX, id="thought_prefix"),
    pytest.param(True, THOUGHT_PREFIX, id="thought_prefix_streaming"),
    pytest.param(False, THOUGHT_PREFIX_ONLY, id="thought_prefix_only"),
    pytest.param(True, THOUGHT_PREFIX_ONLY, id="thought_prefix_only_streaming"),
    pytest.param(False, THOUGHT_PREFIX_MULTILINE, id="thought_prefix_multiline"),
    pytest.param(
        True, THOUGHT_PREFIX_MULTILINE, id="thought_prefix_multiline_streaming"
    ),
    pytest.param(False, THOUGHT_PREFIX_DIVERGE, id="thought_prefix_diverge"),
    pytest.param(True, THOUGHT_PREFIX_DIVERGE, id="thought_prefix_diverge_streaming"),
    pytest.param(False, TOOL_CALL_STARTED, id="tool_call_started"),
    pytest.param(True, TOOL_CALL_STARTED, id="tool_call_started_streaming"),
]


def gemma4_encode_output(generic_tokenizer, output: str) -> list[int]:
    # Resolve token IDs dynamically from the real tokenizer
    vocab = generic_tokenizer.get_vocab()
    start_token_id = vocab["<|channel>"]
    end_token_id = vocab["<channel|>"]

    index_start = output.find("<|channel>")
    len_start = len("<|channel>")
    index_end = output.find("<channel|>")
    len_end = len("<channel|>")

    output_tokens = []

    def _encode(text: str) -> list[int]:
        if not text:
            return []
        # Handle both raw transformers and vLLM wrappers
        enc = getattr(generic_tokenizer, "tokenizer", generic_tokenizer)
        try:
            return enc.encode(text, add_special_tokens=False)
        except TypeError:
            return enc.encode(text)

    if index_start != -1:
        output_before = output[:index_start]
        output_tokens += _encode(output_before)
        output_tokens += [start_token_id]

        if index_end != -1:
            output_middle = output[index_start + len_start : index_end]
            output_after = output[index_end + len_end :]
            output_tokens += _encode(output_middle)
            output_tokens += [end_token_id]
            output_tokens += _encode(output_after)
        else:
            output_middle = output[index_start + len_start :]
            output_tokens += _encode(output_middle)
    elif index_end != -1:
        output_before = output[:index_end]
        output_after = output[index_end + len_end :]
        output_tokens += _encode(output_before)
        output_tokens += [end_token_id]
        output_tokens += _encode(output_after)
    else:
        output_tokens += _encode(output)

    return output_tokens


@pytest.mark.parametrize("streaming, param_dict", TEST_CASES)
def test_gemma4_reasoning(
    streaming: bool,
    param_dict: dict,
    generic_tokenizer,
):
    output = param_dict["output"]
    output_tokens = gemma4_encode_output(generic_tokenizer, output)

    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        generic_tokenizer
    )

    # We use the generic run_reasoning_extraction from utils
    # Use decode per token to get standard spaces instead of
    # SentencePiece space characters
    output_token_strings = [generic_tokenizer.decode([t]) for t in output_tokens]
    reasoning, content = run_reasoning_extraction(
        parser, output_token_strings, streaming=streaming
    )

    assert reasoning == param_dict["reasoning"]
    assert content == param_dict["content"]

    # Test is_reasoning_end
    is_reasoning_end = parser.is_reasoning_end(output_tokens)
    assert is_reasoning_end == param_dict["is_reasoning_end"]


def test_gemma4_adjust_request(generic_tokenizer):
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        generic_tokenizer
    )

    request = ChatCompletionRequest(messages=[], model="test-model")
    assert request.skip_special_tokens is True

    result = parser.adjust_request(request)
    assert result.skip_special_tokens is False
    assert result is request


def test_gemma4_previous_turn_reasoning_is_reasoning_end(generic_tokenizer):
    output = (
        "<|channel>thought\n1st thought<channel|>1st content<turn|>\n"
        "<|turn>user\nThanks<|turn>model\n"
    )
    output_tokens = gemma4_encode_output(generic_tokenizer, output)
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        generic_tokenizer
    )
    is_reasoning_end = parser.is_reasoning_end(output_tokens)
    assert not is_reasoning_end


# ---------------------------------------------------------------------------
# Unit tests for is_reasoning_end with a mock tokenizer (no network access)
# These directly test the fix for #39885 without requiring the real tokenizer.
# ---------------------------------------------------------------------------

# Stable synthetic token IDs for the mock vocab
_CHANNEL_START = 10  # <|channel>
_CHANNEL_END = 11  # <channel|>
_TURN = 12  # <|turn>
_TOOL_CALL = 13  # <|tool_call>
_TOOL_RESP = 14  # <|tool_response>
_TEXT_A = 20  # generic text token
_TEXT_B = 21  # generic text token


@pytest.fixture(scope="module")
def mock_gemma4_tokenizer():
    """Mock tokenizer with synthetic Gemma4 special tokens."""
    from unittest.mock import MagicMock

    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {
        "<|channel>": _CHANNEL_START,
        "<channel|>": _CHANNEL_END,
        "<|turn>": _TURN,
        "<|tool_call>": _TOOL_CALL,
        "<|tool_response>": _TOOL_RESP,
    }
    return tokenizer


@pytest.fixture(scope="module")
def mock_parser(mock_gemma4_tokenizer):
    return ReasoningParserManager.get_reasoning_parser(parser_name)(
        mock_gemma4_tokenizer
    )


@pytest.mark.parametrize(
    "input_ids, expected, description",
    [
        # Single tool_call with no prior context → current-turn tool call
        ([_TOOL_CALL], True, "bare tool_call (current turn)"),
        # Active reasoning followed by tool_call → True
        ([_CHANNEL_START, _TEXT_A, _TOOL_CALL], True, "channel start + tool_call"),
        # Turn boundary before tool_call → prior-turn tool call → False
        (
            [_TURN, _TEXT_A, _TOOL_CALL, _TEXT_B],
            False,
            "prior-turn tool_call (turn boundary)",
        ),
        # Tool response before tool_call → prior-turn → False
        (
            [_TOOL_CALL, _TEXT_A, _TOOL_RESP, _TEXT_B],
            False,
            "prior-turn tool_call (tool_response boundary)",
        ),
        # Channel end token in sequence → reasoning ended → True
        ([_CHANNEL_START, _TEXT_A, _CHANNEL_END], True, "channel end"),
        # Channel start without end → reasoning active → False
        ([_CHANNEL_START, _TEXT_A], False, "active reasoning"),
        # Empty sequence → False
        ([], False, "empty"),
        # Regular text only → False
        ([_TEXT_A, _TEXT_B], False, "plain text"),
        # Multi-turn: prior tool call then turn then model text → False
        # (This is the exact #39885 A4B scenario: prompt ends with
        #  <|turn>model\n<|tool_call> where turn boundary comes before tool_call
        #  in the backwards scan, i.e. <|tool_call> is to the right of <|turn>)
        (
            [_TURN, _TEXT_A, _TOOL_CALL, _TEXT_B, _TURN, _TEXT_B],
            False,
            "A4B-style: tool_call inside prior turn, new turn at end",
        ),
    ],
)
def test_is_reasoning_end_mock(mock_parser, input_ids, expected, description):
    """Test is_reasoning_end with synthetic token IDs (no real tokenizer needed).

    Covers the #39885 regression: prior-turn <|tool_call> in the prompt must
    not cause is_reasoning_end to return True prematurely.
    """
    result = mock_parser.is_reasoning_end(input_ids)
    assert result == expected, (
        f"is_reasoning_end({input_ids}) expected {expected} for case: {description}"
    )
