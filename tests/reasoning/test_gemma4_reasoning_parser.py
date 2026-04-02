# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.reasoning.utils import run_reasoning_extraction
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
        "Before\n<|channel>This is a reasoning section<channel|>\n"
        "This is the rest"
    ),
    "reasoning": "This is a reasoning section",
    "content": "\nThis is the rest",
    "is_reasoning_end": True,
}
NEW_LINE_STREAMING = {
    "output": (
        "Before\n<|channel>This is a reasoning section<channel|>\n"
        "This is the rest"
    ),
    "reasoning": "This is a reasoning section",
    "content": "Before\n\nThis is the rest",
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
        True, COMPLETE_REASONING_WITH_CHANNEL,
        id="complete_reasoning_streaming"
    ),
    pytest.param(False, MULTIPLE_LINES_WITH_CHANNEL, id="multiple_lines"),
    pytest.param(True, MULTIPLE_LINES_WITH_CHANNEL, id="multiple_lines_streaming"),
    pytest.param(False, CHANNEL_NO_END, id="no_end"),
    pytest.param(True, CHANNEL_NO_END, id="no_end_streaming"),
    pytest.param(False, EMPTY, id="empty"),
    pytest.param(False, NEW_LINE_NONSTREAMING, id="new_line"),
    pytest.param(True, NEW_LINE_STREAMING, id="new_line_streaming"),
]


@pytest.mark.parametrize("streaming, param_dict", TEST_CASES)
def test_gemma4_reasoning(
    streaming: bool,
    param_dict: dict,
    generic_tokenizer,
):
    output = param_dict["output"]

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

