# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest
from transformers import AutoTokenizer

from tests.reasoning.utils import run_reasoning_extraction
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.reasoning import ReasoningParser, ReasoningParserManager

parser_name = "step3"
start_token = "<think>"
end_token = "</think>"

# Step3 uses the same </think> end-of-reasoning token as DeepSeek-R1.
# The distilled Qwen-1.5B tokenizer carries </think> as a special token,
# so it serves as a lightweight stand-in for the full Step3 tokenizer.
REASONING_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


@pytest.fixture(scope="module")
def step3_tokenizer():
    return AutoTokenizer.from_pretrained(REASONING_MODEL_NAME)


@pytest.fixture(scope="module")
def step3_parser(step3_tokenizer):
    return ReasoningParserManager.get_reasoning_parser(parser_name)(step3_tokenizer)


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
SHORTEST_REASONING_NON_STREAMING = {
    "output": "</think>This is the rest",
    "reasoning": "",
    "content": "This is the rest",
    "is_reasoning_end": True,
}
SHORTEST_REASONING_STREAMING = {
    "output": "</think>This is the rest",
    "reasoning": None,
    "content": "This is the rest",
    "is_reasoning_end": True,
}
THINK_NO_END = {
    # Step3 does NOT strip the <think> start marker from reasoning.
    "output": "<think>This is a reasoning section",
    "reasoning": "<think>This is a reasoning section",
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

TEST_CASES = [
    pytest.param(False, SIMPLE_REASONING, id="simple_reasoning"),
    pytest.param(True, SIMPLE_REASONING, id="simple_reasoning_streaming"),
    pytest.param(False, COMPLETE_REASONING, id="complete_reasoning"),
    pytest.param(True, COMPLETE_REASONING, id="complete_reasoning_streaming"),
    pytest.param(False, NO_CONTENT, id="no_content"),
    pytest.param(True, NO_REASONING_STREAMING, id="no_reasoning_streaming"),
    pytest.param(False, MULTIPLE_LINES, id="multiple_lines"),
    pytest.param(True, MULTIPLE_LINES, id="multiple_lines_streaming"),
    pytest.param(False, SHORTEST_REASONING_NON_STREAMING, id="shortest_non_streaming"),
    pytest.param(True, SHORTEST_REASONING_STREAMING, id="shortest_streaming"),
    pytest.param(False, THINK_NO_END, id="think_no_end"),
    pytest.param(True, THINK_NO_END, id="think_no_end_streaming"),
    pytest.param(False, EMPTY, id="empty"),
    pytest.param(True, EMPTY_STREAMING, id="empty_streaming"),
]


@pytest.mark.parametrize("streaming, param_dict", TEST_CASES)
def test_reasoning(
    streaming: bool,
    param_dict: dict,
    step3_tokenizer,
):
    output = step3_tokenizer.tokenize(param_dict["output"])
    output_tokens: list[str] = [
        step3_tokenizer.convert_tokens_to_string([token]) for token in output
    ]
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        step3_tokenizer
    )

    reasoning, content = run_reasoning_extraction(
        parser, output_tokens, streaming=streaming
    )

    assert reasoning == param_dict["reasoning"]
    assert content == param_dict["content"]

    output_ids = step3_tokenizer.convert_tokens_to_ids(output)
    assert parser.is_reasoning_end(output_ids) == param_dict["is_reasoning_end"]

    if param_dict["content"] is not None:
        content_ids = parser.extract_content_ids(output_ids)
        assert content_ids == step3_tokenizer.convert_tokens_to_ids(
            step3_tokenizer.tokenize(param_dict["content"])
        )
    else:
        assert parser.extract_content_ids(output_ids) == []


def test_think_prefix_not_stripped(step3_parser):
    """Non-streaming extract_reasoning keeps <think> in the reasoning string.

    Unlike DeepSeekR1, Step3 does not strip the <think> start marker when
    processing a complete (non-streaming) response.
    """
    request = ChatCompletionRequest(messages=[], model="test-model")
    reasoning, content = step3_parser.extract_reasoning(
        "<think>some reasoning</think>some content", request=request
    )
    assert reasoning == "<think>some reasoning"
    assert content == "some content"


def test_think_end_token_required():
    """Parser construction must fail when the tokenizer lacks </think>."""
    bad_tokenizer = MagicMock()
    bad_tokenizer.get_vocab.return_value = {}

    with pytest.raises(RuntimeError, match="think end token"):
        ReasoningParserManager.get_reasoning_parser(parser_name)(bad_tokenizer)


def test_is_reasoning_end_streaming(step3_parser):
    """`is_reasoning_end_streaming` returns True only when </think> is in the
    current delta token IDs, not just in the accumulated prefix."""
    end_id = step3_parser.think_end_token_id
    other_ids = [end_id + 1]

    assert step3_parser.is_reasoning_end_streaming([], [end_id])
    assert not step3_parser.is_reasoning_end_streaming([end_id], other_ids)
