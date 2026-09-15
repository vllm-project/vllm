# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import overload

import pytest
from transformers import AutoTokenizer

from tests.reasoning.utils import run_reasoning_extraction
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser

parser_name = "deepseek_r1"
start_token = "<think>"
end_token = "</think>"

REASONING_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


@pytest.fixture(scope="module")
def deepseek_r1_qwen_tokenizer():
    return AutoTokenizer.from_pretrained(REASONING_MODEL_NAME)


class _CountingSequence(Sequence[int]):
    items_read = 0

    def __init__(self, values: list[int]) -> None:
        self._values = values

    @overload
    def __getitem__(self, index: int) -> int: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[int]: ...

    def __getitem__(self, index: int | slice) -> int | Sequence[int]:
        if isinstance(index, slice):
            value = self._values[index]
            type(self).items_read += len(value)
            return value
        type(self).items_read += 1
        return self._values[index]

    def __len__(self) -> int:
        return len(self._values)


class _TestTokenizer:
    def get_vocab(self) -> dict[str, int]:
        return {"<think>": 1, "</think>": 2, "token": 3}


def test_streaming_membership_only_reads_new_token_ids():
    parser = DeepSeekR1ReasoningParser(_TestTokenizer())
    _CountingSequence.items_read = 0

    for length in range(500):
        previous_ids = [3] * length
        delta = parser.extract_reasoning_streaming(
            previous_text="x" * length,
            current_text="x" * (length + 1),
            delta_text="x",
            previous_token_ids=_CountingSequence(previous_ids),
            current_token_ids=previous_ids + [3],
            delta_token_ids=[3],
        )
        assert delta is not None
        assert delta.reasoning == "x"

    # The cache reads the new tail plus constant-size boundary checks.
    # Re-scanning the accumulated list per probe would read ~125k items.
    assert _CountingSequence.items_read <= 3 * 500


def test_streaming_membership_rebuilds_after_history_rewind():
    parser = DeepSeekR1ReasoningParser(_TestTokenizer())

    # History holds both markers, so the delta is post-reasoning content.
    delta = parser.extract_reasoning_streaming(
        previous_text="<think>a</think>",
        current_text="<think>a</think>b",
        delta_text="b",
        previous_token_ids=[1, 3, 2],
        current_token_ids=[1, 3, 2, 3],
        delta_token_ids=[3],
    )
    assert delta is not None
    assert delta.content == "b"

    # A rewound history must not leave the stale markers in the cache.
    delta = parser.extract_reasoning_streaming(
        previous_text="a",
        current_text="ab",
        delta_text="b",
        previous_token_ids=[3],
        current_token_ids=[3, 3],
        delta_token_ids=[3],
    )
    assert delta is not None
    assert delta.reasoning == "b"


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
]


@pytest.mark.parametrize("streaming, param_dict", TEST_CASES)
def test_reasoning(
    streaming: bool,
    param_dict: dict,
    deepseek_r1_qwen_tokenizer,
):
    output = deepseek_r1_qwen_tokenizer.tokenize(param_dict["output"])
    # decode everything to tokens
    output_tokens: list[str] = [
        deepseek_r1_qwen_tokenizer.convert_tokens_to_string([token]) for token in output
    ]
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        deepseek_r1_qwen_tokenizer
    )

    reasoning, content = run_reasoning_extraction(
        parser, output_tokens, streaming=streaming
    )

    assert reasoning == param_dict["reasoning"]
    assert content == param_dict["content"]

    # Test is_reasoning_end
    output_ids = deepseek_r1_qwen_tokenizer.convert_tokens_to_ids(output)
    is_reasoning_end = parser.is_reasoning_end(output_ids)
    assert is_reasoning_end == param_dict["is_reasoning_end"]

    # Test extract_content
    if param_dict["content"] is not None:
        content = parser.extract_content_ids(output_ids)
        assert content == deepseek_r1_qwen_tokenizer.convert_tokens_to_ids(
            deepseek_r1_qwen_tokenizer.tokenize(param_dict["content"])
        )
    else:
        content = parser.extract_content_ids(output)
        assert content == []
