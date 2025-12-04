# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoTokenizer

from tests.reasoning.utils import run_reasoning_extraction
from vllm.reasoning import ReasoningParser, ReasoningParserManager

parser_name = "glm45"
start_token = "<think>"
end_token = "</think>"

REASONING_MODEL_NAME = "zai-org/GLM-4.5"


@pytest.fixture(scope="module")
def glm45_tokenizer():
    return AutoTokenizer.from_pretrained(REASONING_MODEL_NAME)


WITH_THINK = {
    "output": "<think>This is a reasoning section</think>This is the rest",
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
    "is_reasoning_end": True,
}

WITH_THINK_STREAM = {
    "output": "<think>This is a reasoning section</think>This is the rest",
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
    "is_reasoning_end": True,
}

WITHOUT_THINK = {
    "output": "This is the rest",
    "reasoning": None,
    "content": "This is the rest",
    "is_reasoning_end": False,
}

WITHOUT_THINK_STREAM = {
    "output": "This is the rest",
    "reasoning": None,
    "content": "This is the rest",
    "is_reasoning_end": False,
}

COMPLETE_REASONING = {
    "output": "<think>This is a reasoning section</think>",
    "reasoning": "This is a reasoning section",
    "content": None,
    "is_reasoning_end": True,
}
MULTILINE_REASONING = {
    "output": "<think>This is a reasoning\nsection</think>This is the rest\nThat",
    "reasoning": "This is a reasoning\nsection",
    "content": "This is the rest\nThat",
    "is_reasoning_end": True,
}
ONLY_OPEN_TAG = {
    "output": "<think>This is a reasoning section",
    "reasoning": None,
    "content": "<think>This is a reasoning section",
    "is_reasoning_end": False,
}

ONLY_OPEN_TAG_STREAM = {
    "output": "<think>This is a reasoning section",
    "reasoning": "This is a reasoning section",
    "content": None,
    "is_reasoning_end": False,
}

TEST_CASES = [
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
]

STILL_REASONING_PROMPT = """[gMASK]<sop><|system|>
You are a helpful assistant.<|user|>
What is the capital of France?<|assistant|>
<think>The user is asking for the capital of"""

DONE_REASONING_PROMPT = """[gMASK]<sop><|system|>
You are a helpful assistant.<|user|>
What is the capital of France?<|assistant|>
<think>The user is asking for the capital of France.</think>
The capital of France is Paris."""

MULTI_TURN_STILL_REASONING_PROMPT = """[gMASK]<sop><|system|>
You are a helpful assistant.<|user|>
What is the capital of France?<|assistant|>
<think></think>
The capital of France is Paris.<|user|>
What about Chile?<|assistant|>
<think>The user is asking for the capital of"""

MULTI_TURN_DONE_REASONING_PROMPT = """[gMASK]<sop><|system|>
You are a helpful assistant.<|user|>
What is the capital of France?<|assistant|>
<think></think>
The capital of France is Paris.<|user|>
What about Chile?<|assistant|>
<think>The user is asking for the capital of Chile.</think>
The capital of Chile is Santiago."""

REASONING_END_TEST_CASES = [
    pytest.param(STILL_REASONING_PROMPT, False, id="still_reasoning"),
    pytest.param(DONE_REASONING_PROMPT, True, id="done_reasoning"),
    pytest.param(
        MULTI_TURN_STILL_REASONING_PROMPT, False, id="multi_turn_still_reasoning"
    ),
    pytest.param(
        MULTI_TURN_DONE_REASONING_PROMPT, True, id="multi_turn_done_reasoning"
    ),
]


@pytest.mark.parametrize("streaming, param_dict", TEST_CASES)
def test_reasoning(
    streaming: bool,
    param_dict: dict,
    glm45_tokenizer,
):
    output = glm45_tokenizer.tokenize(param_dict["output"])
    output_tokens: list[str] = [
        glm45_tokenizer.convert_tokens_to_string([token]) for token in output
    ]
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        glm45_tokenizer
    )

    reasoning, content = run_reasoning_extraction(
        parser, output_tokens, streaming=streaming
    )

    assert reasoning == param_dict["reasoning"]
    assert content == param_dict["content"]

    output_ids = glm45_tokenizer.convert_tokens_to_ids(output)
    is_reasoning_end = parser.is_reasoning_end(output_ids)
    assert is_reasoning_end == param_dict["is_reasoning_end"]


@pytest.mark.parametrize("prompt, is_reasoning_end", REASONING_END_TEST_CASES)
def test_is_reasoning_end_full_prompt(
    prompt: str, is_reasoning_end: bool, glm45_tokenizer
):
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        glm45_tokenizer
    )
    tokens = glm45_tokenizer.tokenize(prompt)
    token_ids = glm45_tokenizer.convert_tokens_to_ids(tokens)
    check_is_reasoning_end = parser.is_reasoning_end(token_ids)
    assert check_is_reasoning_end == is_reasoning_end
