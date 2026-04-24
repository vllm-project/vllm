# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from tests.reasoning.utils import run_reasoning_extraction
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.tokenizers import get_tokenizer

parser_name = "hy_v3"
MODEL = "tencent/Hy3-preview"


@pytest.fixture(scope="module")
def hy_v3_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL)


WITH_THINK = {
    "output": "This is a reasoning section</think>This is the rest",
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
    "is_reasoning_end": True,
    "reasoning_effort": "high",
}

WITH_THINK_STREAM = {
    "output": "This is a reasoning section</think>This is the rest",
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
    "is_reasoning_end": True,
    "reasoning_effort": "high",
}

WITHOUT_THINK = {
    "output": "This is the rest",
    "reasoning": None,
    "content": "This is the rest",
    "is_reasoning_end": True,
    "reasoning_effort": "no_think",
}

WITHOUT_THINK_STREAM = {
    "output": "This is the rest",
    "reasoning": None,
    "content": "This is the rest",
    "is_reasoning_end": True,
    "reasoning_effort": "no_think",
}

WITH_REASONING_EFFORT_NONE = {
    "output": "This is the rest",
    "reasoning": None,
    "content": "This is the rest",
    "is_reasoning_end": True,
}

WITH_REASONING_EFFORT_NONE_STREAM = {
    "output": "This is the rest",
    "reasoning": None,
    "content": "This is the rest",
    "is_reasoning_end": True,
}

COMPLETE_REASONING = {
    "output": "This is a reasoning section</think>",
    "reasoning": "This is a reasoning section",
    "content": None,
    "is_reasoning_end": True,
    "reasoning_effort": "high",
}
MULTILINE_REASONING = {
    "output": "This is a reasoning\nsection</think>This is the rest\nThat",
    "reasoning": "This is a reasoning\nsection",
    "content": "This is the rest\nThat",
    "is_reasoning_end": True,
    "reasoning_effort": "high",
}
ONLY_OPEN_TAG = {
    "output": "This is a reasoning section",
    "reasoning": "This is a reasoning section",
    "content": None,
    "is_reasoning_end": False,
    "reasoning_effort": "high",
}

ONLY_OPEN_TAG_STREAM = {
    "output": "This is a reasoning section",
    "reasoning": "This is a reasoning section",
    "content": None,
    "is_reasoning_end": False,
    "reasoning_effort": "high",
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
        WITH_REASONING_EFFORT_NONE,
        id="with_reasoning_effort_none",
    ),
    pytest.param(
        True,
        WITH_REASONING_EFFORT_NONE_STREAM,
        id="with_reasoning_effort_none_stream",
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

STILL_REASONING_PROMPT = """<｜hy_begin▁of▁sentence｜>
You are a helpful assistant.
<｜reasoning_mode｜>reasoning_effort:high<｜hy_User｜>
What is the capital of France?<｜hy_Assistant｜>
<think>The user is asking for the capital of"""

DONE_REASONING_PROMPT = """<｜hy_begin▁of▁sentence｜>
You are a helpful assistant.
<｜reasoning_mode｜>reasoning_effort:high<｜hy_User｜>
What is the capital of France?<｜hy_Assistant｜>
<think>The user is asking for the capital of France.</think>
The capital of France is Paris."""

MULTI_TURN_STILL_REASONING_PROMPT = """<｜hy_begin▁of▁sentence｜>
You are a helpful assistant.
<｜reasoning_mode｜>reasoning_effort:high<｜hy_User｜>
What is the capital of France?<｜hy_Assistant｜
><think></think>The capital of France is Paris.<eos:6124c78e>
<｜hy_User｜>What about Chile?<｜hy_Assistant｜>
<think>The user is asking for the capital of"""

MULTI_TURN_DONE_REASONING_PROMPT = """<｜hy_begin▁of▁sentence｜>
You are a helpful assistant.
<｜reasoning_mode｜>reasoning_effort:high<｜hy_User｜>
What is the capital of France?<｜hy_Assistant｜
><think></think>The capital of France is Paris.<eos:6124c78e>
<｜hy_User｜>What about Chile?<｜hy_Assistant｜>
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
    hy_v3_tokenizer,
):
    output = hy_v3_tokenizer.tokenize(param_dict["output"])
    output_tokens: list[str] = [
        hy_v3_tokenizer.convert_tokens_to_string([token]) for token in output
    ]

    parser_kwargs = {}
    if "reasoning_effort" in param_dict:
        parser_kwargs["chat_template_kwargs"] = {
            "reasoning_effort": param_dict["reasoning_effort"]
        }
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        hy_v3_tokenizer,
        **parser_kwargs,
    )

    reasoning, content = run_reasoning_extraction(
        parser, output_tokens, streaming=streaming
    )

    assert reasoning == param_dict["reasoning"]
    assert content == param_dict["content"]

    output_ids = hy_v3_tokenizer.convert_tokens_to_ids(output)
    is_reasoning_end = parser.is_reasoning_end(output_ids)
    assert is_reasoning_end == param_dict["is_reasoning_end"]


@pytest.mark.parametrize("prompt, is_reasoning_end", REASONING_END_TEST_CASES)
def test_is_reasoning_end_full_prompt(
    prompt: str, is_reasoning_end: bool, hy_v3_tokenizer
):
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        hy_v3_tokenizer,
        chat_template_kwargs={"reasoning_effort": "high"},
    )
    tokens = hy_v3_tokenizer.tokenize(prompt)
    token_ids = hy_v3_tokenizer.convert_tokens_to_ids(tokens)
    check_is_reasoning_end = parser.is_reasoning_end(token_ids)
    assert check_is_reasoning_end == is_reasoning_end
