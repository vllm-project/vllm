# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoTokenizer

from tests.reasoning.utils import run_reasoning_extraction
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.entrypoints.openai.protocol import ChatCompletionRequest

parser_name = "exaone4"
start_token = "<think>"
end_token = "</think>"

REASONING_MODEL_NAME = "LGAI-EXAONE/EXAONE-4.0-1.2B"


@pytest.fixture(scope="module")
def exaone4_tokenizer():
    return AutoTokenizer.from_pretrained(REASONING_MODEL_NAME)


SIMPLE_REASONING = {
    "output": "This is a reasoning section</think>This is the rest",
    "reasoning_content": "This is a reasoning section",
    "content": "This is the rest",
    "is_reasoning_end": True,
}
COMPLETE_REASONING = {
    "output": "This is a reasoning section</think>",
    "reasoning_content": "This is a reasoning section",
    "content": None,
    "is_reasoning_end": True,
}
NO_REASONING = {
    "output": "This is content",
    "reasoning_content": None,
    "content": "This is content",
    "is_reasoning_end": False,
    "skip_extract_content": True,
}
NO_REASONING_STREAMING = {
    "output": "This is a normal section",
    "reasoning_content": None,
    "content": "This is a normal section",
    "is_reasoning_end": False,
    "skip_extract_content": True,
}
NO_REASONING_STREAMING_WITH_THINK = {
    "output": "This is a normal section",
    "reasoning_content": "This is a normal section",
    "content": None,
    "is_reasoning_end": False,
}
MULTIPLE_LINES = {
    "output": "This\nThat</think>This is the rest\nThat",
    "reasoning_content": "This\nThat",
    "content": "This is the rest\nThat",
    "is_reasoning_end": True,
}
SHORTEST_REASONING_NO_STREAMING = {
    "output": "</think>This is the rest",
    "reasoning_content": "",
    "content": "This is the rest",
    "is_reasoning_end": True,
}
SHORTEST_REASONING = {
    "output": "</think>This is the rest",
    "reasoning_content": None,
    "content": "This is the rest",
    "is_reasoning_end": True,
}
REASONING_WITH_THINK = {
    "output": "<think>This is a reasoning section</think>This is the rest",
    "reasoning_content": "This is a reasoning section",
    "content": "This is the rest",
    "is_reasoning_end": True,
}
COMPLETE_REASONING_WITH_THINK = {
    "output": "<think>This is a reasoning section</think>",
    "reasoning_content": "This is a reasoning section",
    "content": None,
    "is_reasoning_end": True,
}
MULTIPLE_LINES_WITH_THINK = {
    "output": "<think>This\nThat</think>This is the rest\nThat",
    "reasoning_content": "This\nThat",
    "content": "This is the rest\nThat",
    "is_reasoning_end": True,
}
SHORTEST_REASONING_NO_STREAMING_WITH_THINK = {
    "output": "</think>This is the rest",
    "reasoning_content": "",
    "content": "This is the rest",
    "is_reasoning_end": True,
}
SHORTEST_REASONING_WITH_THINK = {
    "output": "</think>This is the rest",
    "reasoning_content": None,
    "content": "This is the rest",
    "is_reasoning_end": True,
}
THINK_NO_END = {
    "output": "<think>This is a reasoning section",
    "reasoning_content": "This is a reasoning section",
    "content": None,
    "is_reasoning_end": False,
}
EMPTY = {
    "output": "",
    "reasoning_content": None,
    "content": "",
    "is_reasoning_end": False,
}
EMPTY_STREAMING = {
    "output": "",
    "reasoning_content": None,
    "content": None,
    "is_reasoning_end": False,
}
NEW_LINE = {
    "output": "\n<think>This is a reasoning section</think>\nThis is the rest",
    "reasoning_content": "This is a reasoning section",
    "content": "\nThis is the rest",
    "is_reasoning_end": True,
}
NEW_LINE_STREAMING = {
    "output": "\n<think>This is a reasoning section</think>\nThis is the rest",
    "reasoning_content": "\nThis is a reasoning section",
    "content": "\nThis is the rest",
    "is_reasoning_end": True,
}

TEST_CASES = [
    pytest.param(
        False,
        SIMPLE_REASONING,
        False,
        id="simple_reasoning",
    ),
    pytest.param(
        True,
        SIMPLE_REASONING,
        True,
        id="simple_reasoning_streaming",
    ),
    pytest.param(
        False,
        COMPLETE_REASONING,
        False,
        id="complete_reasoning",
    ),
    pytest.param(
        True,
        COMPLETE_REASONING,
        True,
        id="complete_reasoning_streaming",
    ),
    pytest.param(
        False,
        NO_REASONING,
        False,
        id="no_reasoning_token",
    ),
    pytest.param(
        True,
        NO_REASONING_STREAMING,
        False,
        id="no_reasoning_token_streaming",
    ),
    pytest.param(
        True,
        NO_REASONING_STREAMING_WITH_THINK,
        True,
        id="no_reasoning_token_streaming_with_think",
    ),
    pytest.param(
        False,
        MULTIPLE_LINES,
        False,
        id="multiple_lines",
    ),
    pytest.param(
        True,
        MULTIPLE_LINES,
        True,
        id="multiple_lines_streaming",
    ),
    pytest.param(
        True,
        SHORTEST_REASONING,
        False,
        id="shortest",
    ),
    pytest.param(
        False,
        SHORTEST_REASONING_NO_STREAMING,
        True,
        id="shortest_streaming",
    ),
    pytest.param(
        False,
        REASONING_WITH_THINK,
        False,
        id="reasoning_with_think",
    ),
    pytest.param(
        True,
        REASONING_WITH_THINK,
        True,
        id="reasoning_with_think_streaming",
    ),
    pytest.param(
        False,
        COMPLETE_REASONING_WITH_THINK,
        False,
        id="complete_reasoning_with_think",
    ),
    pytest.param(
        True,
        COMPLETE_REASONING_WITH_THINK,
        True,
        id="complete_reasoning_with_think_streaming",
    ),
    pytest.param(
        False,
        MULTIPLE_LINES_WITH_THINK,
        False,
        id="multiple_lines_with_think",
    ),
    pytest.param(
        True,
        MULTIPLE_LINES_WITH_THINK,
        True,
        id="multiple_lines_with_think_streaming",
    ),
    pytest.param(
        False,
        SHORTEST_REASONING_NO_STREAMING_WITH_THINK,
        False,
        id="shortest_with_think",
    ),
    pytest.param(
        True,
        SHORTEST_REASONING_WITH_THINK,
        True,
        id="shortest_with_think_streaming",
    ),
    pytest.param(
        False,
        THINK_NO_END,
        False,
        id="think_no_end",
    ),
    pytest.param(
        True,
        THINK_NO_END,
        True,
        id="think_no_end_streaming",
    ),
    pytest.param(
        False,
        EMPTY,
        False,
        id="empty",
    ),
    pytest.param(
        True,
        EMPTY_STREAMING,
        True,
        id="empty_streaming",
    ),
    pytest.param(
        False,
        NEW_LINE,
        False,
        id="new_line",
    ),
    pytest.param(
        True,
        NEW_LINE_STREAMING,
        True,
        id="new_line_streaming",
    ),
]


@pytest.mark.parametrize("streaming, param_dict, enable_thinking", TEST_CASES)
def test_reasoning(
    streaming: bool,
    param_dict: dict,
    enable_thinking: bool,
    exaone4_tokenizer,
):
    output = exaone4_tokenizer.tokenize(param_dict["output"])
    # decode everything to tokens
    output_tokens: list[str] = [
        exaone4_tokenizer.convert_tokens_to_string([token])
        for token in output
    ]
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(
        parser_name)(exaone4_tokenizer)

    dummy_request = ChatCompletionRequest(
        messages=[],
        chat_template_kwargs={"enable_thinking": enable_thinking},
    )
    reasoning, content = run_reasoning_extraction(parser,
                                                  output_tokens,
                                                  request=dummy_request if enable_thinking else None,
                                                  streaming=streaming)

    assert reasoning == param_dict["reasoning_content"]
    assert content == param_dict["content"]

    # Test is_reasoning_end
    output_ids = exaone4_tokenizer.convert_tokens_to_ids(output)
    is_reasoning_end = parser.is_reasoning_end(output_ids)
    assert is_reasoning_end == param_dict["is_reasoning_end"]

    # Test extract_content
    
    # NOTE: In case of `no_reasoning_token`s, We omit the extract_content test.
    # By default, EXAONE 4.0 parser assumes the content is the whole output 
    # if there is no '<think>' or '</think>', and `enable_thinking=False`. 
    # `extract_content_ids()` cannot get `enable_thinking` from the request,
    # and it is only used for removing the reasoning content from the output
    # on vllm.entrypoints.openai.serving_chat.py.
    # So we let `extract_content_ids()` as is (assume the output is reasoning content 
    # with the condition: no '<think>' or '</think>' and `enable_thinking=False`).
    if param_dict.get("skip_extract_content", False):
        return

    if param_dict["content"] is not None:
        content = parser.extract_content_ids(output_ids)
        assert content == exaone4_tokenizer.convert_tokens_to_ids(
            exaone4_tokenizer.tokenize(param_dict["content"]))
    else:
        content = parser.extract_content_ids(output)
        assert content == []
