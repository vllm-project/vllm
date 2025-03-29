# SPDX-License-Identifier: Apache-2.0

import pytest
from transformers import AutoTokenizer

from tests.reasoning.utils import run_reasoning_extraction
from vllm.reasoning import ReasoningParser, ReasoningParserManager

parser_name = "deepseek_r1"
start_token = "<think>"
end_token = "</think>"

REASONING_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


@pytest.fixture(scope="module")
def deepseek_r1_qwen_tokenizer():
    return AutoTokenizer.from_pretrained(REASONING_MODEL_NAME)


SIMPLE_REASONING = {
    "tokens": [
        'This', 'Ġis', 'Ġa', 'Ġreasoning', 'Ġsection', '</think>', 'This',
        'Ġis', 'Ġthe', 'Ġrest'
    ],
    "output":
    "This is a reasoning section</think>This is the rest",
    "reasoning_content":
    "This is a reasoning section",
    "content":
    "This is the rest",
    "is_reasoning_end":
    True,
}
COMPLETE_REASONING = {
    "tokens": ['This', 'Ġis', 'Ġa', 'Ġreasoning', 'Ġsection', '</think>'],
    "output": "This is a reasoning section</think>",
    "reasoning_content": "This is a reasoning section",
    "content": None,
    "is_reasoning_end": True,
}
NO_CONTENT = {
    "tokens": ['This', 'Ġis', 'Ġcontent'],
    "output": "This is content",
    "reasoning_content": "This is content",
    "content": None,
    "is_reasoning_end": False,
}
NO_REASONING_STREAMING = {
    "tokens": ['This', 'Ġis', 'Ġa', 'Ġreasoning', 'Ġsection'],
    "output": "This is a reasoning section",
    "reasoning_content": "This is a reasoning section",
    "content": None,
    "is_reasoning_end": False,
}
MULTIPLE_LINES = {
    "tokens": [
        'This', 'Ċ', 'That', '</think>', 'This', 'Ġis', 'Ġthe', 'Ġrest', 'Ċ',
        'That'
    ],
    "output":
    "This\nThat</think>This is the rest\nThat",
    "reasoning_content":
    "This\nThat",
    "content":
    "This is the rest\nThat",
    "is_reasoning_end":
    True,
}
SHORTEST_REASONING_NO_STREAMING = {
    "tokens": ['</think>', 'This', 'Ġis', 'Ġthe', 'Ġrest'],
    "output": "</think>This is the rest",
    "reasoning_content": "",
    "content": "This is the rest",
    "is_reasoning_end": True,
}
SHORTEST_REASONING = {
    "tokens": ['</think>', 'This', 'Ġis', 'Ġthe', 'Ġrest'],
    "output": "</think>This is the rest",
    "reasoning_content": None,
    "content": "This is the rest",
    "is_reasoning_end": True,
}
REASONING_WITH_THINK = {
    "tokens": [
        '<think>', 'This', 'Ġis', 'Ġa', 'Ġreasoning', 'Ġsection', '</think>',
        'This', 'Ġis', 'Ġthe', 'Ġrest'
    ],
    "output":
    "<think>This is a reasoning section</think>This is the rest",
    "reasoning_content":
    "This is a reasoning section",
    "content":
    "This is the rest",
    "is_reasoning_end":
    True,
}
COMPLETE_REASONING_WITH_THINK = {
    "tokens":
    ['<think>', 'This', 'Ġis', 'Ġa', 'Ġreasoning', 'Ġsection', '</think>'],
    "output":
    "<think>This is a reasoning section</think>",
    "reasoning_content":
    "This is a reasoning section",
    "content":
    None,
    "is_reasoning_end":
    True,
}
MULTIPLE_LINES_WITH_THINK = {
    "tokens": [
        '<think>', 'This', 'Ċ', 'That', '</think>', 'This', 'Ġis', 'Ġthe',
        'Ġrest', 'Ċ', 'That'
    ],
    "output":
    "<think>This\nThat</think>This is the rest\nThat",
    "reasoning_content":
    "This\nThat",
    "content":
    "This is the rest\nThat",
    "is_reasoning_end":
    True,
}
SHORTEST_REASONING_NO_STREAMING_WITH_THINK = {
    "tokens": ['</think>', 'This', 'Ġis', 'Ġthe', 'Ġrest'],
    "output": "</think>This is the rest",
    "reasoning_content": "",
    "content": "This is the rest",
    "is_reasoning_end": True,
}
SHORTEST_REASONING_WITH_THINK = {
    "tokens": ['</think>', 'This', 'Ġis', 'Ġthe', 'Ġrest'],
    "output": "</think>This is the rest",
    "reasoning_content": None,
    "content": "This is the rest",
    "is_reasoning_end": True,
}
THINK_NO_END = {
    "tokens": ['<think>', 'This', 'Ġis', 'Ġa', 'Ġreasoning', 'Ġsection'],
    "output": "<think>This is a reasoning section",
    "reasoning_content": "This is a reasoning section",
    "content": None,
    "is_reasoning_end": False,
}
EMPTY = {
    "tokens": [],
    "output": "",
    "reasoning_content": "",
    "content": None,
    "is_reasoning_end": False,
}
EMPTY_STREAMING = {
    "tokens": [],
    "output": "",
    "reasoning_content": None,
    "content": None,
    "is_reasoning_end": False,
}
NEW_LINE = {
    "tokens": [
        'Ċ', '<think>', 'This', 'Ġis', 'Ġa', 'Ġreasoning', 'Ġsection',
        '</think>', 'Ċ', 'This', 'Ġis', 'Ġthe', 'Ġrest'
    ],
    "output":
    "\n<think>This is a reasoning section</think>\nThis is the rest",
    "reasoning_content":
    "This is a reasoning section",
    "content":
    "\nThis is the rest",
    "is_reasoning_end":
    True,
}
# Streaming cannot handle new lines at the beginning of the output
# because we need to support <think>...</think> and </think>...
# We cannot know if the text before <think> is reasoning content
# or not.
NEW_LINE_STREAMING = {
    "tokens": [
        'Ċ', '<think>', 'This', 'Ġis', 'Ġa', 'Ġreasoning', 'Ġsection',
        '</think>', 'Ċ', 'This', 'Ġis', 'Ġthe', 'Ġrest'
    ],
    "output":
    "\n<think>This is a reasoning section</think>\nThis is the rest",
    "reasoning_content":
    "\nThis is a reasoning section",
    "content":
    "\nThis is the rest",
    "is_reasoning_end":
    True,
}

# this is a special case: when ['<','/','think','>'] appeared
SPECIAL_CASE_FOR_NO_STREAMING = {
    "tokens": [
        'This', 'Ġis', 'Ġa', 'Ġreasoning', 'Ġsection', ':', 'Ġ</', 'think',
        '>', 'Ġis', 'Ġspecial', 'Ġtoken', '</think>', 'This', 'Ġis', 'Ġthe',
        'Ġrest', ',', 'Ġ</', 'think', '>', 'Ġis', 'Ġgood', 'Ġflag', '.'
    ],
    "output": ("This is a reasoning section: </think> is special token</think>"
               "This is the rest, </think> is good flag."),
    "reasoning_content":
    "This is a reasoning section: </think> is special token",
    "content":
    "This is the rest, </think> is good flag.",
    "is_reasoning_end":
    True,
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
        False,
        SPECIAL_CASE_FOR_NO_STREAMING,
        id="special_case_for_no_streaming",
    ),
]


@pytest.mark.parametrize("streaming, param_dict", TEST_CASES)
def test_reasoning(
    streaming: bool,
    param_dict: dict,
    deepseek_r1_qwen_tokenizer,
):
    output_tokens = param_dict["tokens"]
    model_output = param_dict["output"]
    token_ids = deepseek_r1_qwen_tokenizer.convert_tokens_to_ids(output_tokens)
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(
        parser_name)(deepseek_r1_qwen_tokenizer)

    reasoning, content = run_reasoning_extraction(parser,
                                                  model_output,
                                                  output_tokens,
                                                  token_ids,
                                                  streaming=streaming)

    assert reasoning == param_dict["reasoning_content"]
    assert content == param_dict["content"]

    # Test is_reasoning_end
    output_ids = deepseek_r1_qwen_tokenizer.convert_tokens_to_ids(
        output_tokens)
    is_reasoning_end = parser.is_reasoning_end(output_ids)
    assert is_reasoning_end == param_dict["is_reasoning_end"]

    # Test extract_content
    if param_dict["content"] is not None:
        content_token_ids = parser.extract_content_ids(output_ids)
        assert param_dict["content"] == deepseek_r1_qwen_tokenizer.decode(
            content_token_ids)
    else:
        content = parser.extract_content_ids(output_ids)
        assert content == []
