# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.reasoning.utils import run_reasoning_extraction_mistral
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.tokenizers.mistral import MistralTokenizer

parser_name = "mistral"


@pytest.fixture(scope="module")
def mistral_tokenizer():
    mistral_tokenizer = MistralTokenizer.from_pretrained(
        "mistralai/Magistral-Small-2509"
    )
    return mistral_tokenizer


INVALID_SIMPLE_REASONING = {
    "output": "This is a reasoning section[/THINK]This is the rest",
    "reasoning": None,
    "content": "This is a reasoning sectionThis is the rest",
    "is_reasoning_end": False,
}
INVALID_COMPLETE_REASONING = {
    "output": "This is a reasoning section[/THINK]",
    "reasoning": None,
    "content": "This is a reasoning section",
    "is_reasoning_end": False,
}
NO_CONTENT = {
    "output": "[THINK]This is reasoning",
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
NO_REASONING_STREAMING = {
    "output": "This is a reasoning section",
    "reasoning": None,
    "content": "This is a reasoning section",
    "is_reasoning_end": False,
}
INVALID_MULTIPLE_LINES = {
    "output": "This\nThat[/THINK]This is the rest\nThat",
    "reasoning": None,
    "content": "This\nThatThis is the rest\nThat",
    "is_reasoning_end": False,
}
INVALID_SHORTEST_REASONING_NO_STREAMING = {
    "output": "[/THINK]This is the rest",
    "reasoning": None,
    "content": "This is the rest",
    "is_reasoning_end": False,
}
INVALID_SHORTEST_REASONING = {
    "output": "[/THINK]This is the rest",
    "reasoning": None,
    "content": "This is the rest",
    "is_reasoning_end": False,
}
REASONING_WITH_THINK = {
    "output": "[THINK]This is a reasoning section[/THINK]This is the rest",
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
    "is_reasoning_end": True,
}
COMPLETE_REASONING_WITH_THINK = {
    "output": "[THINK]This is a reasoning section[/THINK]",
    "reasoning": "This is a reasoning section",
    "content": None,
    "is_reasoning_end": True,
}
MULTIPLE_LINES_WITH_THINK = {
    "output": "[THINK]This\nThat[/THINK]This is the rest\nThat",
    "reasoning": "This\nThat",
    "content": "This is the rest\nThat",
    "is_reasoning_end": True,
}
INVALID_SHORTEST_REASONING_NO_STREAMING_WITH_THINK = {
    "output": "[/THINK]This is the rest",
    "reasoning": None,
    "content": "This is the rest",
    "is_reasoning_end": False,
}
INVALID_SHORTEST_REASONING_WITH_THINK = {
    "output": "[/THINK]This is the rest",
    "reasoning": None,
    "content": "This is the rest",
    "is_reasoning_end": False,
}
THINK_NO_END = {
    "output": "[THINK]This is a reasoning section",
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
EMPTY_STREAMING = {
    "output": "",
    "reasoning": None,
    "content": None,
    "is_reasoning_end": False,
}
NEW_LINE = {
    "output": "Before\n[THINK]This is a reasoning section[/THINK]\nThis is the rest",
    "reasoning": "This is a reasoning section",
    "content": "Before\n\nThis is the rest",
    "is_reasoning_end": True,
}
NEW_LINE_STREAMING = {
    "output": "Before\n[THINK]This is a reasoning section[/THINK]\nThis is the rest",
    "reasoning": "This is a reasoning section",
    "content": "Before\n\nThis is the rest",
    "is_reasoning_end": True,
}

TEST_CASES = [
    pytest.param(
        False,
        INVALID_SIMPLE_REASONING,
        id="invalid_simple_reasoning",
    ),
    pytest.param(
        True,
        INVALID_SIMPLE_REASONING,
        id="invalid_simple_reasoning_streaming",
    ),
    pytest.param(
        False,
        INVALID_COMPLETE_REASONING,
        id="invalid_complete_reasoning",
    ),
    pytest.param(
        True,
        INVALID_COMPLETE_REASONING,
        id="invalid_complete_reasoning_streaming",
    ),
    pytest.param(
        False,
        NO_CONTENT,
        id="no_content",
    ),
    pytest.param(
        False,
        NO_REASONING,
        id="no_reasoning",
    ),
    pytest.param(
        True,
        NO_REASONING_STREAMING,
        id="no_reasoning_token_streaming",
    ),
    pytest.param(
        False,
        INVALID_MULTIPLE_LINES,
        id="invalid_multiple_lines",
    ),
    pytest.param(
        True,
        INVALID_MULTIPLE_LINES,
        id="invalid_multiple_lines_streaming",
    ),
    pytest.param(
        True,
        INVALID_SHORTEST_REASONING,
        id="invalid_shortest",
    ),
    pytest.param(
        False,
        INVALID_SHORTEST_REASONING_NO_STREAMING,
        id="invalid_shortest_streaming",
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
        INVALID_SHORTEST_REASONING_NO_STREAMING_WITH_THINK,
        id="invalid_shortest_with_think",
    ),
    pytest.param(
        True,
        INVALID_SHORTEST_REASONING_WITH_THINK,
        id="invalid_shortest_with_think_streaming",
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
def test_mistral_reasoning(
    streaming: bool,
    param_dict: dict,
    mistral_tokenizer: MistralTokenizer,
):
    output = param_dict["output"]

    index_think = output.find("[THINK]")
    len_think = len("[THINK]")
    index_end_think = output.find("[/THINK]")
    len_end_think = len("[/THINK]")

    # encode everything to tokens ids
    output_tokens = []
    if index_think != -1:
        output_before_think = output[:index_think]
        output_tokens += mistral_tokenizer.tokenizer.encode(
            output_before_think, False, False
        )
        output_tokens += [mistral_tokenizer.instruct.BEGIN_THINK]

        if index_end_think != -1:
            output_middle = output[index_think + len_think : index_end_think]
            output_after_think = output[index_end_think + len_end_think :]
            output_tokens += mistral_tokenizer.tokenizer.encode(
                output_middle, False, False
            )
            output_tokens += [mistral_tokenizer.instruct.END_THINK]
            output_tokens += mistral_tokenizer.tokenizer.encode(
                output_after_think, False, False
            )
        else:
            output_middle = output[index_think + len_think :]
            output_tokens += mistral_tokenizer.tokenizer.encode(
                output_middle, False, False
            )
    elif index_end_think != -1:
        output_before_think = output[:index_end_think]
        output_after_think = output[index_end_think + len_end_think :]
        output_tokens += mistral_tokenizer.tokenizer.encode(
            output_before_think, False, False
        )
        output_tokens += [mistral_tokenizer.instruct.END_THINK]
        output_tokens += mistral_tokenizer.tokenizer.encode(
            output_after_think, False, False
        )
    else:
        output_tokens += mistral_tokenizer.tokenizer.encode(output, False, False)

    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        mistral_tokenizer
    )

    reasoning, content = run_reasoning_extraction_mistral(
        parser, output_tokens, streaming=streaming
    )

    assert reasoning == param_dict["reasoning"]
    assert content == param_dict["content"]

    # Test is_reasoning_end
    is_reasoning_end = parser.is_reasoning_end(output_tokens)
    assert is_reasoning_end == param_dict["is_reasoning_end"]

    # Test extract_content
    if param_dict["content"] is not None:
        # Handle the case where there are tokens outputted before Thinking.
        # This should not occur if the model is well trained and prompted.
        if "[THINK]" in param_dict["output"] and not param_dict["output"].startswith(
            "[THINK]"
        ):
            before_content = param_dict["output"].split("[THINK]")[0]
            before_token_ids = mistral_tokenizer.tokenizer.encode(
                before_content, bos=False, eos=False
            )
            left_to_encode = param_dict["content"][len(before_content) :]
        # Normal situation.
        else:
            before_token_ids = []
            left_to_encode = param_dict["content"]

        content_tokens = parser.extract_content_ids(output_tokens)
        expected_token_ids = before_token_ids + mistral_tokenizer.tokenizer.encode(
            left_to_encode, bos=False, eos=False
        )
        assert content_tokens == expected_token_ids
    else:
        content = parser.extract_content_ids(output_tokens)
        assert content == []
