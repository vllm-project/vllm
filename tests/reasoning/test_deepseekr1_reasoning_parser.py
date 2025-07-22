# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from mistral_common.tokens.tokenizers.base import SpecialTokens
from mistral_common.tokens.tokenizers.tekken import (SpecialTokenInfo,
                                                     Tekkenizer)
from transformers import AutoTokenizer

from tests.reasoning.utils import run_reasoning_extraction
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.transformers_utils.tokenizers.mistral import MistralTokenizer

deepseek_parser_name = "deepseek_r1"
mistral_parser_name = "mistral"
QWEN_REASONING_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


@pytest.fixture(scope="module")
def deepseek_r1_qwen_tokenizer():
    return AutoTokenizer.from_pretrained(QWEN_REASONING_MODEL_NAME)


@pytest.fixture(scope="module")
def mistral_tokenizer():
    # TODO(Julien): upon model release change to a tokenizer already configured.
    # =================================================================
    mistral_tokenizer = MistralTokenizer.from_pretrained(
        "mistralai/Devstral-Small-2507")
    assert isinstance(mistral_tokenizer.tokenizer, Tekkenizer)
    # Add think special tokens to the tokenizer
    mistral_tokenizer.tokenizer._all_special_tokens[35] = SpecialTokenInfo(
        SpecialTokenInfo(rank=35,
                         is_control=True,
                         token_str=SpecialTokens.begin_think.value))
    mistral_tokenizer.tokenizer._all_special_tokens[36] = SpecialTokenInfo(
        SpecialTokenInfo(rank=36,
                         is_control=True,
                         token_str=SpecialTokens.end_think.value))
    mistral_tokenizer.tokenizer._special_tokens_reverse_vocab = {
        k: v
        for k, v in
        mistral_tokenizer.tokenizer._special_tokens_reverse_vocab.items()
        if v not in {35, 36}
    }
    mistral_tokenizer.tokenizer._special_tokens_reverse_vocab[
        SpecialTokens.begin_think.value] = 35
    mistral_tokenizer.tokenizer._special_tokens_reverse_vocab[
        SpecialTokens.end_think.value] = 36
    mistral_tokenizer.instruct.BEGIN_THINK = 35
    mistral_tokenizer.instruct.END_THINK = 36
    # =================================================================
    return mistral_tokenizer


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
NO_CONTENT = {
    "output": "This is content",
    "reasoning_content": "This is content",
    "content": None,
    "is_reasoning_end": False,
}
NO_REASONING_STREAMING = {
    "output": "This is a reasoning section",
    "reasoning_content": "This is a reasoning section",
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
    "reasoning_content": "",
    "content": None,
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
# Streaming cannot handle new lines at the beginning of the output
# because we need to support <think>...</think> and </think>...
# We cannot know if the text before <think> is reasoning content
# or not.
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
        deepseek_r1_qwen_tokenizer.convert_tokens_to_string([token])
        for token in output
    ]
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(
        deepseek_parser_name)(deepseek_r1_qwen_tokenizer)

    reasoning, content = run_reasoning_extraction(parser,
                                                  output_tokens,
                                                  streaming=streaming)

    assert reasoning == param_dict["reasoning_content"]
    assert content == param_dict["content"]

    # Test is_reasoning_end
    output_ids = deepseek_r1_qwen_tokenizer.convert_tokens_to_ids(output)
    is_reasoning_end = parser.is_reasoning_end(output_ids)
    assert is_reasoning_end == param_dict["is_reasoning_end"]

    # Test extract_content
    if param_dict["content"] is not None:
        content = parser.extract_content_ids(output_ids)
        assert content == deepseek_r1_qwen_tokenizer.convert_tokens_to_ids(
            deepseek_r1_qwen_tokenizer.tokenize(param_dict["content"]))
    else:
        content = parser.extract_content_ids(output)
        assert content == []


@pytest.mark.parametrize("streaming, param_dict", TEST_CASES)
def test_mistral_reasoning(
    streaming: bool,
    param_dict: dict,
    mistral_tokenizer: MistralTokenizer,
):
    output = param_dict["output"]

    index_think = output.find("<think>")
    len_think = len("<think>")
    index_end_think = output.find("</think>")
    len_end_think = len("</think>")

    # encode everything to tokens ids
    output_tokens = []
    if index_think != -1:
        output_before_think = output[:index_think]
        output_tokens += mistral_tokenizer.tokenizer.encode(
            output_before_think, False, False)
        output_tokens += [mistral_tokenizer.instruct.BEGIN_THINK]

        if index_end_think != -1:
            output_middle = output[index_think + len_think:index_end_think]
            output_after_think = output[index_end_think + len_end_think:]
            output_tokens += mistral_tokenizer.tokenizer.encode(
                output_middle, False, False)
            output_tokens += [mistral_tokenizer.instruct.END_THINK]
            output_tokens += mistral_tokenizer.tokenizer.encode(
                output_after_think, False, False)
        else:
            output_middle = output[index_think + len_think:]
            output_tokens += mistral_tokenizer.tokenizer.encode(
                output_middle, False, False)
    elif index_end_think != -1:
        output_before_think = output[:index_end_think]
        output_after_think = output[index_end_think + len_end_think:]
        output_tokens += mistral_tokenizer.tokenizer.encode(
            output_before_think, False, False)
        output_tokens += [mistral_tokenizer.instruct.END_THINK]
        output_tokens += mistral_tokenizer.tokenizer.encode(
            output_after_think, False, False)
    else:
        output_tokens += mistral_tokenizer.tokenizer.encode(
            output, False, False)

    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(
        mistral_parser_name)(mistral_tokenizer)

    reasoning, content = run_reasoning_extraction(parser,
                                                  output_tokens,
                                                  streaming=streaming)

    assert reasoning == param_dict["reasoning_content"]
    assert content == param_dict["content"]

    # Test is_reasoning_end
    is_reasoning_end = parser.is_reasoning_end(output_tokens)
    assert is_reasoning_end == param_dict["is_reasoning_end"]

    # Test extract_content
    if param_dict["content"] is not None:
        content = parser.extract_content_ids(output_tokens)
        assert content == mistral_tokenizer.tokenizer.encode(
            param_dict["content"], bos=False, eos=False)
    else:
        content = parser.extract_content_ids(output_tokens)
        assert content == []
