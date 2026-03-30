# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoTokenizer

from tests.reasoning.test_deepseekr1_reasoning_parser import TEST_CASES
from tests.reasoning.utils import run_reasoning_extraction
from vllm.reasoning import ReasoningParser, ReasoningParserManager

parser_name = "pangu"
start_token = "<think>"
end_token = "</think>"

REASONING_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


@pytest.fixture(scope="module")
def pangu_tokenizer():
    return AutoTokenizer.from_pretrained(REASONING_MODEL_NAME, trust_remote_code=True)


@pytest.mark.parametrize("streaming, param_dict", TEST_CASES)
def test_reasoning(
    streaming: bool,
    param_dict: dict,
    pangu_tokenizer,
):
    output = pangu_tokenizer.tokenize(param_dict["output"])
    # decode everything to tokens
    output_tokens: list[str] = [
        pangu_tokenizer.convert_tokens_to_string([token]) for token in output
    ]
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        pangu_tokenizer
    )

    reasoning, content = run_reasoning_extraction(
        parser, output_tokens, streaming=streaming
    )

    assert reasoning == param_dict["reasoning"]
    assert content == param_dict["content"]

    # Test is_reasoning_end
    output_ids = pangu_tokenizer.convert_tokens_to_ids(output)
    parser.delta_token_ids = output_ids
    is_reasoning_end = parser.is_reasoning_end(output_ids)
    assert is_reasoning_end == param_dict["is_reasoning_end"]

    # Test extract_content
    if param_dict["content"] is not None:
        content = parser.extract_content_ids(output_ids)
        assert content == pangu_tokenizer.convert_tokens_to_ids(
            pangu_tokenizer.tokenize(param_dict["content"])
        )
    else:
        content = parser.extract_content_ids(output_ids)
        assert content == []


single_token_output = [
    "<think>",
    "Some ",
    "reasoning ",
    "content",
    "</think>",
    "Final ",
    "answer",
]
mutil_tokens_output = [
    "<think>This ",
    "is a ",
    "reasoning process ",
    "section</think>",
    "This is ",
    "the rest",
]

SIMPLE_REASONING = {
    "output": single_token_output,
    "reasoning": "Some reasoning content",
    "content": "Final answer",
    "is_reasoning_end": True,
}

MUTIL_TOKENS_REASONING = {
    "output": mutil_tokens_output,
    "reasoning": "This is a reasoning process section",
    "content": "This is the rest",
    "is_reasoning_end": True,
}

TEST_CASES_STREAMING = [
    pytest.param(
        True,
        SIMPLE_REASONING,
        id="simple_reasoning_streaming",
    ),
    pytest.param(
        True,
        MUTIL_TOKENS_REASONING,
        id="mutil_tokens_reasoning_streaming",
    ),
    pytest.param(
        False,
        SIMPLE_REASONING,
        id="simple_reasoning",
    ),
    pytest.param(
        False,
        MUTIL_TOKENS_REASONING,
        id="mutil_tokens_reasoning",
    ),
]


class TestPanguReasoningParserStreaming:
    """Test streaming functionality of PanguReasoningParser."""

    @pytest.mark.parametrize("streaming, param_dict", TEST_CASES_STREAMING)
    def test_pangu_reasoning_extraction(self, pangu_tokenizer, streaming, param_dict):
        """
        Test basic reasoning extraction in streaming modes.
        """
        parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(
            parser_name
        )(pangu_tokenizer)

        reasoning, content = run_reasoning_extraction(
            parser, param_dict["output"], streaming=streaming
        )

        assert reasoning == param_dict["reasoning"]
        assert content == param_dict["content"]
