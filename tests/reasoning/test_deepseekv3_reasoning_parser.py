# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoTokenizer

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from vllm.reasoning.deepseek_v3_reasoning_parser import DeepSeekV3ReasoningParser
from vllm.reasoning.identity_reasoning_parser import IdentityReasoningParser

REASONING_MODEL_NAME = "deepseek-ai/DeepSeek-V3.1"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(REASONING_MODEL_NAME)


@pytest.mark.parametrize(
    "thinking,expected_parser_type",
    [
        (True, DeepSeekR1ReasoningParser),
        (False, IdentityReasoningParser),
    ],
)
def test_parser_selection(tokenizer, thinking, expected_parser_type):
    parser = DeepSeekV3ReasoningParser(
        tokenizer, chat_template_kwargs={"thinking": thinking}
    )

    assert isinstance(parser._parser, expected_parser_type)


def test_identity_reasoning_parser_basic(tokenizer):
    parser = IdentityReasoningParser(tokenizer)

    # Test is_reasoning_end always returns True
    input_text = "This is some output"
    input_tokens = tokenizer.tokenize(input_text)
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    assert parser.is_reasoning_end(input_ids) is True
    assert parser.is_reasoning_end_streaming(input_ids, input_ids) is True

    # Test extract_content_ids returns all input_ids
    assert parser.extract_content_ids(input_ids) == input_ids

    # Test extract_reasoning returns (None, model_output)
    request = ChatCompletionRequest(model="test-model", messages=[], temperature=1.0)
    reasoning, content = parser.extract_reasoning(input_text, request)
    assert reasoning is None
    assert content == input_text

    # Test extract_reasoning_streaming returns DeltaMessage or None
    result = parser.extract_reasoning_streaming(
        previous_text="",
        current_text="Hello world",
        delta_text="Hello world",
        previous_token_ids=[],
        current_token_ids=input_ids,
        delta_token_ids=input_ids,
    )
    assert isinstance(result, DeltaMessage)
    assert result.content == "Hello world"

    # If delta_text is empty, should return None
    result_none = parser.extract_reasoning_streaming(
        previous_text="Hello world",
        current_text="Hello world",
        delta_text="",
        previous_token_ids=input_ids,
        current_token_ids=input_ids,
        delta_token_ids=[],
    )
    assert result_none is None
