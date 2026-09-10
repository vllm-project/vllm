# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
from transformers import AutoTokenizer

from tests.reasoning.utils import run_reasoning_extraction
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from vllm.reasoning.gigachat35_reasoning_parser import (
    GigaChat35ReasoningParser,
    GigaChat35ReasoningWithThinkingParser,
)
from vllm.reasoning.identity_reasoning_parser import IdentityReasoningParser

REASONING_MODEL_NAME = "ai-sage/GigaChat3.5-432B-A28B"

START_TOKEN = "<think>"
END_TOKEN = "</think>"

# The chat template opens <think> in the generation prompt, so the model
# output contains only the closing </think>.
FORCED_OPEN_OUTPUT = {
    "output": [
        "The user asks 2+2.",
        " Simple arithmetic.",
        END_TOKEN,
        "\n\n",
        "The answer is 4.",
    ],
    "reasoning": "The user asks 2+2. Simple arithmetic.",
    "content": "\n\nThe answer is 4.",
}

TRUNCATED_REASONING_OUTPUT = {
    "output": ["The user asks 2+2.", " Simple arithmetic"],
    "reasoning": "The user asks 2+2. Simple arithmetic",
    "content": None,
}

TOOL_CALL_OUTPUT = {
    "output": [
        "Need the weather tool.",
        END_TOKEN,
        "\n\n",
        '<｜GCML｜tool_calls>\n<｜GCML｜invoke name="get_weather">\n'
        '<｜GCML｜parameter name="city" string="true">Москва</｜GCML｜parameter>\n'
        "</｜GCML｜invoke>\n</｜GCML｜tool_calls>",
    ],
    "reasoning": "Need the weather tool.",
    "content": "\n\n<｜GCML｜tool_calls>\n<｜GCML｜invoke "
    'name="get_weather">\n<｜GCML｜parameter name="city" '
    'string="true">Москва</｜GCML｜parameter>\n'
    "</｜GCML｜invoke>\n</｜GCML｜tool_calls>",
}


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(REASONING_MODEL_NAME, trust_remote_code=True)


@pytest.mark.parametrize(
    "reasoning,expected_parser_type",
    [
        (True, DeepSeekR1ReasoningParser),
        (False, IdentityReasoningParser),
    ],
)
@pytest.mark.parametrize(
    "parser_cls",
    [GigaChat35ReasoningParser, GigaChat35ReasoningWithThinkingParser],
)
def test_parser_selection(tokenizer, parser_cls, reasoning, expected_parser_type):
    parser = parser_cls(
        tokenizer,
        chat_template_kwargs={
            "reasoning": reasoning,
        },
    )

    assert isinstance(parser._parser, expected_parser_type)


def test_default_parser_is_identity(tokenizer):
    # The base GigaChat 3.5 template defaults to reasoning=False.
    parser = GigaChat35ReasoningParser(tokenizer)

    assert isinstance(parser._parser, IdentityReasoningParser)


def test_thinking_default_parser_is_deepseekr1(tokenizer):
    # The GigaChat 3.5 Reasoning template always opens <think>.
    parser = GigaChat35ReasoningWithThinkingParser(tokenizer)

    assert isinstance(parser._parser, DeepSeekR1ReasoningParser)


def test_thinking_supports_structured_output(tokenizer):
    # Structured output manager uses the reasoning parser to check if the
    # reasoning content is ended before applying the grammar. The main function
    # used is is_reasoning_end. This test checks if the parser is able to
    # correctly identify the end of the reasoning content.

    # important to not pass chat_template_kwargs here as it is done in the
    # StructuredOutputManager
    parser = GigaChat35ReasoningWithThinkingParser(tokenizer)

    end_token_id = tokenizer.encode(END_TOKEN, add_special_tokens=False)[0]

    assert parser.is_reasoning_end([1, 2, 4, end_token_id])
    assert not parser.is_reasoning_end([1, 2, 4])
    assert parser.is_reasoning_end([1, 2, 4, end_token_id, 5])


@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.parametrize(
    "case",
    [FORCED_OPEN_OUTPUT, TRUNCATED_REASONING_OUTPUT, TOOL_CALL_OUTPUT],
    ids=["forced_open", "truncated_reasoning", "tool_call"],
)
def test_reasoning_extraction(tokenizer, case, streaming):
    parser = GigaChat35ReasoningWithThinkingParser(tokenizer)

    reasoning, content = run_reasoning_extraction(
        parser, case["output"], streaming=streaming
    )

    assert reasoning == case["reasoning"]
    assert content == case["content"]


@pytest.mark.parametrize("streaming", [True, False])
def test_identity_extraction(tokenizer, streaming):
    parser = GigaChat35ReasoningParser(
        tokenizer, chat_template_kwargs={"reasoning": False}
    )

    output = ["The answer", " is 4."]
    reasoning, content = run_reasoning_extraction(parser, output, streaming=streaming)

    assert reasoning is None
    assert content == "The answer is 4."
